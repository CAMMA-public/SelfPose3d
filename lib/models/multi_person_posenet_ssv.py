# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from cv2 import imshow

import torch
import torch.nn as nn
import utils.cameras as cameras
from copy import deepcopy
from models import pose_resnet
from models.cuboid_proposal_net_soft import CuboidProposalNetSoft
from models.pose_regression_net import PoseRegressionNet
from core.loss import PerJointMSELoss
from core.loss import PerJointL1Loss
from core.proposal import max_pool
from vedo import Volume, show
import torch.nn.functional as F
from utils.transforms import affine_transform_pts_cuda as do_transform

from scipy.optimize import linear_sum_assignment

from torchvision.utils import save_image

class MultiPersonPoseNetSSV(nn.Module):
    def __init__(self, backbone, cfg, attn=None):
        super(MultiPersonPoseNetSSV, self).__init__()
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.num_joints = cfg.NETWORK.NUM_JOINTS

        self.backbone = backbone
        self.WITH_ATTN = cfg.WITH_ATTN
        if self.WITH_ATTN:
            self.attn = attn
            self.attn_weight = cfg.ATTN_WEIGHT
            # for param in self.attn.parameters():
            #     param.requires_grad = False
        self.USE_L1 = cfg.USE_L1
        self.L1_WEIGHT = cfg.L1_WEIGHT
        self.L1_ATTN = cfg.L1_ATTN
        self.L1_EPOCH = cfg.TRAIN.L1_EPOCH
        self.height = list(cfg.NETWORK.IMAGE_SIZE)[1]
        self.width = list(cfg.NETWORK.IMAGE_SIZE)[0]

        self.use_root_gt = cfg.NETWORK.USE_GT
        self.train_only_2d = cfg.NETWORK.TRAIN_ONLY_2D
        self.root_id = cfg.DATASET.ROOTIDX
        self.dataset_name = cfg.DATASET.TEST_DATASET
        self.train_only_rootnet = cfg.NETWORK.TRAIN_ONLY_ROOTNET
        self.rootnet_train_synth = cfg.NETWORK.ROOTNET_TRAIN_SYNTH
        self.freeze_rootnet = cfg.NETWORK.FREEZE_ROOTNET
        self.number = 0
        self.eval_rootnet_only = cfg.EVAL_ROOTNET_ONLY
        self.single_aug_training_posenet = cfg.NETWORK.SINGLE_AUG_TRAINING_POSENET
        self.root_reg_loss = cfg.NETWORK.ROOT_CONSISTENCY_LOSS
        self.weight_root_syn = cfg.NETWORK.WEIGHT_ROOT_SYN
        self.weight_root_reg = cfg.NETWORK.WEIGHT_ROOT_REG

        if self.train_only_2d:
            self.use_root_gt = True
        else:
            if not self.train_only_rootnet:
                self.pose_net = PoseRegressionNet(cfg)
                # for param in self.pose_net.parameters():
                #     param.requires_grad = False

        if not self.use_root_gt:
            self.root_net = CuboidProposalNetSoft(cfg)
            # for param in self.root_net.parameters():
            #     param.requires_grad = False

        self.heatmap_width = cfg.NETWORK.HEATMAP_SIZE[0]
        self.heatmap_height = cfg.NETWORK.HEATMAP_SIZE[1]
        self.pose_net_cube_size = cfg.PICT_STRUCT.CUBE_SIZE
        self.pose_net_cube_size = cfg.PICT_STRUCT.CUBE_SIZE
        self.rootnet_cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.rootnet_roothm = cfg.NETWORK.ROOTNET_ROOTHM
        if self.rootnet_roothm:
            self.rootnet_num_joints = 1
        else:
            self.rootnet_num_joints = cfg.NETWORK.NUM_JOINTS
        self.posenet_num_joints = cfg.NETWORK.NUM_JOINTS
        xx = torch.tensor([i for i in range(self.heatmap_width)]).to(torch.float32)
        yy = torch.tensor([i for i in range(self.heatmap_height)]).to(torch.float32)
        yy, xx = torch.meshgrid(yy, xx)
        xx, yy = xx.view(1, 1, *xx.shape), yy.view(1, 1, *yy.shape)
        zero_tensor_posenet = torch.zeros(
            cfg.TRAIN.BATCH_SIZE,
            self.posenet_num_joints,
            self.pose_net_cube_size[0],
            self.pose_net_cube_size[1],
            self.pose_net_cube_size[2],
        )
        self.init_train_epochs_rootnet = cfg.NETWORK.INIT_TRAIN_EPOCHS_ROOTNET
        self.register_buffer("hm_xx", xx, persistent=False)
        self.register_buffer("hm_yy", yy, persistent=False)
        self.register_buffer("zero_tensor_posenet", zero_tensor_posenet, persistent=False)


    # make forward for train and test
    def do_inference(self, views=None, meta=None, input_heatmaps=None, visualize_attn=False):
        if views is not None:
            all_heatmaps = []
            for view in views:
                heatmaps = self.backbone(view)
                all_heatmaps.append(heatmaps)
        else:
            all_heatmaps = input_heatmaps

        if visualize_attn:
            if views is not None:
                attns = []
                for view in views:
                    attns.append(self.attn(view))
                attns = torch.stack(attns, 0)

        device = all_heatmaps[0].device
        batch_size = all_heatmaps[0].shape[0]

        if self.use_root_gt:
            num_person = meta[0]["num_person"]
            grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)
            grid_centers[:, :, 0:3] = meta[0]["roots_3d"].float()
            grid_centers[:, :, 3] = -1.0
            for i in range(batch_size):
                grid_centers[i, : num_person[i], 3] = torch.tensor(range(num_person[i]), device=device)
                grid_centers[i, : num_person[i], 4] = 1.0
        else:
            _, _, _, grid_centers = self.root_net(all_heatmaps, meta)

        pred = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
        pred[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)

        if self.eval_rootnet_only:
            return pred, all_heatmaps, grid_centers

        if not self.train_only_rootnet:
            if not self.train_only_2d:
                for n in range(self.num_cand):
                    index = pred[:, n, 0, 3] >= 0
                    if torch.sum(index) > 0:
                        single_pose = self.pose_net(all_heatmaps, meta, grid_centers[:, n])
                        pred[:, n, :, 0:3] = single_pose.detach()
                        del single_pose

        if visualize_attn:
            return pred, all_heatmaps, grid_centers, attns
        else:
            return pred, all_heatmaps, grid_centers

    def l1_matching_loss(self, pred, meta):
        # meta[0]['joints'].shape: [batch_size, num_person, num_joint, 2]
        num_batch = len(pred[0])
        device = pred[0][0].device
        num_view = len(meta)

        losses = torch.zeros(num_view * num_batch, device=device)
        for nv in range(num_view):
            for bs in range(num_batch):
                num_gt = (meta[nv]['joints'][bs].sum(-1).sum(-1)!=0).sum()
                num_pred = len(pred[nv][bs])
                if num_pred == 0 or num_gt == 0:
                    continue
                d_matrix = torch.zeros(num_gt, num_pred, device=device)

                target = meta[nv]['joints'][bs][:num_gt]
                target_vis = meta[nv]['joints_vis'][bs][:num_gt]
                one_pred = pred[nv][bs][:]

                target[:,:,0] = target[:,:,0] / self.width
                target[:,:,1] = target[:,:,1] / self.height
                one_pred[:,:,0] = one_pred[:,:,0] / self.width
                one_pred[:,:,1] = one_pred[:,:,1] / self.height

                for t_n in range(num_gt):
                    for p_n in range(num_pred):
                        d_matrix[t_n, p_n] = ((one_pred[p_n] - target[t_n]) * target_vis[t_n]).abs().mean()
                matches_x, matches_y = linear_sum_assignment(d_matrix[:].cpu().detach()) # Hungarian matching
                best_loss = d_matrix[matches_x, matches_y].sum()
                idx = nv * num_batch + bs
                losses[idx] = best_loss
        
        if self.L1_ATTN:
            mask = torch.ones(num_view * num_batch, device=device)
            mask[torch.argmax(losses)] = 0.
            final_losses = (losses * mask).sum() / (num_batch * num_view - 1)
        else:
            final_losses = losses.mean()

        return final_losses


    def forward(
        self,
        views1=None,
        meta1=None,
        targets_2d1=None,
        weights_2d1=None,
        targets_3d1=None,
        input_heatmaps1=None,
        views2=None,
        meta2=None,
        targets_2d2=None,
        weights_2d2=None,
        targets_3d2=None,
        input_heatmaps2=None,
        views3=None,
        meta3=None,
        targets_2d3=None,
        weights_2d3=None,
        targets_3d3=None,
        input_heatmaps3=None,
        inference=False,
        visualize_attn=False,
        epoch=0,
    ):
        if inference:
            return self.do_inference(views=views1, meta=meta1, input_heatmaps=input_heatmaps1, visualize_attn=visualize_attn)
        FLIP_LR_JOINTS15 = [0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8]

        # view3 is only for root_net training, it won't go through affine augmentation
        if views3 is not None:
            all_heatmaps3 = []
            for view in views3:
                heatmaps3 = self.backbone(view)
                all_heatmaps3.append(heatmaps3)
        else:
            all_heatmaps3 = input_heatmaps3

        if self.WITH_ATTN:
            if views1 is not None:
                attns1 = []
                for view in views1:
                    attns1.append(self.attn(view))
                attns1 = torch.stack(attns1, 0)
            if views2 is not None:
                attns2 = []
                for view in views2:
                    attns2.append(self.attn(view))
                attns2 = torch.stack(attns2, 0)
            if views1 is not None:
                all_heatmaps1 = []
                for view in views1:
                    heatmaps = self.backbone(view)
                    all_heatmaps1.append(heatmaps)
            else:
                all_heatmaps1 = input_heatmaps1
            
            if views2 is not None:
                all_heatmaps2 = []
                for view in views2:
                    heatmaps2 = self.backbone(view)
                    all_heatmaps2.append(heatmaps2)
            else:
                all_heatmaps2 = input_heatmaps2
        else:
            if views1 is not None:
                all_heatmaps1 = []
                for view in views1:
                    heatmaps = self.backbone(view)
                    all_heatmaps1.append(heatmaps)
            else:
                all_heatmaps1 = input_heatmaps1
            
            if views2 is not None:
                all_heatmaps2 = []
                for view in views2:
                    heatmaps2 = self.backbone(view)
                    all_heatmaps2.append(heatmaps2)
            else:
                all_heatmaps2 = input_heatmaps2

        device = all_heatmaps1[0].device
        batch_size = views1[0].shape[0]

        losses = {}
        if targets_2d1 is not None and targets_2d2 is not None:
            targets_2d1 = torch.cat([t[None] for t in targets_2d1])
            targets_2d2 = torch.cat([t[None] for t in targets_2d2])
            targets_2d3 = torch.cat([t[None] for t in targets_2d3])
            loss_2d1 = F.mse_loss(targets_2d1, torch.cat([a[None] for a in all_heatmaps1]))
            loss_2d2 = F.mse_loss(targets_2d2, torch.cat([a[None] for a in all_heatmaps2]))
            loss_2d3 = F.mse_loss(targets_2d3, torch.cat([a[None] for a in all_heatmaps3]))
            losses["loss_2d"] = (loss_2d1 + loss_2d2 + loss_2d3) / 3.0
        else:
            losses["loss_2d"] = self.backbone(torch.zeros(1, 3, 512, 960, device=device)).mean() * 0.0
        # return None, all_heatmaps3, None, losses

        # fix later
        if self.train_only_2d:
            return None, all_heatmaps3, None, losses

        if self.use_root_gt:
            num_person = meta3[0]["num_person"]
            grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=device)
            grid_centers[:, :, 0:3] = meta3[0]["roots_3d"].float()
            grid_centers[:, :, 3] = -1.0
            for i in range(batch_size):
                grid_centers[i, : num_person[i], 3] = torch.tensor(range(num_person[i]), device=device)
                grid_centers[i, : num_person[i], 4] = 1.0
        else:
            if self.freeze_rootnet:
                _, _, _, grid_centers = self.root_net(all_heatmaps3, meta3, flip_xcoords=meta3[0]["hflip"])
            else:
                if self.rootnet_train_synth:
                    root_cubes_main1, root_cubes_syn1, target_cubes1, _ = self.root_net(
                        all_heatmaps1, meta1, flip_xcoords=meta1[0]["hflip"]
                    )
                    root_cubes_main2, root_cubes_syn2, target_cubes2, _ = self.root_net(
                        all_heatmaps2, meta2, flip_xcoords=meta2[0]["hflip"]
                    )
                    root_cubes_main3, root_cubes_syn3, target_cubes3, grid_centers = self.root_net(
                        all_heatmaps3, meta3, flip_xcoords=meta3[0]["hflip"]
                    )
                    loss_root_syn = (
                        F.mse_loss(root_cubes_syn1, target_cubes1)
                        + F.mse_loss(root_cubes_syn2, target_cubes2)
                        + F.mse_loss(root_cubes_syn3, target_cubes3)
                    )
                    root_cubes_main3 = root_cubes_main3.detach()
                    loss_root_reg = F.mse_loss(root_cubes_main1, root_cubes_main3) + F.mse_loss(
                        root_cubes_main2, root_cubes_main3
                    )
                    losses["loss_root_syn"] = self.weight_root_syn * loss_root_syn
                    if self.root_reg_loss:
                        losses["loss_root_reg"] = self.weight_root_reg * loss_root_reg
                else:
                    root_cubes1, _, _, _ = self.root_net(all_heatmaps1, meta1, flip_xcoords=meta1[0]["hflip"])
                    root_cubes2, _, _, _ = self.root_net(all_heatmaps2, meta2, flip_xcoords=meta2[0]["hflip"])
                    _, _, _, grid_centers = self.root_net(all_heatmaps3, meta3, flip_xcoords=meta3[0]["hflip"])
                    losses["loss_root_reg"] = F.mse_loss(root_cubes1, targets_3d1) + F.mse_loss(root_cubes2, targets_3d2)

        if self.train_only_rootnet:
            return None, all_heatmaps3, grid_centers, losses

        if epoch >= self.init_train_epochs_rootnet:
            if self.single_aug_training_posenet:
                loss_pose3d_ssv1 = F.mse_loss(torch.zeros(1, device=device), torch.zeros(1, device=device))
                pred1 = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
                pred1[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)
            else:
                loss_pose3d_ssv1 = F.mse_loss(torch.zeros(1, device=device), torch.zeros(1, device=device))
                loss_pose3d_ssv2 = F.mse_loss(torch.zeros(1, device=device), torch.zeros(1, device=device))
                pred1 = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
                pred2 = torch.zeros(batch_size, self.num_cand, self.num_joints, 5, device=device)
                pred1[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)
                pred2[:, :, :, 3:] = grid_centers[:, :, 3:].reshape(batch_size, -1, 1, 2)

            if self.single_aug_training_posenet:
                for n in range(self.num_cand):
                    index1 = pred1[:, n, 0, 3] >= 0
                    if torch.sum(index1) > 0:
                        single_pose1 = self.pose_net(
                            all_heatmaps1,
                            meta1,
                            grid_centers[:, n],
                            flip_xcoords=meta1[0]["hflip"],
                        )
                        pred1[:, n, :, 0:3] = single_pose1               
            else:
                for n in range(self.num_cand):
                    index1 = pred1[:, n, 0, 3] >= 0
                    index2 = pred2[:, n, 0, 3] >= 0
                    if torch.sum(index1) > 0:
                        single_pose1 = self.pose_net(
                            all_heatmaps1,
                            meta1,
                            grid_centers[:, n],
                            flip_xcoords=meta1[0]["hflip"],
                        )
                        pred1[:, n, :, 0:3] = single_pose1
                    if torch.sum(index2) > 0:
                        single_pose2 = self.pose_net(
                            all_heatmaps2,
                            meta2,
                            grid_centers[:, n],
                            flip_xcoords=meta2[0]["hflip"],
                        )
                        pred2[:, n, :, 0:3] = single_pose2

            if self.single_aug_training_posenet:
                pred2_out = pred1.detach().clone()
                pred1 = [pred1[pp, 0 : (grid_centers[pp, ..., 3] >= 0).sum().item(), :, :3] for pp in range(pred1.shape[0])]
                proj_cameras = [deepcopy(c["camera"]) for c in meta1]
                trans1 = meta1[0]["trans"]
            else:
                # 1.0 project 3d coords to 2D on all the planes -> 2D coords for each view
                # pred1 -> project to MV2, pred2 -> project to MV1
                # compute the mse loss
                pred2_out = pred2.detach().clone()
                pred1 = [pred1[pp, 0 : (grid_centers[pp, ..., 3] >= 0).sum().item(), :, :3] for pp in range(pred1.shape[0])]
                pred2 = [pred2[pp, 0 : (grid_centers[pp, ..., 3] >= 0).sum().item(), :, :3] for pp in range(pred2.shape[0])]
                # for pp in pred2:
                #     pp = pp[:, FLIP_LR_JOINTS15, :]
                #     pp[..., 0] = -pp[..., 0]
                # 3D to multiview
                proj_cameras = [deepcopy(c["camera"]) for c in meta1]
                trans1 = meta1[0]["trans"]
                trans2 = meta2[0]["trans"]
                # kps_2d_11 = [cameras.project_pose_batch(pred1, cam, trans1) for cam in cameras1] # not that interesting
                # kps_2d_22 = [cameras.project_pose_batch(pred2, cam, trans2) for cam in cameras2] # not that interesting
                # print("pred1", [p.shape for p in pred1])
                # print("pred2", [p.shape for p in pred2])
            
            if self.single_aug_training_posenet:
                if pred1[0].shape[0] > 0:
                    kps_2d_11 = [cameras.project_pose_batch(pred1, cam, trans1) for cam in proj_cameras]    
                    heatmaps_all_11 = []
                    for kps_views1 in kps_2d_11:
                        hm_b = []
                        for i_b, kps_batch in enumerate(kps_views1):
                            x = kps_batch[..., 0].view(-1, kps_batch[..., 0].shape[-1], 1, 1) / 4.0
                            y = kps_batch[..., 1].view(-1, kps_batch[..., 1].shape[-1], 1, 1) / 4.0
                            xx, yy = self.hm_xx.clone(), self.hm_yy.clone()
                            heatmaps = torch.exp(-(((xx - x) / 3.0) ** 2) / 2 - (((yy - y) / 3.0) ** 2) / 2)
                            heatmaps = torch.clip(torch.sum(heatmaps, 0), min=0.0, max=1.0)[None]
                            hm_b.append(heatmaps)
                        heatmaps_all_11.append(torch.cat(hm_b, 0)[None])     
                    heatmaps_all_11 = torch.cat(heatmaps_all_11, 0)

                    if targets_2d1 is not None:
                        loss_pose3d_ssv1 = F.mse_loss(targets_2d1, heatmaps_all_11)
                    losses["loss_pose3d_ssv"] = loss_pose3d_ssv1                                                    
                else:
                    losses["loss_pose3d_ssv"] = self.pose_net.v2v_net(self.zero_tensor_posenet).mean() * 0.0
            else:
                if pred1[0].shape[0] > 0 and pred2[0].shape[0] > 0:
                    kps_2d_12 = [
                        cameras.project_pose_batch(pred1, cam, trans2) for cam in proj_cameras
                    ]  # project the 3D poses to MV2
                    kps_2d_21 = [
                        cameras.project_pose_batch(pred2, cam, trans1) for cam in proj_cameras
                    ]  # project the 3D poses to MV1
                    # 2.0 check 2D coords for each view with ground truth (easy check; i guess it is good)
                    # 3.0 generate heatmaps from these coords (see sspose) I hope this is differential
                    
                    heatmaps_all_21, heatmaps_all_12 = [], []
                    for kps_views1 in kps_2d_21:
                        hm_b = []
                        for i_b, kps_batch in enumerate(kps_views1):
                            x = kps_batch[..., 0].view(-1, kps_batch[..., 0].shape[-1], 1, 1) / 4.0
                            y = kps_batch[..., 1].view(-1, kps_batch[..., 1].shape[-1], 1, 1) / 4.0
                            xx, yy = self.hm_xx.clone(), self.hm_yy.clone()
                            heatmaps = torch.exp(-(((xx - x) / 3.0) ** 2) / 2 - (((yy - y) / 3.0) ** 2) / 2)
                            heatmaps = torch.clip(torch.sum(heatmaps, 0), min=0.0, max=1.0)[None]
                            hm_b.append(heatmaps)
                        heatmaps_all_21.append(torch.cat(hm_b, 0)[None])

                    for kps_views2 in kps_2d_12:
                        hm_b = []
                        for kps_batch in kps_views2:
                            xx, yy = self.hm_xx.clone(), self.hm_yy.clone()
                            x = kps_batch[..., 0].view(-1, kps_batch[..., 0].shape[-1], 1, 1) / 4.0
                            y = kps_batch[..., 1].view(-1, kps_batch[..., 1].shape[-1], 1, 1) / 4.0
                            heatmaps = torch.exp(-(((xx - x) / 3.0) ** 2) / 2 - (((yy - y) / 3.0) ** 2) / 2)
                            heatmaps = torch.clip(torch.sum(heatmaps, 0), min=0.0, max=1.0)[None]
                            hm_b.append(heatmaps)
                        heatmaps_all_12.append(torch.cat(hm_b, 0)[None])

                    heatmaps_all_21 = torch.cat(heatmaps_all_21, 0)
                    heatmaps_all_12 = torch.cat(heatmaps_all_12, 0)

                    if targets_2d1 is not None:
                        if self.WITH_ATTN:
                            loss_pose3d_ssv1 = (F.mse_loss(targets_2d1, heatmaps_all_21, reduction='none') * attns1).mean()
                        else:
                            loss_pose3d_ssv1 = F.mse_loss(targets_2d1, heatmaps_all_21)
                    if targets_2d2 is not None:
                        if self.WITH_ATTN:
                            loss_pose3d_ssv2 = (F.mse_loss(targets_2d2, heatmaps_all_12, reduction='none') * attns2).mean()
                        else:
                            loss_pose3d_ssv2 = F.mse_loss(targets_2d2, heatmaps_all_12)
                    losses["loss_pose3d_ssv"] = loss_pose3d_ssv1 + loss_pose3d_ssv2
                    
                    if self.WITH_ATTN:
                        attns1_gt = torch.ones_like(attns1, device=device)
                        attns2_gt = torch.ones_like(attns2, device=device)
                        # attns1.shape: [5,1,15,128,240]
                        losses['loss_attn_ssv'] = (F.mse_loss(attns1, attns1_gt) + F.mse_loss(attns2, attns2_gt)) * self.attn_weight
                    
                    if self.USE_L1 and epoch >= self.L1_EPOCH:
                        losses['loss_pose3d_l1_ssv'] = (self.l1_matching_loss(kps_2d_12, meta2) + self.l1_matching_loss(kps_2d_21, meta1)) * self.L1_WEIGHT
                else:
                    if self.WITH_ATTN:
                        attns1_gt = torch.ones_like(attns1, device=device)
                        attns2_gt = torch.ones_like(attns2, device=device)
                        # attns1.shape: [5,1,15,128,240]
                        losses['loss_attn_ssv'] = (F.mse_loss(attns1, attns1_gt) + F.mse_loss(attns2, attns2_gt)) * 0.0
                    if self.USE_L1 and epoch >= self.L1_EPOCH:
                        l1_losses = torch.zeros(5, device=device).mean()
                        losses['loss_pose3d_l1_ssv'] = l1_losses * 0.0
                    losses["loss_pose3d_ssv"] = self.pose_net.v2v_net(self.zero_tensor_posenet).mean() * 0.0
        else:
            pred2_out = None
            losses["loss_pose3d_ssv"] = self.pose_net.v2v_net(self.zero_tensor_posenet).mean() * 0.0

        return pred2_out, all_heatmaps3, grid_centers, losses


def get_multi_person_pose_net(cfg, is_train=True):
    if cfg.BACKBONE_MODEL:
        backbone = eval(cfg.BACKBONE_MODEL + ".get_pose_net")(cfg, is_train=is_train)
    else:
        backbone = None
    if cfg.WITH_ATTN:
        attn = eval(cfg.BACKBONE_MODEL + ".get_pose_attn_net")(cfg, is_train=is_train)
        model = MultiPersonPoseNetSSV(backbone, cfg, attn)
    else:
        model = MultiPersonPoseNetSSV(backbone, cfg)
    return model
