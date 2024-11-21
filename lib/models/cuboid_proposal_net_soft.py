# ------------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from models.v2v_net import V2VNet
from models.project_layer import ProjectLayer
from core.proposal import nms
import numpy as np
from copy import deepcopy
import utils.cameras as cameras
from vedo import Volume, show
import cv2


class ProposalLayerSoft(nn.Module):
    def __init__(self, cfg):
        super(ProposalLayerSoft, self).__init__()
        self.grid_size = torch.tensor(cfg.MULTI_PERSON.SPACE_SIZE)
        self.cube_size = torch.tensor(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)
        self.grid_center = torch.tensor(cfg.MULTI_PERSON.SPACE_CENTER)
        self.num_cand = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.root_id = cfg.DATASET.ROOTIDX
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.threshold = cfg.MULTI_PERSON.THRESHOLD

    def filter_proposal(self, topk_index, gt_3d, num_person):
        batch_size = topk_index.shape[0]
        cand_num = topk_index.shape[1]
        cand2gt = torch.zeros(batch_size, cand_num)

        for i in range(batch_size):
            cand = topk_index[i].reshape(cand_num, 1, -1)
            gt = gt_3d[i, : num_person[i]].reshape(1, num_person[i], -1)

            dist = torch.sqrt(torch.sum((cand - gt) ** 2, dim=-1))
            min_dist, min_gt = torch.min(dist, dim=-1)

            cand2gt[i] = min_gt
            cand2gt[i][min_dist > 500.0] = -1.0

        return cand2gt

    def get_real_loc(self, index):
        device = index.device
        cube_size = self.cube_size.to(device=device, dtype=torch.float)
        grid_size = self.grid_size.to(device=device)
        grid_center = self.grid_center.to(device=device)
        loc = index.float() / (cube_size - 1) * grid_size + grid_center - grid_size / 2.0
        return loc

    def forward(self, root_cubes, meta, grids):
        batch_size = root_cubes.shape[0]

        topk_values, topk_unravel_index = nms(root_cubes.detach(), self.num_cand)
        topk_unravel_index = self.get_real_loc(topk_unravel_index)

        grid_centers = torch.zeros(batch_size, self.num_cand, 5, device=root_cubes.device)
        grid_centers[:, :, 0:3] = topk_unravel_index
        grid_centers[:, :, 4] = topk_values

        grid_centers[:, :, 3] = (
            topk_values > self.threshold
        ).float() - 1.0  # if ground-truths are not available.

        return grid_centers


class CuboidProposalNetSoft(nn.Module):
    def __init__(self, cfg):
        super(CuboidProposalNetSoft, self).__init__()
        self.grid_size = cfg.MULTI_PERSON.SPACE_SIZE
        self.cube_size = cfg.MULTI_PERSON.INITIAL_CUBE_SIZE
        self.grid_center = cfg.MULTI_PERSON.SPACE_CENTER
        self.root_id = cfg.DATASET.ROOTIDX
        self.rootnet_roothm = cfg.NETWORK.ROOTNET_ROOTHM
        self.rootnet_train_synth = cfg.NETWORK.ROOTNET_TRAIN_SYNTH
        self.max_num_people = cfg.MULTI_PERSON.MAX_PEOPLE_NUM
        self.rootnet_syn_range = cfg.NETWORK.ROOTNET_SYN_RANGE

        self.project_layer = ProjectLayer(cfg)
        if self.rootnet_roothm:
            self.v2v_net = V2VNet(1, 1)
        else:
            self.v2v_net = V2VNet(cfg.NETWORK.NUM_JOINTS, 1)
        self.proposal_layer = ProposalLayerSoft(cfg)

        if self.rootnet_train_synth:
            self.cur_sigma = 200.0
            space_size = self.grid_size
            space_center = self.grid_center
            cube_size = self.cube_size
            grid1Dx = (
                np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
            )
            grid1Dy = (
                np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
            )
            grid1Dz = (
                np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]
            )
            self.min_x, self.max_x = grid1Dx.min() + self.rootnet_syn_range[0][0], grid1Dx.max() + self.rootnet_syn_range[0][1]
            self.min_y, self.max_y = grid1Dy.min() + self.rootnet_syn_range[1][0], grid1Dy.max() + self.rootnet_syn_range[1][1]
            self.min_z, self.max_z = grid1Dz.min() + self.rootnet_syn_range[2][0], grid1Dz.max() + self.rootnet_syn_range[2][1]
            target = np.zeros(
                (cfg.TRAIN.BATCH_SIZE, cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32
            )
            self.register_buffer(
                "grid1Dx", torch.from_numpy(grid1Dx).to(torch.float32), persistent=False
            )
            self.register_buffer(
                "grid1Dy", torch.from_numpy(grid1Dy).to(torch.float32), persistent=False
            )
            self.register_buffer(
                "grid1Dz", torch.from_numpy(grid1Dz).to(torch.float32), persistent=False
            )
            self.register_buffer("target", torch.from_numpy(target), persistent=False)
            self.heatmap_width = cfg.NETWORK.HEATMAP_SIZE[0]
            self.heatmap_height = cfg.NETWORK.HEATMAP_SIZE[1]
            xx = torch.tensor([i for i in range(self.heatmap_width)]).to(torch.float32)
            yy = torch.tensor([i for i in range(self.heatmap_height)]).to(torch.float32)
            yy, xx = torch.meshgrid(yy, xx)
            xx, yy = xx.view(1, 1, *xx.shape), yy.view(1, 1, *yy.shape)
            self.register_buffer("hm_xx", xx, persistent=False)
            self.register_buffer("hm_yy", yy, persistent=False)

    def get_grid_centres(self, all_heatmaps, meta, flip_xcoords):
        # with torch.no_grad():
        # self.v2v_net.eval()
        if self.rootnet_roothm:
            all_heatmaps_copy = [a[:, self.root_id, :, :][:, None].clone() for a in all_heatmaps]
        else:
            all_heatmaps_copy = all_heatmaps

        initial_cubes, grids = self.project_layer(
            all_heatmaps_copy,
            meta,
            self.grid_size,
            [self.grid_center],
            self.cube_size,
            flip_xcoords=flip_xcoords,
        )
        root_cubes = self.v2v_net(initial_cubes)
        root_cubes = root_cubes.squeeze(1)
        grid_centers = self.proposal_layer(root_cubes, meta, grids)

        return root_cubes, grid_centers

    def train_rootnet(self, batch_size, meta, pred_hms, flip_xcoords=False):
        with torch.no_grad():
            # step 1: Generate random 3D root points between 1 to self.max_num_people in the range [80, 80, 20] (with no torch grad)
            # step 2: Generate the target_cubes (with no torch grad)
            num_roots = torch.randint(1, self.max_num_people, (1,)).item()
            x_coords = (self.max_x - self.min_x) * torch.rand(batch_size, num_roots, 1) + self.min_x
            y_coords = (self.max_y - self.min_y) * torch.rand(batch_size, num_roots, 1) + self.min_y
            z_coords = (self.max_z - self.min_z) * torch.rand(batch_size, 1, 1) + self.min_z
            z_coords = z_coords.expand(1, num_roots, 1)
            z_coords = z_coords + torch.randn_like(z_coords) * 50
            rand_coords = torch.cat((x_coords, y_coords, z_coords), -1).to(
                device=self.grid1Dx.device, dtype=torch.float32
            )
            # num_roots = meta[0]["num_person"].item()
            # rand_coords = meta[0]["roots_3d"][:, 0:meta[0]["num_person"].item(), :].to(torch.float32)
            # print("num_roots:", num_roots)
            # print("rand_coords:", rand_coords)
            target_cubes = []
            for n_b in range(batch_size):
                target = self.target.clone()
                for n_r in range(num_roots):
                    mu_x, mu_y, mu_z = [p.item() for p in rand_coords[n_b][n_r]]
                    i_x = [
                        torch.searchsorted(self.grid1Dx, mu_x - 3 * self.cur_sigma),
                        torch.searchsorted(self.grid1Dx, mu_x + 3 * self.cur_sigma, right=True),
                    ]
                    i_y = [
                        torch.searchsorted(self.grid1Dy, mu_y - 3 * self.cur_sigma),
                        torch.searchsorted(self.grid1Dy, mu_y + 3 * self.cur_sigma, right=True),
                    ]
                    i_z = [
                        torch.searchsorted(self.grid1Dz, mu_z - 3 * self.cur_sigma),
                        torch.searchsorted(self.grid1Dz, mu_z + 3 * self.cur_sigma, right=True),
                    ]
                    if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                        continue
                    gridx, gridy, gridz = torch.meshgrid(
                        (
                            self.grid1Dx[i_x[0] : i_x[1]],
                            self.grid1Dy[i_y[0] : i_y[1]],
                            self.grid1Dz[i_z[0] : i_z[1]],
                        )
                    )
                    g = torch.exp(
                        -((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2)
                        / (2 * self.cur_sigma ** 2)
                    )
                    target[n_b, i_x[0] : i_x[1], i_y[0] : i_y[1], i_z[0] : i_z[1]] = torch.maximum(
                        target[n_b, i_x[0] : i_x[1], i_y[0] : i_y[1], i_z[0] : i_z[1]], g
                    )
                target = torch.clip(target, 0, 1)
                target_cubes.append(target)
            target_cubes = torch.cat(target_cubes, 0)
            # rand_indices = torch.randint(0, self.buffer_size, (batch_size,))
            # rand_coords = self.coords_buffer[rand_indices]
            # num_roots = self.num_people_buffer[rand_indices]
            # target_cubes = self.target_cubes[rand_indices]

            # step 3: Project these points to each camera and get the heatmaps (with no torch grad)
            center_pts = [rand_coords[ii][None] for ii in range(rand_coords.shape[0])]
            cams = [deepcopy(c["camera"]) for c in meta]
            trans = meta[0]["trans"]
            cps_2d = [cameras.project_pose_batch(center_pts, cam, trans) for cam in cams]
            heatmaps_all = []
            for cps_views in cps_2d:
                hm_b = []
                for i_b, cps_batch in enumerate(cps_views):
                    cps_batch = cps_batch.permute(1, 0, 2)
                    x = cps_batch[..., 0].view(-1, cps_batch[..., 0].shape[-1], 1, 1) / 4.0
                    y = cps_batch[..., 1].view(-1, cps_batch[..., 1].shape[-1], 1, 1) / 4.0
                    xx, yy = self.hm_xx.clone(), self.hm_yy.clone()
                    heatmaps = torch.exp(-(((xx - x) / 3.0) ** 2) / 2 - (((yy - y) / 3.0) ** 2) / 2)
                    heatmaps = torch.clip(torch.sum(heatmaps, 0), min=0.0, max=1.0)
                    rand_noise = 0.02 * torch.randn_like(heatmaps)
                    heatmaps = torch.clip(heatmaps + rand_noise, min=0.0, max=1.0)
                    hm_b.append(heatmaps)
                heatmaps_all.append(torch.cat(hm_b, 0)[None])
            # heatmaps_all = torch.cat(heatmaps_all, 0)

        # step 4: pass the heatmaps to the self.v2v_net to get the root_cubes
        initial_cubes, _ = self.project_layer(
            heatmaps_all,
            meta,
            self.grid_size,
            [self.grid_center],
            self.cube_size,
            flip_xcoords=flip_xcoords,
        )
        root_cubes = self.v2v_net(initial_cubes)
        root_cubes = root_cubes.squeeze(1)
        return root_cubes, target_cubes

    def forward(self, all_heatmaps, meta, flip_xcoords=None):
        if self.rootnet_train_synth:
            root_cubes_main, grid_centers = self.get_grid_centres(all_heatmaps, meta, flip_xcoords)
            if self.training:
                batch_size = all_heatmaps[0].shape[0]
                # self.v2v_net.train()
                root_cubes_syn, target_cubes = self.train_rootnet(
                    batch_size, meta, all_heatmaps, flip_xcoords
                )
                return root_cubes_main, root_cubes_syn, target_cubes, grid_centers
            else:
                return root_cubes_main, None, None, grid_centers
        else:
            if self.rootnet_roothm:
                all_heatmaps_copy = [
                    a[:, self.root_id, :, :][:, None].clone() for a in all_heatmaps
                ]
            else:
                all_heatmaps_copy = all_heatmaps

            initial_cubes, grids = self.project_layer(
                all_heatmaps_copy,
                meta,
                self.grid_size,
                [self.grid_center],
                self.cube_size,
                flip_xcoords=flip_xcoords,
            )

            root_cubes = self.v2v_net(initial_cubes)
            root_cubes = root_cubes.squeeze(1)
            grid_centers = self.proposal_layer(root_cubes, meta, grids)

            return root_cubes, None, None, grid_centers

