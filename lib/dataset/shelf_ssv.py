'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


import logging

from copy import deepcopy
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from PIL import Image
from utils.transforms import get_affine_transform
from utils.transforms import affine_transform, get_scale
from dataset.randaugment import RandAugment, Cutout
from vedo import Volume, show

import json_tricks as json
import pickle
from torchvision.utils import save_image


logger = logging.getLogger(__name__)

JOINTS_DEF = {
    "neck": 0,
    "nose": 1,
    "mid-hip": 2,
    "l-shoulder": 3,
    "l-elbow": 4,
    "l-wrist": 5,
    "l-hip": 6,
    "l-knee": 7,
    "l-ankle": 8,
    "r-shoulder": 9,
    "r-elbow": 10,
    "r-wrist": 11,
    "r-hip": 12,
    "r-knee": 13,
    "r-ankle": 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}

FLIP_LR_JOINTS15 = [0, 1, 2, 9, 10, 11, 12, 13, 14, 3, 4, 5, 6, 7, 8]

class RandomAugumnetCutOut:
    """Resize image to low res and transform back"""

    def __init__(self, apply_cutout=True):
        self.random_transform = RandAugment()
        self.apply_cutout = apply_cutout
        if self.apply_cutout:
            self.cutout = Cutout()

    def __call__(self, img):
        img = self.random_transform(img)
        if self.apply_cutout:
            num = np.random.randint(2, 16)
            for _ in range(num):
                img = self.cutout(img, np.random.randint(20, 40))
        return img


class shelf_ssv(Dataset):
    def __init__(self, cfg, image_set, is_train, transform=None):
        self.cfg = cfg
        self.num_joints = len(JOINTS_DEF)
        self.pixel_std = 200
        self.flip_pairs = []
        self.flip_indices = FLIP_LR_JOINTS15
        self.maximum_person = cfg.MULTI_PERSON.MAX_PEOPLE_NUM

        self.is_train = is_train

        this_dir = os.path.dirname(__file__)
        dataset_root = os.path.join(this_dir, "../..", cfg.DATASET.ROOT)
        self.dataset_root = os.path.abspath(dataset_root)
        self.root_id = cfg.DATASET.ROOTIDX
        self.image_set = image_set
        self.dataset_name = cfg.DATASET.TEST_DATASET

        self.data_format = cfg.DATASET.DATA_FORMAT
        self.data_augmentation = cfg.DATASET.DATA_AUGMENTATION

        self.cameras = cfg.DATASET.CAMERAS
        self.num_views = len(self.cameras)
        self.camera_num_total = cfg.DATASET.CAMERA_NUM_TOTAL

        self.scale_factor1 = cfg.DATASET.SCALE_FACTOR1
        self.scale_factor2 = cfg.DATASET.SCALE_FACTOR2
        self.rotation_factor1 = cfg.DATASET.ROT_FACTOR1
        self.rotation_factor2 = cfg.DATASET.ROT_FACTOR2
        self.flip = cfg.DATASET.FLIP

        if self.is_train:
            self.rand_augment = RandomAugumnetCutOut(apply_cutout=cfg.DATASET.APPLY_CUTOUT)
            self.apply_rand_aug = cfg.DATASET.APPLY_RANDAUG
        else: # during validation, no augmentation
            self.rand_augment = RandomAugumnetCutOut(apply_cutout=False)
            self.apply_rand_aug = False
        
        self.color_rgb = cfg.DATASET.COLOR_RGB
        image_size_orig = np.array(cfg.NETWORK.IMAGE_SIZE_ORIG)
        self.height_orig = image_size_orig[1]
        self.width_orig = image_size_orig[0]

        self.target_type = cfg.NETWORK.TARGET_TYPE
        self.image_size = np.array(cfg.NETWORK.IMAGE_SIZE)
        self.heatmap_size = np.array(cfg.NETWORK.HEATMAP_SIZE)
        self.sigma = cfg.NETWORK.SIGMA
        self.use_different_joints_weight = cfg.LOSS.USE_DIFFERENT_JOINTS_WEIGHT
        self.joints_weight = 1

        self.transform = transform
        self.db = []
        self.debug = False
        self.mis_count = 0

        self.space_size = np.array(cfg.MULTI_PERSON.SPACE_SIZE)
        self.space_center = np.array(cfg.MULTI_PERSON.SPACE_CENTER)
        self.initial_cube_size = np.array(cfg.MULTI_PERSON.INITIAL_CUBE_SIZE)
        self.min_views_check = cfg.MIN_VIEWS_CHECK

        self.db_file = "shelf_mmpose.pkl"
        self.db_file = os.path.join(self.dataset_root, self.db_file)

        if os.path.exists(self.db_file):
            print("=> loading the pickle file = ", self.db_file)
            info = pickle.load(open(self.db_file, "rb"))
            self.db = info["db"]
            for p in info["db"]:
                p["image"] = os.path.join("./data", p["image"])
            print("=> self.db", len(self.db))
        else:
            raise ValueError('DB file not found! ')
        self.db_size = len(self.db)

        self.cameras_param = self._get_cam()

    def _get_cam(self):
        cam_file = os.path.join(self.dataset_root, "calibration_shelf.json")
        with open(cam_file) as cfile:
            cameras = json.load(cfile)

        for id, cam in cameras.items():
            for k, v in cam.items():
                cameras[id][k] = np.array(v)

        return cameras
    
    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(
        self,
    ):
        return self.db_size // self.camera_num_total

    def __getitem__(self, idx):
        (
            inputs1,
            target_heatmaps1,
            target_weights1,
            target_heatmap_roots1,
            target_weight_roots1,
            target_3ds1,
            metas1,
            input_heatmaps1,
            inputs2,
            target_heatmaps2,
            target_weights2,
            target_heatmap_roots2,
            target_weight_roots2,
            target_3ds2,
            metas2,
            input_heatmaps2,
            inputs3,
            target_heatmaps3,
            target_weights3,
            target_heatmap_roots3,
            target_weight_roots3,
            target_3ds3,
            metas3,
            input_heatmaps3,
        ) = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        while True:
            # just compute the joints
            db_rec_list, trans1_list, trans2_list, trans3_list = [], [], [], []
            (
                joints1_list,
                joints2_list,
                joints3_list,
                joints_vis1_list,
                joints_vis2_list,
                joints_vis3_list,
            ) = (
                [],
                [],
                [],
                [],
                [],
                [],
            )
            r1 = (
                np.clip(
                    np.random.uniform(-1,1) * self.rotation_factor1,
                    -self.rotation_factor1,
                    self.rotation_factor1,
                )
                if random.random() <= 0.5
                else 0
            )
            r2 = (
                np.clip(
                    np.random.uniform(-1,1) * self.rotation_factor2,
                    -self.rotation_factor2,
                    self.rotation_factor2,
                )
                if random.random() <= 0.5
                else 0
            )
            if self.flip:
                do_hflip1 = random.random() <= 0.5
                do_hflip2 = random.random() <= 0.5
            else:
                do_hflip1, do_hflip2 = False, False

            if self.scale_factor1 == 0:
                s1 = 0.0
            else:
                s1 = (
                    np.random.uniform(0.1, self.scale_factor1)
                    if random.random() <= 0.5
                    else -np.random.uniform(0.1, self.scale_factor1) / 2.0
                )
            if self.scale_factor2 == 0:
                s2 = 0.0
            else:
                s2 = (
                    np.random.uniform(0.1, self.scale_factor2)
                    if random.random() <= 0.5
                    else -np.random.uniform(0.1, self.scale_factor2) / 2.0
                )
            npersons_list = []
            for k in range(self.num_views):
                index = self.camera_num_total * idx + self.cameras[k]
                db_rec = deepcopy(self.db[index])
                db_rec['camera'] = deepcopy(self.cameras_param[str(k)])
                db_rec["camera"]["f"] = np.array([db_rec["camera"]["fx"], db_rec["camera"]["fy"]])[
                    ..., None
                ]
                db_rec["camera"]["c"] = np.array([db_rec["camera"]["cx"], db_rec["camera"]["cy"]])[
                    ..., None
                ]
                for cam_key, cam_value in db_rec["camera"].items():
                    db_rec["camera"][cam_key] = torch.from_numpy(cam_value.astype(np.float32))
                db_rec_list.append(db_rec)

                joints1, joints2, joints3 = (
                    deepcopy(db_rec["joints_2d"]),
                    deepcopy(db_rec["joints_2d"]),
                    deepcopy(db_rec["joints_2d"]),
                )
                joints_vis1, joints_vis2, joints_vis3 = (
                    deepcopy(db_rec["joints_2d_vis"]),
                    deepcopy(db_rec["joints_2d_vis"]),
                    deepcopy(db_rec["joints_2d_vis"]),
                )

                nposes = len(joints1)
                npersons_list.append(nposes)

                height, width = self.height_orig, self.width_orig
                c = np.array([width / 2.0, height / 2.0])
                s = get_scale((width, height), self.image_size)
                sc1 = np.array([_s + (_s * s1) for _s in s])
                sc2 = np.array([_s + (_s * s2) for _s in s])
                trans1 = get_affine_transform(c, sc1, r1, self.image_size)
                trans2 = get_affine_transform(c, sc2, r2, self.image_size)
                trans3 = get_affine_transform(c, s, 0, self.image_size)
                trans1_list.append(trans1)
                trans2_list.append(trans2)
                trans3_list.append(trans3)

                # optmize this loop
                for n in range(nposes):
                    for i in range(len(joints1[0])):
                        if joints_vis1[n][i, 0] > 0.0:
                            joints1[n][i, 0:2] = affine_transform(joints1[n][i, 0:2], trans1)
                            joints2[n][i, 0:2] = affine_transform(joints2[n][i, 0:2], trans2)
                            joints3[n][i, 0:2] = affine_transform(joints3[n][i, 0:2], trans3)
                            if (
                                np.min(joints1[n][i, :2]) < 0
                                or joints1[n][i, 0] >= self.image_size[0]
                                or joints1[n][i, 1] >= self.image_size[1]
                            ):
                                joints_vis1[n][i, :] = 0
                            if (
                                np.min(joints2[n][i, :2]) < 0
                                or joints2[n][i, 0] >= self.image_size[0]
                                or joints2[n][i, 1] >= self.image_size[1]
                            ):
                                joints_vis2[n][i, :] = 0
                            if (
                                np.min(joints3[n][i, :2]) < 0
                                or joints3[n][i, 0] >= self.image_size[0]
                                or joints3[n][i, 1] >= self.image_size[1]
                            ):
                                joints_vis3[n][i, :] = 0

                    if do_hflip1:
                        joints1[n][..., 0:2] = joints1[n][..., 0:2][self.flip_indices]
                        joints1[n][..., 0] = self.image_size[0] - joints1[n][..., 0]

                    if do_hflip2:
                        joints2[n][..., 0:2] = joints2[n][..., 0:2][self.flip_indices]
                        joints2[n][..., 0] = self.image_size[0] - joints2[n][..., 0]

                joints1_list.append(joints1)
                joints2_list.append(joints2)
                joints3_list.append(joints3)
                joints_vis1_list.append(joints_vis1)
                joints_vis2_list.append(joints_vis2)
                joints_vis3_list.append(joints_vis3)
            # c1 c2 c3 ensure that all the views should have atleast one person
            c1 = np.all(np.array([len(p) for p in joints_vis1_list]) > 0)
            c2 = np.all(np.array([len(p) for p in joints_vis2_list]) > 0)
            c3 = np.all(np.array([len(p) for p in joints_vis3_list]) > 0)
            if c1 and c2 and c3: 
                roots1 = np.sort(
                    np.array(
                        [
                            np.any(np.array(p)[:, self.root_id], 1).astype(np.int32).sum()
                            for p in joints_vis1_list
                        ]
                    )
                )[-self.min_views_check :]
                roots2 = np.sort(
                    np.array(
                        [
                            np.any(np.array(p)[:, self.root_id], 1).astype(np.int32).sum()
                            for p in joints_vis2_list
                        ]
                    )
                )[-self.min_views_check :]
                min_vis_roots1 = roots1.sum() / self.min_views_check
                min_vis_roots2 = roots2.sum() / self.min_views_check
                npers_allviews = np.max(npersons_list)

                if int(npers_allviews) == int(min_vis_roots1) and int(npers_allviews) == int(
                    min_vis_roots2
                ):
                    break
                else:
                    idx = np.random.randint(0, (len(self) / self.num_views) - 10)
                    self.mis_count += 1
            else:
                idx = np.random.randint(0, (len(self) / self.num_views) - 10)
                self.mis_count += 1                


        for k in range(self.num_views):
            db_rec = db_rec_list[k]
            joints1, joints_vis1 = joints1_list[k], joints_vis1_list[k]
            joints2, joints_vis2 = joints2_list[k], joints_vis2_list[k]
            joints3, joints_vis3 = joints3_list[k], joints_vis3_list[k]
            trans1, trans2, trans3 = trans1_list[k], trans2_list[k], trans3_list[k]
            nposes = npersons_list[k]

            if "joints_3d" in db_rec:
                joints_3d = db_rec["joints_3d"]
                joints_3d_vis = db_rec["joints_3d_vis"]
                with_3d = True
                assert nposes <= self.maximum_person, "too many persons"
            else:
                with_3d = False

            if nposes > self.maximum_person:
                joints1 = [joints1[j] for j in range(self.maximum_person)]
                joints_vis1 = [joints_vis1[j] for j in range(self.maximum_person)]
                joints2 = [joints2[j] for j in range(self.maximum_person)]
                joints_vis2 = [joints_vis2[j] for j in range(self.maximum_person)]
                joints3 = [joints3[j] for j in range(self.maximum_person)]
                joints_vis3 = [joints_vis3[j] for j in range(self.maximum_person)]
                nposes = self.maximum_person

            image_file = db_rec["image"]
            data_numpy = cv2.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

            if data_numpy is None:
                return (None,) * 12

            if self.color_rgb:
                data_numpy = cv2.cvtColor(data_numpy, cv2.COLOR_BGR2RGB)

            input1 = cv2.warpAffine(
                data_numpy,
                trans1,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR,
            )
            input2 = cv2.warpAffine(
                data_numpy,
                trans2,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR,
            )
            input3 = cv2.warpAffine(
                data_numpy,
                trans3,
                (int(self.image_size[0]), int(self.image_size[1])),
                flags=cv2.INTER_LINEAR,
            )
            if do_hflip1:
                input1 = cv2.flip(input1, 1)
            if do_hflip2:
                input2 = cv2.flip(input2, 1)

            if self.apply_rand_aug:
                input1 = deepcopy(np.asarray(self.rand_augment(Image.fromarray(input1))))
                input2 = deepcopy(np.asarray(self.rand_augment(Image.fromarray(input2))))

            if "pred_pose2d" in db_rec and db_rec["pred_pose2d"] != None:
                # For convenience, we use predicted poses and corresponding values at the original heatmaps
                # to generate 2d heatmaps for Campus and Shelf dataset.
                # You can also use other 2d backbone trained on COCO to generate 2d heatmaps directly.
                pred_pose2d1, pred_pose2d2, pred_pose2d3 = (
                    deepcopy(db_rec["pred_pose2d"]),
                    deepcopy(db_rec["pred_pose2d"]),
                    deepcopy(db_rec["pred_pose2d"]),
                )
                for n in range(len(pred_pose2d1)):
                    for i in range(len(pred_pose2d1[n])):
                        pred_pose2d1[n][i, 0:2] = affine_transform(pred_pose2d1[n][i, 0:2], trans1)
                        pred_pose2d2[n][i, 0:2] = affine_transform(pred_pose2d2[n][i, 0:2], trans2)
                        pred_pose2d3[n][i, 0:2] = affine_transform(pred_pose2d3[n][i, 0:2], trans2)
                input_heatmap1 = self.generate_input_heatmap(pred_pose2d1)
                input_heatmap1 = torch.from_numpy(input_heatmap1)
                input_heatmap2 = self.generate_input_heatmap(pred_pose2d2)
                input_heatmap2 = torch.from_numpy(input_heatmap2)
                input_heatmap3 = self.generate_input_heatmap(pred_pose2d3)
                input_heatmap3 = torch.from_numpy(input_heatmap3)
            else:
                input_heatmap1 = torch.zeros(
                    self.cfg.NETWORK.NUM_JOINTS,
                    self.heatmap_size[1],
                    self.heatmap_size[0],
                )
                input_heatmap2 = torch.zeros(
                    self.cfg.NETWORK.NUM_JOINTS,
                    self.heatmap_size[1],
                    self.heatmap_size[0],
                )
                input_heatmap3 = torch.zeros(
                    self.cfg.NETWORK.NUM_JOINTS,
                    self.heatmap_size[1],
                    self.heatmap_size[0],
                )

            target_heatmap1, target_weight1 = self.generate_target_heatmap(joints1, joints_vis1)
            target_heatmap2, target_weight2 = self.generate_target_heatmap(joints2, joints_vis2)
            target_heatmap3, target_weight3 = self.generate_target_heatmap(joints3, joints_vis3)
            # target_heatmap1_root, target_weight1_root = self.generate_target_heatmap_roots(
            #     joints1, joints_vis1
            # )
            # target_heatmap2_root, target_weight2_root = self.generate_target_heatmap_roots(
            #     joints2, joints_vis2
            # )
            # target_heatmap3_root, target_weight3_root = self.generate_target_heatmap_roots(
            #     joints3, joints_vis3
            # )

            # visualize the 2D joints and heatmaps
            # if self.debug:
            # input1_vis = cv2.cvtColor(deepcopy(input1), cv2.COLOR_RGB2BGR)
            # input2_vis = cv2.cvtColor(deepcopy(input2), cv2.COLOR_RGB2BGR)
            # input3_vis = cv2.cvtColor(deepcopy(input3), cv2.COLOR_RGB2BGR)
            # heatmaps1_1 = [cv2.resize(cv2.applyColorMap((m*255).astype(np.uint8), cv2.COLORMAP_JET), (960, 512)) for m in target_heatmap1]
            # heatmaps2_1 = [cv2.resize(cv2.applyColorMap((m*255).astype(np.uint8), cv2.COLORMAP_JET), (960, 512)) for m in target_heatmap2]
            # heatmaps3_1 = [cv2.resize(cv2.applyColorMap((m*255).astype(np.uint8), cv2.COLORMAP_JET), (960, 512)) for m in target_heatmap3]
            # heatmaps1 = [((input1_vis*0.3) + (m*0.7)).astype(np.uint8) for m in heatmaps1_1]
            # heatmaps2 = [((input2_vis*0.3) + (m*0.7)).astype(np.uint8) for m in heatmaps2_1]
            # heatmaps3 = [((input3_vis*0.3) + (m*0.7)).astype(np.uint8) for m in heatmaps3_1]
            # for n in range(nposes):
            #     for i in range(len(joints1[0])):
            #         cv2.circle(input1_vis, (int(joints1[n][i][0]), int(joints1[n][i][1])), 2, [255, 0, 0], 2)
            #         cv2.circle(input2_vis, (int(joints2[n][i][0]), int(joints2[n][i][1])), 2, [255, 0, 0], 2)
            #         cv2.circle(input3_vis, (int(joints3[n][i][0]), int(joints3[n][i][1])), 2, [255, 0, 0], 2)
            # img = cv2.hconcat((input1_vis, input2_vis, input3_vis))
            # hm1 = cv2.resize(cv2.hconcat(heatmaps1),(1800, 64))
            # hm2 = cv2.resize(cv2.hconcat(heatmaps2),(1800, 64))
            # hm3 = cv2.resize(cv2.hconcat(heatmaps3),(1800, 64))
            # cv2.imshow("hi", img)
            # cv2.imshow("hm1", hm1)
            # cv2.imshow("hm2", hm2)
            # cv2.imshow("hm3", hm3)
            # cv2.imwrite("/home/srivasta/{}_img.png".format(k), img)
            # cv2.imwrite("/home/srivasta/{}_hm1.png".format(k), hm1)
            # cv2.imwrite("/home/srivasta/{}_hm2.png".format(k), hm2)
            # cv2.imwrite("/home/srivasta/{}_hm3.png".format(k), hm3)
            # cv2.waitKey(0)

            target_heatmap1 = torch.from_numpy(target_heatmap1)
            target_weight1 = torch.from_numpy(target_weight1)
            target_heatmap2 = torch.from_numpy(target_heatmap2)
            target_weight2 = torch.from_numpy(target_weight2)
            target_heatmap3 = torch.from_numpy(target_heatmap3)
            target_weight3 = torch.from_numpy(target_weight3)

            
            # im = Image.fromarray(input1)
            # im.save("vis/input1.jpeg")
            # im = Image.fromarray(input2)
            # im.save("vis/input2.jpeg")
            # for j in range(15):
            #     save_image(target_heatmap1[j], 'vis/' + str(j)+'_target_heatmap1.jpg')
            #     save_image(target_heatmap2[j], 'vis/' + str(j)+'_target_heatmap2.jpg')
            # exit(1)

            # target_heatmap1_root = torch.from_numpy(target_heatmap1_root)
            # target_weight1_root = torch.from_numpy(target_weight1_root)
            # target_heatmap2_root = torch.from_numpy(target_heatmap2_root)
            # target_weight2_root = torch.from_numpy(target_weight2_root)
            # target_heatmap3_root = torch.from_numpy(target_heatmap3_root)
            # target_weight3_root = torch.from_numpy(target_weight3_root)

            # make joints and joints_vis having same shape
            joints_u1 = np.zeros((self.maximum_person, self.num_joints, 2))
            joints_vis_u1 = np.zeros((self.maximum_person, self.num_joints, 2))
            joints_u2 = np.zeros((self.maximum_person, self.num_joints, 2))
            joints_vis_u2 = np.zeros((self.maximum_person, self.num_joints, 2))
            joints_u3 = np.zeros((self.maximum_person, self.num_joints, 2))
            joints_vis_u3 = np.zeros((self.maximum_person, self.num_joints, 2))

            for i in range(nposes):
                joints_u1[i] = joints1[i]
                joints_vis_u1[i] = joints_vis1[i]
                joints_u2[i] = joints2[i]
                joints_vis_u2[i] = joints_vis2[i]
                joints_u3[i] = joints3[i]
                joints_vis_u3[i] = joints_vis3[i]

            joints_3d_u = np.zeros((self.maximum_person, self.num_joints, 3))
            joints_3d_vis_u = np.zeros((self.maximum_person, self.num_joints, 3))
            if with_3d:
                for i in range(nposes):
                    joints_3d_u[i] = joints_3d[i][:, 0:3]
                    joints_3d_vis_u[i] = joints_3d_vis[i][:, 0:3]
                target_3d = self.generate_3d_target(joints_3d)
            else:
                cube_size = self.initial_cube_size
                target_3d = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
            target_3d = torch.from_numpy(target_3d)

            # if self.debug:
            #     vol = Volume(target_3d)
            #     show(vol, azimuth=10, axes=True).close()
            if self.transform:
                input1 = self.transform(input1)
                input2 = self.transform(input2)
                input3 = self.transform(input3)

            if isinstance(self.root_id, int):
                roots_3d = joints_3d_u[:, self.root_id]
            elif isinstance(self.root_id, list):
                roots_3d = np.mean([joints_3d_u[:, j] for j in self.root_id], axis=0)
            meta1 = {
                "image": image_file,
                "num_person": nposes,
                "joints_3d": joints_3d_u,
                "joints_3d_vis": joints_3d_vis_u,
                "roots_3d": roots_3d,
                "joints": joints_u1,
                "joints_vis": joints_vis_u1,
                "center": c,
                "scale": sc1,
                "rotation": r1,
                "trans": torch.from_numpy(trans1.astype(np.float32)),
                "camera": db_rec["camera"],
                "hflip": do_hflip1,
                "mis_count": self.mis_count,
            }
            meta2 = {
                "image": image_file,
                "num_person": nposes,
                "joints_3d": joints_3d_u,
                "joints_3d_vis": joints_3d_vis_u,
                "roots_3d": roots_3d,
                "joints": joints_u2,
                "joints_vis": joints_vis_u2,
                "center": c,
                "scale": sc2,
                "rotation": r2,
                "trans": torch.from_numpy(trans2.astype(np.float32)),
                "camera": db_rec["camera"],
                "hflip": do_hflip2,
                "mis_count": self.mis_count,
            }
            meta3 = {
                "image": image_file,
                "num_person": nposes,
                "joints_3d": joints_3d_u,
                "joints_3d_vis": joints_3d_vis_u,
                "roots_3d": roots_3d,
                "joints": joints_u3,
                "joints_vis": joints_vis_u3,
                "center": c,
                "scale": s,
                "rotation": 0,
                "trans": torch.from_numpy(trans3.astype(np.float32)),
                "camera": db_rec["camera"],
                "hflip": False,
                "mis_count": self.mis_count,
            }

            inputs1.append(input1)
            target_heatmaps1.append(target_heatmap1)
            target_weights1.append(target_weight1)
            #target_heatmap_roots1.append(target_heatmap1_root)
            #target_weight_roots1.append(target_weight1_root)
            target_3ds1.append(target_3d)
            metas1.append(meta1)
            input_heatmaps1.append(input_heatmap1)

            inputs2.append(input2)
            target_heatmaps2.append(target_heatmap2)
            target_weights2.append(target_weight2)
            #target_heatmap_roots2.append(target_heatmap2_root)
            #target_weight_roots2.append(target_weight2_root)
            target_3ds2.append(target_3d)
            metas2.append(meta2)
            input_heatmaps2.append(input_heatmap2)

            inputs3.append(input3)
            target_heatmaps3.append(target_heatmap3)
            target_weights3.append(target_weight3)
            #target_heatmap_roots3.append(target_heatmap3_root)
            #target_weight_roots3.append(target_weight3_root)
            target_3ds3.append(target_3d)
            metas3.append(meta3)
            input_heatmaps3.append(input_heatmap3)
        return (
            inputs1,
            target_heatmaps1,
            target_weights1,
            #target_heatmap_roots1,
            #target_weight_roots1,
            target_3ds1,
            metas1,
            input_heatmaps1,
            inputs2,
            target_heatmaps2,
            target_weights2,
            #target_heatmap_roots2,
            #target_weight_roots2,
            target_3ds2,
            metas2,
            input_heatmaps2,
            inputs3,
            target_heatmaps3,
            target_weights3,
            #target_heatmap_roots3,
            #target_weight_roots3,
            target_3ds3,
            metas3,
            input_heatmaps3,
        )

    def compute_human_scale(self, pose, joints_vis):
        idx = joints_vis[:, 0] == 1
        if np.sum(idx) == 0:
            return 0
        minx, maxx = np.min(pose[idx, 0]), np.max(pose[idx, 0])
        miny, maxy = np.min(pose[idx, 1]), np.max(pose[idx, 1])
        # return np.clip((maxy - miny) * (maxx - minx), 1.0 / 4 * 256**2,
        #                4 * 256**2)
        return np.clip(np.maximum(maxy - miny, maxx - minx) ** 2, 1.0 / 4 * 96 ** 2, 4 * 96 ** 2)

    def generate_target_heatmap(self, joints, joints_vis):
        """
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        nposes = len(joints)
        num_joints = self.num_joints
        target_weight = np.zeros((num_joints, 1), dtype=np.float32)
        for i in range(num_joints):
            for n in range(nposes):
                if joints_vis[n][i, 0] == 1:
                    target_weight[i, 0] = 1

        assert self.target_type == "gaussian", "Only support gaussian map now!"

        if self.target_type == "gaussian":
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32,
            )
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma  # * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if (
                        joints_vis[n][joint_id, 0] == 0
                        or ul[0] >= self.heatmap_size[0]
                        or ul[1] >= self.heatmap_size[1]
                        or br[0] < 0
                        or br[1] < 0
                    ):
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * cur_sigma ** 2))

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
                        target[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]],
                        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
                    )
                target = np.clip(target, 0, 1)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_target_heatmap_roots(self, joints, joints_vis):
        """
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        """
        nposes = len(joints)
        target_weight = np.zeros((1, 1), dtype=np.float32)
        for n in range(nposes):
            if joints_vis[n][self.root_id, 0] == 1:
                target_weight[0, 0] = 1

        assert self.target_type == "gaussian", "Only support gaussian map now!"

        if self.target_type == "gaussian":
            target = np.zeros(
                (1, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32,
            )
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(joints[n] / feat_stride, joints_vis[n])
                if human_scale == 0:
                    continue
                cur_sigma = self.sigma  # * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                joint_id = self.root_id
                feat_stride = self.image_size / self.heatmap_size
                mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if (
                    joints_vis[n][joint_id, 0] == 0
                    or ul[0] >= self.heatmap_size[0]
                    or ul[1] >= self.heatmap_size[1]
                    or br[0] < 0
                    or br[1] < 0
                ):
                    continue

                size = 2 * tmp_size + 1
                x = np.arange(0, size, 1, np.float32)
                y = x[:, np.newaxis]
                x0 = y0 = size // 2
                g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * cur_sigma ** 2))

                # Usable gaussian range
                g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                # Image range
                img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                target[0][img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
                    target[0][img_y[0] : img_y[1], img_x[0] : img_x[1]],
                    g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
                )
                target = np.clip(target, 0, 1)

        if self.use_different_joints_weight:
            target_weight = np.multiply(target_weight, self.joints_weight)

        return target, target_weight

    def generate_3d_target(self, joints_3d):
        num_people = len(joints_3d)

        space_size = self.space_size
        space_center = self.space_center
        cube_size = self.initial_cube_size
        grid1Dx = np.linspace(-space_size[0] / 2, space_size[0] / 2, cube_size[0]) + space_center[0]
        grid1Dy = np.linspace(-space_size[1] / 2, space_size[1] / 2, cube_size[1]) + space_center[1]
        grid1Dz = np.linspace(-space_size[2] / 2, space_size[2] / 2, cube_size[2]) + space_center[2]

        target = np.zeros((cube_size[0], cube_size[1], cube_size[2]), dtype=np.float32)
        cur_sigma = 200.0

        for n in range(num_people):
            joint_id = self.root_id  # mid-hip
            if isinstance(joint_id, int):
                mu_x = joints_3d[n][joint_id][0]
                mu_y = joints_3d[n][joint_id][1]
                mu_z = joints_3d[n][joint_id][2]
            elif isinstance(joint_id, list):
                mu_x = (joints_3d[n][joint_id[0]][0] + joints_3d[n][joint_id[1]][0]) / 2.0
                mu_y = (joints_3d[n][joint_id[0]][1] + joints_3d[n][joint_id[1]][1]) / 2.0
                mu_z = (joints_3d[n][joint_id[0]][2] + joints_3d[n][joint_id[1]][2]) / 2.0
            i_x = [
                np.searchsorted(grid1Dx, mu_x - 3 * cur_sigma),
                np.searchsorted(grid1Dx, mu_x + 3 * cur_sigma, "right"),
            ]
            i_y = [
                np.searchsorted(grid1Dy, mu_y - 3 * cur_sigma),
                np.searchsorted(grid1Dy, mu_y + 3 * cur_sigma, "right"),
            ]
            i_z = [
                np.searchsorted(grid1Dz, mu_z - 3 * cur_sigma),
                np.searchsorted(grid1Dz, mu_z + 3 * cur_sigma, "right"),
            ]
            if i_x[0] >= i_x[1] or i_y[0] >= i_y[1] or i_z[0] >= i_z[1]:
                continue

            gridx, gridy, gridz = np.meshgrid(
                grid1Dx[i_x[0] : i_x[1]],
                grid1Dy[i_y[0] : i_y[1]],
                grid1Dz[i_z[0] : i_z[1]],
                indexing="ij",
            )
            g = np.exp(
                -((gridx - mu_x) ** 2 + (gridy - mu_y) ** 2 + (gridz - mu_z) ** 2)
                / (2 * cur_sigma ** 2)
            )
            target[i_x[0] : i_x[1], i_y[0] : i_y[1], i_z[0] : i_z[1]] = np.maximum(
                target[i_x[0] : i_x[1], i_y[0] : i_y[1], i_z[0] : i_z[1]], g
            )

        target = np.clip(target, 0, 1)
        return target

    def generate_input_heatmap(self, joints):
        """
        :param joints:  [[num_joints, 3]]
        :param joints_vis: [num_joints, 3]
        :return: input_heatmap
        """
        nposes = len(joints)
        num_joints = self.cfg.NETWORK.NUM_JOINTS

        assert self.target_type == "gaussian", "Only support gaussian map now!"

        if self.target_type == "gaussian":
            target = np.zeros(
                (num_joints, self.heatmap_size[1], self.heatmap_size[0]),
                dtype=np.float32,
            )
            feat_stride = self.image_size / self.heatmap_size

            for n in range(nposes):
                human_scale = 2 * self.compute_human_scale(
                    joints[n][:, 0:2] / feat_stride, np.ones((num_joints, 1))
                )
                if human_scale == 0:
                    continue

                cur_sigma = self.sigma  # * np.sqrt((human_scale / (96.0 * 96.0)))
                tmp_size = cur_sigma * 3
                for joint_id in range(num_joints):
                    feat_stride = self.image_size / self.heatmap_size
                    mu_x = int(joints[n][joint_id][0] / feat_stride[0])
                    mu_y = int(joints[n][joint_id][1] / feat_stride[1])
                    ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                    br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                    if (
                        ul[0] >= self.heatmap_size[0]
                        or ul[1] >= self.heatmap_size[1]
                        or br[0] < 0
                        or br[1] < 0
                    ):
                        continue

                    size = 2 * tmp_size + 1
                    x = np.arange(0, size, 1, np.float32)
                    y = x[:, np.newaxis]
                    x0 = y0 = size // 2
                    if "campus" in self.dataset_name:
                        max_value = 1.0
                    else:
                        max_value = joints[n][joint_id][2] if len(joints[n][joint_id]) == 3 else 1.0
                        # max_value = max_value**0.5
                    g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * cur_sigma ** 2)) * max_value

                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
                    img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

                    target[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]] = np.maximum(
                        target[joint_id][img_y[0] : img_y[1], img_x[0] : img_x[1]],
                        g[g_y[0] : g_y[1], g_x[0] : g_x[1]],
                    )
                target = np.clip(target, 0, 1)

        return target
