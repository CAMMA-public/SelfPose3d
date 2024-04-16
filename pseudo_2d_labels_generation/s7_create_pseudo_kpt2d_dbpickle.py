'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import pickle
import json
import numpy as np
from copy import deepcopy
from tqdm import tqdm

PSEUDO_GT_JSON = "./pseudo_labels/pseudo_kpt2d_panoptic.json"
TRAIN_DB_PATH = "./data/panoptic-toolbox/data/group_train_cam5_sub.pkl"
OUT_PATH_HRNET_HARD = "./pseudo_labels/group_train_cam5_pseudo_hrnet_hard.pkl"
OUT_PATH_HRNET_SOFT = "./pseudo_labels/group_train_cam5_pseudo_hrnet_soft.pkl"
OUT_PATH_KRCNN_HARD = "./pseudo_labels/group_train_cam5_pseudo_krcnn_hard.pkl"
OUT_PATH_KRCNN_SOFT = "./pseudo_labels/group_train_cam5_pseudo_krcnn_soft.pkl"

JOINTS_DEF_OUT = [
    "neck",
    "nose",
    "mid-hip",
    "left_shoulder",
    "left_elbow",
    "left_wrist",
    "left_hip",
    "left_knee",
    "left_ankle",
    "right_shoulder",
    "right_elbow",
    "right_wrist",
    "right_hip",
    "right_knee",
    "right_ankle",
]

JOINTS_DEF_IN = [
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
]


def get_mapping():
    mapping = [0] * len(JOINTS_DEF_OUT)
    for i, p in enumerate(JOINTS_DEF_OUT):
        if p in JOINTS_DEF_IN:
            mapping[i] = JOINTS_DEF_IN.index(p)
        else:
            mapping[i] = -1
    return mapping


def coco2panoptic(kp, mapping):
    kp_np = np.array(kp).reshape(-1, 3)
    left_shldr = kp_np[JOINTS_DEF_IN.index("left_shoulder")]
    right_shldr = kp_np[JOINTS_DEF_IN.index("right_shoulder")]
    left_hip = kp_np[JOINTS_DEF_IN.index("left_hip")]
    right_hip = kp_np[JOINTS_DEF_IN.index("right_hip")]

    # get the neck
    if left_shldr[2] > 0 and right_shldr[2] > 0:
        neck = (left_shldr + right_shldr) / 2.0
        neck[2] = 2.0
    elif left_shldr[2] > 0 and right_shldr[2] == 0:
        neck = left_shldr
        neck[2] = 2.0
    elif left_shldr[2] == 0 and right_shldr[2] > 0:
        neck = right_shldr
        neck[2] = 2.0
    else:
        neck = np.array([0.0, 0.0, 0.0])

    # get the hip
    if left_hip[2] > 0 and right_hip[2] > 0:
        hip = (left_hip + right_hip) / 2.0
        hip[2] = 2.0
    elif left_hip[2] > 0 and right_hip[2] == 0:
        hip = left_hip
        hip[2] = 2.0
    elif left_hip[2] == 0 and right_hip[2] > 0:
        hip = right_hip
        hip[2] = 2.0
    else:
        hip = np.array([0.0, 0.0, 0.0])

    kp_po = deepcopy(kp_np[mapping])
    kp_po[JOINTS_DEF_OUT.index("neck")] = neck
    kp_po[JOINTS_DEF_OUT.index("mid-hip")] = hip
    joints_2d = kp_po[:, :2]
    joints_2d_vis = kp_po[:, 2:] > 0
    joints_2d_vis = np.concatenate([joints_2d_vis, joints_2d_vis], 1)
    return joints_2d, joints_2d_vis


def init_outdata(gt_data):
    out_data = {}
    out_data["interval"] = gt_data["interval"]
    out_data["cam_list"] = gt_data["cam_list"]
    out_data["sequence_list"] = gt_data["sequence_list"]
    out_data["db"] = []
    return out_data


def get_dict(key, pseudo_data, ii, ds, joints_2d, joints_2d_vis, bboxes, scores, centers, scales):
    rt_dict = {
        "key": key,
        "image": pseudo_data["images"][ii]["file_name"],
        "height": pseudo_data["images"][ii]["height"],
        "width": pseudo_data["images"][ii]["width"],
        "camera": ds["camera"],
        "joints_2d": joints_2d,
        "joints_2d_vis": joints_2d_vis,
        "bboxes": bboxes,
        "scores": scores,
        "centers": centers,
        "scales": scales,
    }
    return rt_dict


def main():
    mapping = get_mapping()
    gt_data = pickle.load(open(TRAIN_DB_PATH, "rb"))
    pseudo_data = json.load(open(PSEUDO_GT_JSON))
    dets = {k["id"]: [] for k in pseudo_data["images"]}
    for d in pseudo_data["annotations"]:
        dets[d["image_id"]].append(d)

    out_data_hrnet_hard = init_outdata(gt_data)
    out_data_hrnet_soft = init_outdata(gt_data)
    out_data_krcnn_hard = init_outdata(gt_data)
    out_data_krcnn_soft = init_outdata(gt_data)

    for ii in tqdm(range(len(gt_data["db"]))):
        key = gt_data["db"][ii]["key"]
        assert key == pseudo_data["images"][ii]["key"]
        ds = gt_data["db"][ii]
        im_id = pseudo_data["images"][ii]["id"]
        pseudo_anns = dets[im_id]

        joints_2d_hrnet_hard_lst, joints_2d_vis_hrnet_hard_lst = [], []
        joints_2d_hrnet_soft_lst, joints_2d_vis_hrnet_soft_lst = [], []
        joints_2d_krcnn_hard_lst, joints_2d_vis_krcnn_hard_lst = [], []
        joints_2d_krcnn_soft_lst, joints_2d_vis_krcnn_soft_lst = [], []
        bboxes, scores, centers, scales = [], [], [], []
        for ann in pseudo_anns:
            joints_2d_kpt, joints_2d_vis_kpt = coco2panoptic(ann["keypoints"], mapping)
            joints_2d_soft, joints_2d_vis_soft = coco2panoptic(ann["keypoints_soft"], mapping)
            joints_2d_krcnn, joints_2d_vis_krcnn = coco2panoptic(ann["keypoints_krcnn"], mapping)
            joints_2d_krcnn_soft, joints_2d_vis_krcnn_soft = coco2panoptic(ann["keypoints_krcnn_soft"], mapping)

            joints_2d_hrnet_hard_lst.append(joints_2d_kpt)
            joints_2d_vis_hrnet_hard_lst.append(joints_2d_vis_kpt)

            joints_2d_hrnet_soft_lst.append(joints_2d_soft)
            joints_2d_vis_hrnet_soft_lst.append(joints_2d_vis_soft)

            joints_2d_krcnn_hard_lst.append(joints_2d_krcnn)
            joints_2d_vis_krcnn_hard_lst.append(joints_2d_vis_krcnn)

            joints_2d_krcnn_soft_lst.append(joints_2d_krcnn_soft)
            joints_2d_vis_krcnn_soft_lst.append(joints_2d_vis_krcnn_soft)

            bboxes.append(ann["bbox"])
            scores.append(ann["score"])
            centers.append(ann["center"])
            scales.append(ann["scale"])

        dict_hrnet_hard = get_dict(key, pseudo_data, ii, ds, joints_2d_hrnet_hard_lst, joints_2d_vis_hrnet_hard_lst, bboxes, scores, centers, scales)
        dict_hrnet_soft = get_dict(key, pseudo_data, ii, ds, joints_2d_hrnet_soft_lst, joints_2d_vis_hrnet_soft_lst, bboxes, scores, centers, scales)
        dict_krcnn_hard = get_dict(key, pseudo_data, ii, ds, joints_2d_krcnn_hard_lst, joints_2d_vis_krcnn_hard_lst, bboxes, scores, centers, scales)
        dict_krcnn_soft = get_dict(key, pseudo_data, ii, ds, joints_2d_krcnn_soft_lst, joints_2d_vis_krcnn_soft_lst, bboxes, scores, centers, scales)
            
        out_data_hrnet_hard["db"].append(dict_hrnet_hard)
        out_data_hrnet_soft["db"].append(dict_hrnet_soft)
        out_data_krcnn_hard["db"].append(dict_krcnn_hard)
        out_data_krcnn_soft["db"].append(dict_krcnn_soft)        

    pickle.dump(out_data_hrnet_hard, open(OUT_PATH_HRNET_HARD, "wb"))
    pickle.dump(out_data_hrnet_soft, open(OUT_PATH_HRNET_SOFT, "wb"))
    pickle.dump(out_data_krcnn_hard, open(OUT_PATH_KRCNN_HARD, "wb"))
    pickle.dump(out_data_krcnn_soft, open(OUT_PATH_KRCNN_SOFT, "wb"))


if __name__ == "__main__":
    main()
