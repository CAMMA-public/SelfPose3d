'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import json
import cv2
import numpy as np
import os
from tqdm import tqdm
from copy import deepcopy
KPT_FILE = "./pseudo_labels/kpt2d_inference/mvor/pose_hrnet/w48_384x288_adam_lr1e-3_generate/results/keypoints_val2017_results_0.json"
GT_FILE = "./pseudo_labels/pseudo_bboxes_panoptic.json"
OUT_FILE = "./pseudo_labels/pseudo_kpt2d_panoptic.json"

def process_kps(kpts, x1, y1, x2, y2, thresh=0.3):
    pose = np.array(kpts).reshape(-1, 3)
    xd = pose[:, 0]
    yd = pose[:, 1]
    score = pose[:, 2]
    score = np.where(score < thresh, 0, 2)
    num_kps = int(np.sum(score == 2))
    kps_count = 0
    f_kps = []
    if num_kps > 3:
        for p, _ in enumerate(xd):
            if score[[p]].item() == 2:
                xx, yy, ss = xd[[p]].item(), yd[[p]].item(), score[[p]].item()
                if xx >= x1 and xx <= x2 and yy >= y1 and yy <= y2:
                    f_kps.append(xx)
                    f_kps.append(yy)
                    f_kps.append(ss)
                    kps_count += 1
                else:
                    f_kps.append(0)
                    f_kps.append(0)
                    f_kps.append(0)
            else:
                f_kps.append(0)
                f_kps.append(0)
                f_kps.append(0)
    return f_kps, kps_count    


def main():
    _kpt = json.load(open(KPT_FILE))
    kpt = {a["original_id"]: a for a in _kpt}

    gt = json.load(open(GT_FILE))
    id2im = {k["id"]: k for k in gt["images"]}
    for ann in tqdm(gt["annotations"]):
        if ann["id"] in kpt:
            height = id2im[ann["image_id"]]["height"]
            width = id2im[ann["image_id"]]["width"]
            x, y, w, h = ann["bbox"]
            kp = kpt[ann["id"]]
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if ann["area"] > 0 and x2 >= x1 and y2 >= y1:
                ann["bbox"] = [float(x), float(y), float(w), float(h)]
                ann["delete"] = 0
                ann["keypoints_soft"] = kp["keypoints"]
                ann["center"] = kp["center"]
                ann["scale"] = kp["scale"]
                ann["keypoints_krcnn_soft"] = deepcopy(ann["keypoints_krcnn"])
                # 1. remove keypoints outside the bbox
                # 2. see the appropriate threshold value to remove the kps (0.4)
                f_kps, kps_count  = process_kps(kp["keypoints"], x1, y1, x2, y2, thresh=0.05)
                f_kps_kcnn, kps_count_kcnn  = process_kps(ann["keypoints_krcnn"], x1, y1, x2, y2, thresh=0.05)
                if kps_count >= 3:
                    ann["keypoints"] = f_kps
                    ann["num_keypoints"] = kps_count
                else:
                    ann["keypoints"] = [0] * 51
                    ann["num_keypoints"] = 0
                if kps_count_kcnn >= 3:
                    ann["keypoints_krcnn"] = f_kps_kcnn
                    ann["num_keypoints_krcnn"] = kps_count_kcnn
                else:
                    ann["keypoints_krcnn"] = [0] * 51      
                    ann["num_keypoints_krcnn"] = 0          
            else:
                ann["delete"] = 1
        else:
            ann["delete"] = 1

    delete_count = 0
    for ann in gt["annotations"]:
        if ann["delete"] == 1:
            delete_count += 1
    print("anns to be deleted:", delete_count)
    
    final_annotations = []
    for ann in gt["annotations"]:
        if ann["delete"] == 0:
            final_annotations.append(ann)
    gt["annotations"] = final_annotations   

    delete_count = 0
    for ann in gt["annotations"]:
        if ann["delete"] == 1:
            delete_count += 1
    print("anns to be deleted:", delete_count)

    with open(OUT_FILE, "w") as f:
        json.dump(gt, f)
    print("finish adding keypoint detections")

if __name__ == "__main__":
    main()
