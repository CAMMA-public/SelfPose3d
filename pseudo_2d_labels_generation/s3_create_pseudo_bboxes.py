'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


import json
import numpy as np
import cv2
from tqdm import tqdm
from copy import deepcopy
GT_TRAIN_JSON="./pseudo_labels/image_info_train_panoptic.json"
DT_TRAIN_JSON="./pseudo_labels/bbox_inference/inference/coco_instances_results.json"
OUT_PSEUDO_BBOX="./pseudo_labels/pseudo_bboxes_panoptic.json"



def update_anns(anno_dict):
    for index, ann in tqdm(enumerate(anno_dict)):
        ann["id"] = index+1
        ann["num_keypoints"] = 0
        ann["keypoints_krcnn"] = deepcopy(ann["keypoints"])
        ann["keypoints"] = [0]*51
        ann["area"] = ann["bbox"][2] * ann["bbox"][3]
        ann["iscrowd"] = 0
    return anno_dict


def create_pseudo_bboxes():
    print("loading files")


    gt_train = json.load(open(GT_TRAIN_JSON))
    dt_train = json.load(open(DT_TRAIN_JSON))
    print("files loaded")
    print("total anns train:", len(dt_train))

    dt_train = [g for g in dt_train if g["category_id"] == 1 and g["score"] > 0.7] 
    print("total anns train after filter:", len(dt_train))

    print("procesing each annotations")
    dt_train = update_anns(dt_train)
    print("annotations procesing finished")

    print("finished processing")
    gt_train["annotations"] = dt_train

    with open(OUT_PSEUDO_BBOX, "w") as f:
        json.dump(gt_train, f)

    print("finished writing")


if __name__ == "__main__":
    create_pseudo_bboxes()
