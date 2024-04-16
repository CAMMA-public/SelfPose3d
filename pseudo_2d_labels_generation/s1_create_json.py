'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import pickle
import cv2
import json
import os
from tqdm import tqdm

TRAIN_DB_PATH = "./data/panoptic-toolbox/data/group_train_cam5_sub.pkl"
ROOT_DIR = "./data/"
OUT_FILE = "./pseudo_labels/image_info_train_panoptic.json"


def main():
    out_data = {"annotations": [], "images": [], "categories": []}
    data = pickle.load(open(TRAIN_DB_PATH, "rb"))["db"]
    for ii, d in enumerate(tqdm(data)):
        img = cv2.imread(os.path.join(ROOT_DIR, d["image"]))
        height, width, _ = img.shape
        out_data["images"].append(
            {
                "file_name": d["image"],
                "id": ii,
                "height": height,
                "width": width,
                "key": d["key"],
                "url": d["image"],
            }
        )
        # add dummy data
        out_data["annotations"].append(
            {
                "id": ii,
                "image_id": ii,
                "category_id": 1,
                "score": 1,
                "keypoints": [0] * 51,
                "iscrowd": 0,
                "area": 0,
                "bbox": [0] * 4,
            }
        )
    out_data["categories"].append(
        {
            "supercategory": "person",
            "id": 1,
            "name": "person",
            "keypoints": [
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
            ],
        }
    )
    with open(OUT_FILE, "w") as f:
        json.dump(out_data, f, indent=4)


if __name__ == "__main__":
    main()
