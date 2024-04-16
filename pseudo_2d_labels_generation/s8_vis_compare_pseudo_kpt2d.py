'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

import random
import cv2
import json
import numpy as np
import os

import pickle

random.seed(100)
GT_FILE_PKL = "./data/panoptic-toolbox/data/group_train_cam5_sub.pkl"
PSEUDO_FILE_PKL = "./pseudo_labels/group_train_cam5_pseudo_krcnn_soft.pkl"
IMGDIR = "./data"


COCO_COLOR_LIST = [
    "#e6194b",  # red,       nose
    "#3cb44b",  # green      left_eye
    "#ffe119",  # yellow     right_eye
    "#0082c8",  # blue       left_ear
    "#f58231",  # orange     right_ear
    "#911eb4",  # purple     left_shoulder
    "#46f0f0",  # cyan       right_shoulder
    "#f032e6",  # magenta    left_elbow
    "#d2f53c",  # lime       right_elbow
    "#fabebe",  # pink       left_wrist
    "#008080",  # teal       right_wrist
    "#e6beff",  # lavender   left_hip
    "#aa6e28",  # brown      right_hip
    "#fffac8",  # beige      left_knee
    "#800000",  # maroon     right_knee
    "#aaffc3",  # mint       left_ankle
    "#808000",  # olive      right_ankle
]
coco_colors_skeleton = [
    "m",
    "m",
    "g",
    "g",
    "r",
    "m",
    "g",
    "r",
    "m",
    "g",
    "m",
    "g",
    "r",
    "m",
    "g",
    "m",
    "g",
    "m",
    "g",
]
coco_pairs = [
    [15, 13],
    [13, 11],
    [16, 14],
    [14, 12],
    [11, 12],
    [5, 11],
    [6, 12],
    [5, 6],
    [5, 7],
    [6, 8],
    [7, 9],
    [8, 10],
    [1, 2],
    [0, 1],
    [0, 2],
    [1, 3],
    [2, 4],
    [3, 5],
    [4, 6],
]

PANOPTIC_COLOR_LIST = [
    "#e6194b",  # red,       nose
    "#3cb44b",  # green      left_eye
    "#ffe119",  # yellow     right_eye
    "#0082c8",  # blue       left_ear
    "#f58231",  # orange     right_ear
    "#911eb4",  # purple     left_shoulder
    "#46f0f0",  # cyan       right_shoulder
    "#f032e6",  # magenta    left_elbow
    "#d2f53c",  # lime       right_elbow
    "#fabebe",  # pink       left_wrist
    "#008080",  # teal       right_wrist
    "#e6beff",  # lavender   left_hip
    "#aa6e28",  # brown      right_hip
    "#fffac8",  # beige      left_knee
    "#800000",  # maroon     right_knee
]

panoptic_colors_skeleton = [
    "m",
    "m",
    "g",
    "g",
    "r",
    "m",
    "g",
    "r",
    "m",
    "g",
    "m",
    "g",
    "r",
    "m"
]

panoptic_pairs = [
    [0, 1],
    [0, 2],
    [0, 3],
    [3, 4],
    [4, 5],
    [0, 9],
    [9, 10],
    [10, 11],
    [2, 6],
    [2, 12],
    [6, 7],
    [7, 8],
    [12, 13],
    [13, 14],
]


def fixed_bright_colors():
    return [
        [207, 73, 179],
        [53, 84, 209],
        [31, 252, 54],
        [203, 173, 34],
        [229, 18, 115],
        [236, 31, 98],
        [50, 195, 222],
        [169, 52, 199],
        [44, 69, 172],
        [231, 4, 80],
        [191, 197, 33],
        [46, 194, 180],
        [35, 228, 69],
        [217, 211, 25],
        [253, 10, 48],
        [170, 213, 80],
        [206, 77, 13],
        [197, 178, 11],
        [204, 163, 32],
        [143, 222, 64],
        [45, 208, 109],
        [67, 185, 44],
        [91, 68, 230],
        [249, 246, 20],
        [75, 202, 201],
        [11, 202, 193],
        [221, 75, 180],
        [241, 16, 142],
        [126, 9, 231],
        [40, 210, 122],
        [10, 136, 205],
        [38, 230, 105],
        [193, 97, 26],
        [203, 18, 101],
        [42, 173, 94],
        [222, 45, 135],
        [33, 184, 48],
        [121, 49, 195],
        [31, 39, 226],
        [204, 48, 143],
        [220, 47, 192],
        [223, 220, 73],
        [46, 177, 170],
        [17, 245, 161],
        [159, 51, 107],
        [10, 39, 205],
        [50, 237, 101],
        [116, 35, 171],
        [213, 76, 76],
        [88, 203, 47],
        [202, 205, 14],
        [100, 233, 4],
        [227, 34, 192],
        [21, 79, 239],
        [30, 198, 36],
        [140, 38, 240],
        [97, 26, 215],
        [48, 122, 225],
        [158, 51, 196],
        [11, 212, 45],
        [190, 173, 39],
        [34, 185, 34],
        [98, 58, 219],
        [147, 233, 66],
        [44, 239, 69],
        [192, 177, 38],
        [53, 233, 53],
        [41, 222, 44],
        [228, 70, 120],
        [221, 153, 58],
        [131, 19, 222],
        [203, 27, 140],
        [170, 72, 54],
        [182, 58, 173],
        [194, 218, 84],
        [233, 34, 30],
        [100, 173, 37],
        [72, 92, 227],
        [216, 90, 183],
        [66, 215, 125],
        [183, 63, 41],
        [228, 29, 54],
        [29, 221, 125],
        [172, 12, 207],
        [20, 228, 205],
        [16, 228, 121],
        [210, 21, 198],
        [80, 135, 206],
        [196, 165, 27],
    ]


def draw_2d_keypoints(image, pt2d, use_color):

    colors_skeleton = panoptic_colors_skeleton
    pairs = panoptic_pairs

    for idx in range(len(colors_skeleton)):
        color = use_color
        pair = pairs[idx]
        pt1, sc1 = tuple(pt2d[pair[0], :].astype(int)[0:2]), pt2d[pair[0]][2]
        pt2, sc2 = tuple(pt2d[pair[1], :].astype(int)[0:2]), pt2d[pair[1]][2]
        if 0 not in pt1 + pt2:
            cv2.line(image, pt1, pt2, color, 6, cv2.LINE_AA)

    # draw keypoints
    for idx in range(len(PANOPTIC_COLOR_LIST)):
        pt, sc = tuple(pt2d[idx, :].astype(int)[0:2]), pt2d[idx][2]
        if 0 not in pt:
            c = PANOPTIC_COLOR_LIST[idx].lstrip("#")
            c = tuple(int(c[i : i + 2], 16) for i in (0, 2, 4))
            c = (c[2], c[1], c[0])  # rgb->bgr
            cv2.circle(image, pt, 5, c, 3, cv2.LINE_AA)
            cv2.circle(image, pt, 6, (0, 0, 0), 1, cv2.LINE_AA)


def draw_anns_v2(draw, anns, rand_colors):
    thickness = 2
    for ii, ann in enumerate(anns):
        #x, y, w, h = [int(a) for a in ann["bbox"]]
        color = rand_colors[ii]
        #cv2.rectangle(draw, (x, y), (x + w, y + h), color, thickness)
        kpts_2d = np.array(ann["keypoints"]).reshape(15, 3)
        # print(ann["num_keypoints"], ann["num_keypoints_krcnn"])
        draw_2d_keypoints(draw, kpts_2d, color)
    return draw


def main():

    gt = pickle.load(open(GT_FILE_PKL, "rb"))["db"]
    pseudo = pickle.load(open(PSEUDO_FILE_PKL, "rb"))["db"]
    gt = {k["key"]: k for k in gt}
    pseudo = {k["key"]: k for k in pseudo}
    keys = list(gt.keys())
    rand_colors = fixed_bright_colors()

    #for key in keys:
    for _ in range(1000):
        key = random.choice(keys)
        gt_anns = gt[key]
        pseudo_anns = pseudo[key]
        img_gt = cv2.imread(os.path.join(IMGDIR, gt_anns["image"]))
        img_pseudo = cv2.imread(os.path.join(IMGDIR, pseudo_anns["image"]))

        anns_gt, anns_pseudo = [], []
        for i in range(len(gt_anns["joints_2d"])):
            kpt = gt_anns["joints_2d"][i]
            kpt_vis = gt_anns["joints_2d_vis"][i]
            k = np.concatenate((kpt, kpt_vis[:, 1:]), 1)
            anns_gt.append({"keypoints": k})

        for i in range(len(pseudo_anns["joints_2d"])):
            kpt = pseudo_anns["joints_2d"][i]
            kpt_vis = pseudo_anns["joints_2d_vis"][i]
            k = np.concatenate((kpt, kpt_vis[:, 1:]), 1)
            anns_pseudo.append({"keypoints": k})
        
        anns_gt = sorted(anns_gt, key=lambda k: k["keypoints"][2,0])
        anns_pseudo = sorted(anns_pseudo, key=lambda k: k["keypoints"][2,0])

        img_gt = draw_anns_v2(img_gt, anns_gt, rand_colors)
        img_pseudo = draw_anns_v2(img_pseudo, anns_pseudo, rand_colors)
        if True:
            scale_percent = 25
            width = int(img_gt.shape[1] * scale_percent / 100)
            height = int(img_gt.shape[0] * scale_percent / 100)
            dim = (width, height)            
            img_gt = cv2.resize(img_gt, dim, interpolation = cv2.INTER_LINEAR)        
            img_pseudo = cv2.resize(img_pseudo, dim, interpolation = cv2.INTER_LINEAR)        
    
        cv2.imshow("img_gt", img_gt)
        cv2.imshow("img_pseudo", img_pseudo)
        

        ch = cv2.waitKey(0)
        print("ch=", ch)
        if ch == 27:
            break
        elif ch == 115: #s
            dir_name_out = "vis"
            cv2.imwrite(os.path.join(dir_name_out, "gt_{}.jpg".format(key)), img_gt)
            cv2.imwrite(os.path.join(dir_name_out, "pseudo_{}.jpg".format(key)), img_pseudo)



if __name__ == "__main__":
    main()
