'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''


import math
import numpy as np
import torchvision
import cv2
import os
import matplotlib
from copy import deepcopy
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from vedo import Points, Lines, Picture, show, screenshot
import torch
import utils.cameras as cameras
from utils.transforms import get_affine_transform as get_transform
from utils.transforms import affine_transform_pts_cuda as do_transform
from matplotlib import colors


COLORS = ["m", "b", "g", "y", "orange", "c", "pink", "royalblue", "lightgreen", "gold"]

JOINTS_DEF = {
    'neck': 0,
    'nose': 1,
    'mid-hip': 2,
    'l-shoulder': 3,
    'l-elbow': 4,
    'l-wrist': 5,
    'l-hip': 6,
    'l-knee': 7,
    'l-ankle': 8,
    'r-shoulder': 9,
    'r-elbow': 10,
    'r-wrist': 11,
    'r-hip': 12,
    'r-knee': 13,
    'r-ankle': 14,
    # 'l-eye': 15,
    # 'l-ear': 16,
    # 'r-eye': 17,
    # 'r-ear': 18,
}

# panoptic    r      r        m       m      m        g       g        g        m        g       m       m        g         g
LIMBS15 = [[0, 1], [0, 2], [0, 3], [3, 4], [4, 5], [0, 9], [9, 10], [10, 11], [2, 6], [2, 12], [6, 7], [7, 8], [12, 13], [13, 14],]
LIMBS15_COLORS = ["r","r","m","m","m","g","g","g","m","g","m","m","g","g"]
# # h36m
# LIMBS17 = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8],
#          [8, 9], [9, 10], [8, 14], [14, 15], [15, 16], [8, 11], [11, 12], [12, 13]]
# coco17
LIMBS17 = [[0, 1], [0, 2], [1, 2], [1, 3], [2, 4], [3, 5], [4, 6], [5, 7], [7, 9], [6, 8], [8, 10], [5, 11], [11, 13], [13, 15], [6, 12], [12, 14], [14, 16], [5, 6], [11, 12],]

# shelf / campus
LIMBS14 = [[0, 1], [1, 2], [3, 4], [4, 5], [2, 3], [6, 7], [7, 8], [9, 10], [10, 11], [2, 8], [3, 9], [8, 12], [9, 12], [12, 13],]

def save_batch_image_with_joints_multi(
    batch_image,
    batch_joints,
    batch_joints_vis,
    num_person,
    file_name,
    nrow=8,
    padding=2,
):
    """
    batch_image: [batch_size, channel, height, width]
    batch_joints: [batch_size, num_person, num_joints, 3],
    batch_joints_vis: [batch_size, num_person, num_joints, 1],
    num_person: [batch_size]
    }
    """
    batch_image = batch_image.flip(1)
    grid = torchvision.utils.make_grid(batch_image, nrow, padding, True)
    ndarr = grid.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
    ndarr = ndarr.copy()

    nmaps = batch_image.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height = int(batch_image.size(2) + padding)
    width = int(batch_image.size(3) + padding)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            for n in range(num_person[k]):
                joints = batch_joints[k, n]
                joints_vis = batch_joints_vis[k, n]

                for joint, joint_vis in zip(joints, joints_vis):
                    joint[0] = x * width + padding + joint[0]
                    joint[1] = y * height + padding + joint[1]
                    if joint_vis[0]:
                        cv2.circle(
                            ndarr, (int(joint[0]), int(joint[1])), 2, [0, 255, 255], 2
                        )
            k = k + 1
    cv2.imwrite(file_name, ndarr)


def save_batch_heatmaps_multi(batch_image, batch_heatmaps, file_name, normalize=True):
    """
    batch_image: [batch_size, channel, height, width]
    batch_heatmaps: ['batch_size, num_joints, height, width]
    file_name: saved file name
    """
    if normalize:
        batch_image = batch_image.clone()
        min = float(batch_image.min())
        max = float(batch_image.max())

        batch_image.add_(-min).div_(max - min + 1e-5)
    batch_image = batch_image.flip(1)

    batch_size = batch_heatmaps.size(0)
    num_joints = batch_heatmaps.size(1)
    heatmap_height = batch_heatmaps.size(2)
    heatmap_width = batch_heatmaps.size(3)

    grid_image = np.zeros(
        (batch_size * heatmap_height, (num_joints + 1) * heatmap_width, 3),
        dtype=np.uint8,
    )

    for i in range(batch_size):
        image = (
            batch_image[i].mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()
        )
        heatmaps = batch_heatmaps[i].mul(255).clamp(0, 255).byte().cpu().numpy()

        resized_image = cv2.resize(image, (int(heatmap_width), int(heatmap_height)))

        height_begin = heatmap_height * i
        height_end = heatmap_height * (i + 1)
        for j in range(num_joints):
            heatmap = heatmaps[j, :, :]
            colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
            masked_image = colored_heatmap * 0.7 + resized_image * 0.3

            width_begin = heatmap_width * (j + 1)
            width_end = heatmap_width * (j + 2)
            grid_image[height_begin:height_end, width_begin:width_end, :] = masked_image
            # grid_image[height_begin:height_end, width_begin:width_end, :] = \
            #     colored_heatmap*0.7 + resized_image*0.3

        grid_image[height_begin:height_end, 0:heatmap_width, :] = resized_image

    cv2.imwrite(file_name, grid_image)


def save_debug_images_multi(config, input, meta, target, output, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, "image_with_joints")
    dirname2 = os.path.join(dirname, "batch_heatmaps")

    for dir in [dirname1, dirname2]:
        if not os.path.exists(dir):
            os.makedirs(dir)

    prefix1 = os.path.join(dirname1, basename)
    prefix2 = os.path.join(dirname2, basename)

    if config.DEBUG.SAVE_BATCH_IMAGES_GT:
        save_batch_image_with_joints_multi(
            input,
            meta["joints"],
            meta["joints_vis"],
            meta["num_person"],
            "{}_gt.jpg".format(prefix1),
        )
    if config.DEBUG.SAVE_HEATMAPS_GT:
        save_batch_heatmaps_multi(input, target, "{}_hm_gt.jpg".format(prefix2))
    if config.DEBUG.SAVE_HEATMAPS_PRED:
        save_batch_heatmaps_multi(input, output, "{}_hm_pred.jpg".format(prefix2))



def save_debug_3d_images_all(config, meta, preds, inputs, targets_2d, heatmaps, prefix, plot_gt = False, render_to_file=True):
    img_size = config.NETWORK.IMAGE_SIZE
    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, "3d_joints_all")

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    batch_size = meta[0]["num_person"].shape[0]
    for i in range(batch_size):
        prefix = os.path.join(dirname1, basename)
        file_name_pred = prefix + "_pred_3d_" + str(i) + ".png"
        ims = torch.cat([_i[0][None] for _i in inputs])
        max1, min1 = ims.max(), ims.min()
        ims = ((ims - min1) / (max1 - min1)) * 255
        ims = ims.byte().permute(0, 2, 3, 1).cpu().numpy()
        # stupid hack
        ims_pred = [cv2.cvtColor(cv2.cvtColor(_im, cv2.COLOR_RGB2BGR), cv2.COLOR_RGB2BGR) for _im in ims]
        if preds is not None:
            pred = preds[i]
            pts_pred, lines_pred = None, None
            for n in range(len(pred)):
                joints = pred[n].detach().cpu().numpy()
                if n == 0:
                    pts_pred = Points(joints[..., :3], c="w")
                else:
                    pts_pred += Points(joints[..., :3], c="w")
                if joints[0, 3] >= 0:
                    # project 3D points to 2D
                    for ii, mt in enumerate(meta):
                        center = meta[ii]['center'][i]
                        scale = meta[ii]['scale'][i]                        
                        width, height = center * 2
                        trans = torch.as_tensor(
                            get_transform(center, scale, meta[ii]['rotation'][i].item(), img_size),
                            dtype=torch.float,
                            device=pred.device)
                        cam = {}
                        for k, v in mt['camera'].items():
                            cam[k] = deepcopy(v[i])
                        kpt2d = cameras.project_pose(pred[n][..., :3], cam)
                        kpt2d = torch.clamp(kpt2d, -1.0, max(width, height))
                        kpt2d = do_transform(kpt2d, trans)
                        for k in eval("LIMBS{}".format(len(joints))):
                            pt1 = int(kpt2d[k[0], 0]), int(kpt2d[k[0], 1])
                            pt2 = int(kpt2d[k[1], 0]), int(kpt2d[k[1], 1])
                            cv2.line(ims_pred[ii], pt1, pt2, 
                                    [a*255 for a in colors.to_rgb(COLORS[int(n % 10)])], 
                                    3, cv2.LINE_AA)
                            cv2.circle(ims_pred[ii], pt1, 2, (255, 255, 255), 2, cv2.LINE_AA)
                            cv2.circle(ims_pred[ii], pt2, 2, (255, 255, 255), 2, cv2.LINE_AA)
                    pts_start, pts_end = [], []
                    for k in eval("LIMBS{}".format(len(joints))):
                        pts_start.append(
                            (joints[k[0], 0], joints[k[0], 1], joints[k[0], 2])
                        )
                        pts_end.append(
                            (joints[k[1], 0], joints[k[1], 1], joints[k[1], 2])
                        )
                    if n == 0:
                        lines_pred = Lines(pts_start, pts_end, c=COLORS[int(n % 10)])
                    else:
                        lines_pred += Lines(pts_start, pts_end, c=COLORS[int(n % 10)])
            p1 = Picture(ims_pred[0]).rotateX(90).rotateY(0).rotateZ(100)
            p2 = Picture(ims_pred[1]).rotateX(90).rotateY(0).rotateZ(80)
            p3 = Picture(ims_pred[2]).rotateX(90).rotateX(0).rotateZ(0)
            p4 = Picture(ims_pred[3]).rotateX(90).rotateY(0).rotateZ(80)
            p5 = Picture(ims_pred[4]).rotateX(90).rotateY(0).rotateZ(100)
            p1.z(1000).y(-2000).x(-2000).scale(1.3)
            p2.z(1000).y(0).x(-2000).scale(1.3)
            p3.z(1000).y(2000).x(-1000).scale(1.3)
            p4.z(1000).y(-2000).x(2000).scale(1.3)
            p5.z(1000).y(0).x(2000).scale(1.3)
            if render_to_file:
                show(p1, p2, p3, p4, p5, pts_pred, lines_pred,
                    axes=11,
                    bg="k",
                    offscreen=True,
                    viewup="z",
                    azimuth=15,
                    elevation=45,
                    sharecam=1,
                    roll=15,
                    zoom=1.5,
                    size=(1920, 1080)
                ).screenshot(file_name_pred).close()
            else:
                show(p1, p2, p3, p4, p5, pts_pred, lines_pred,
                    axes=11,
                    bg="k",
                    interactive=True,
                    viewup="z",
                    azimuth=15,
                    elevation=45,
                    sharecam=1,
                    roll=15,
                    size=(1920, 1080)
                ).close()

        if plot_gt:
            file_name_gt = prefix + "_gt_3d_" + str(i) + ".png"
            ims_gt = [cv2.cvtColor(cv2.cvtColor(_im, cv2.COLOR_RGB2BGR), cv2.COLOR_RGB2BGR) for _im in ims]
            num_person = meta[0]["num_person"][i]
            joints_3d = meta[0]["joints_3d"][i]
            joints_3d_vis = meta[0]["joints_3d_vis"][i]
            pts_gt, lines_gt, pts_pred, lines_pred = None, None, None, None
            for n in range(num_person):
                joints = joints_3d[n].numpy()
                if n == 0:
                    pts_gt = Points(joints, c="w")
                else:
                    pts_gt += Points(joints, c="w")
                pts_start, pts_end = [], []
                for k in eval("LIMBS{}".format(len(joints))):
                    pts_start.append((joints[k[0], 0], joints[k[0], 1], joints[k[0], 2]))
                    pts_end.append((joints[k[1], 0], joints[k[1], 1], joints[k[1], 2]))
                if n == 0:
                    lines_gt = Lines(pts_start, pts_end, c=COLORS[int(n % 10)])
                else:
                    lines_gt += Lines(pts_start, pts_end, c=COLORS[int(n % 10)])
                for ii in range(len(meta)):
                    kpt2d = meta[ii]["joints"][i][n]
                    for k in eval("LIMBS{}".format(len(joints))):
                        pt1 = int(kpt2d[k[0], 0]), int(kpt2d[k[0], 1])
                        pt2 = int(kpt2d[k[1], 0]), int(kpt2d[k[1], 1])
                        cv2.line(ims_gt[ii], pt1, pt2, 
                                [a*255 for a in colors.to_rgb(COLORS[int(n % 10)])], 
                                1, cv2.LINE_AA)
                        cv2.circle(ims_gt[ii], pt1, 2, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.circle(ims_gt[ii], pt2, 2, (255, 255, 255), 1, cv2.LINE_AA)                    

            p1 = Picture(ims_gt[0]).rotateX(90).rotateY(0).rotateZ(100)
            p2 = Picture(ims_gt[1]).rotateX(90).rotateY(0).rotateZ(80)
            p3 = Picture(ims_gt[2]).rotateX(90).rotateX(0).rotateZ(0)
            p4 = Picture(ims_gt[3]).rotateX(90).rotateY(0).rotateZ(80)
            p5 = Picture(ims_gt[4]).rotateX(90).rotateY(0).rotateZ(100)
            p1.z(1000).y(-2000).x(-2000).scale(1.3)
            p2.z(1000).y(0).x(-2000).scale(1.3)
            p3.z(1000).y(2000).x(-1000).scale(1.3)
            p4.z(1000).y(-2000).x(2000).scale(1.3)
            p5.z(1000).y(0).x(2000).scale(1.3)
            if render_to_file:
                show(p1, p2, p3, p4, p5, pts_gt, lines_gt,
                    axes=11,
                    bg="k",
                    offscreen=True,
                    viewup="z",
                    azimuth=15,
                    elevation=45,
                    sharecam=1,
                    roll=15,
                    zoom=1.5,
                    size=(1920, 1080)
                ).screenshot(file_name_gt).close()
            else:
                show(p1, p2, p3, p4, p5, pts_gt, lines_gt,
                    axes=11,
                    bg="k",
                    interactive=True,
                    viewup="z",
                    azimuth=15,
                    elevation=45,
                    sharecam=1,
                    roll=15,
                    zoom=1.5,
                    size=(1920, 1080)
                ).close()           
            

def save_debug_3d_images(config, meta, preds, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, "3d_joints")

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_3d.png"

    # preds = preds.cpu().numpy()
    batch_size = meta["num_person"].shape[0]
    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 4.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.15
    )
    for i in range(batch_size):
        num_person = meta["num_person"][i]
        if "joints_3d" in meta:
            joints_3d = meta["joints_3d"][i]
            joints_3d_vis = meta["joints_3d_vis"][i]
            ax = plt.subplot(yplot, xplot, i + 1, projection="3d")
            for n in range(num_person):
                joint = joints_3d[n]
                joint_vis = joints_3d_vis[n]
                for k in eval("LIMBS{}".format(len(joint))):
                    if joint_vis[k[0], 0] and joint_vis[k[1], 0]:
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c='r', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
                    else:
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(x, y, z, c='r', ls='--', lw=1.5, marker='o', markerfacecolor='w', markersize=2,
                                markeredgewidth=1)
        if preds is not None:
            pred = preds[i]
            for n in range(len(pred)):
                joint = pred[n]
                if joint[0, 3] >= 0:
                    for k in eval("LIMBS{}".format(len(joint))):
                        x = [float(joint[k[0], 0]), float(joint[k[1], 0])]
                        y = [float(joint[k[0], 1]), float(joint[k[1], 1])]
                        z = [float(joint[k[0], 2]), float(joint[k[1], 2])]
                        ax.plot(
                            x,
                            y,
                            z,
                            c=COLORS[int(n % 10)],
                            lw=1.5,
                            marker="o",
                            markerfacecolor="w",
                            markersize=2,
                            markeredgewidth=1,
                        )
    plt.savefig(file_name)
    plt.close(0)


def save_debug_3d_cubes(config, meta, root, prefix):
    if not config.DEBUG.DEBUG:
        return

    basename = os.path.basename(prefix)
    dirname = os.path.dirname(prefix)
    dirname1 = os.path.join(dirname, "root_cubes")

    if not os.path.exists(dirname1):
        os.makedirs(dirname1)

    prefix = os.path.join(dirname1, basename)
    file_name = prefix + "_root.png"

    batch_size = root.shape[0]
    root_id = config.DATASET.ROOTIDX

    xplot = min(4, batch_size)
    yplot = int(math.ceil(float(batch_size) / xplot))

    width = 6.0 * xplot
    height = 4.0 * yplot
    fig = plt.figure(0, figsize=(width, height))
    plt.subplots_adjust(
        left=0.05, right=0.95, bottom=0.05, top=0.95, wspace=0.05, hspace=0.15
    )
    for i in range(batch_size):
        roots_gt = meta["roots_3d"][i]
        num_person = meta["num_person"][i]
        roots_pred = root[i]
        ax = plt.subplot(yplot, xplot, i + 1, projection="3d")

        x = roots_gt[:num_person, 0].cpu()
        y = roots_gt[:num_person, 1].cpu()
        z = roots_gt[:num_person, 2].cpu()
        ax.scatter(x, y, z, c="r")

        index = roots_pred[:, 3] >= 0
        x = roots_pred[index, 0].cpu()
        y = roots_pred[index, 1].cpu()
        z = roots_pred[index, 2].cpu()
        ax.scatter(x, y, z, c="b")

        space_size = config.MULTI_PERSON.SPACE_SIZE
        space_center = config.MULTI_PERSON.SPACE_CENTER
        ax.set_xlim(
            space_center[0] - space_size[0] / 2, space_center[0] + space_size[0] / 2
        )
        ax.set_ylim(
            space_center[1] - space_size[1] / 2, space_center[1] + space_size[1] / 2
        )
        ax.set_zlim(
            space_center[2] - space_size[2] / 2, space_center[2] + space_size[2] / 2
        )

    plt.savefig(file_name)
    plt.close(0)
