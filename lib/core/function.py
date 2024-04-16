'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import logging
import os
import copy

import torch
import numpy as np

from utils.vis import save_debug_images_multi
from utils.vis import save_debug_3d_images
from utils.vis import save_debug_3d_cubes
from utils.vis import save_debug_3d_images_all

logger = logging.getLogger(__name__)


def train_3d_ssv(config, model, optimizer, loader, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses_meter = {
        "loss_2d": AverageMeter(),
        "loss_root_reg": AverageMeter(),
        "loss_root_syn": AverageMeter(),
        "loss_pose3d_ssv": AverageMeter(),
        "loss_attn_ssv": AverageMeter(),
        "loss_pose3d_l1_ssv": AverageMeter(),
        "losses": AverageMeter(),
    }

    model.train()
    if not config.NETWORK.TRAIN_BACKBONE:
        if model.module.backbone is not None:
            model.module.backbone.eval()
    if config.NETWORK.FREEZE_ROOTNET:
        if model.module.root_net is not None:
            model.module.root_net.eval()

    end = time.time()
    for i, (
        inputs1,
        targets_2d1,
        weights_2d1,
        targets_3d1,
        meta1,
        input_heatmap1,
        inputs2,
        targets_2d2,
        weights_2d2,
        targets_3d2,
        meta2,
        input_heatmap2,
        inputs3,
        targets_2d3,
        weights_2d3,
        targets_3d3,
        meta3,
        input_heatmap3,
    ) in enumerate(loader):
        data_time.update(time.time() - end)

        if "panoptic" in config.DATASET.TEST_DATASET or "shelf" in config.DATASET.TEST_DATASET or "campus" in config.DATASET.TEST_DATASET:
            pred2, heatmaps3, grid_centers, loss_dict = model(
                views1=inputs1,
                meta1=meta1,
                targets_2d1=targets_2d1,
                weights_2d1=weights_2d1,
                targets_3d1=targets_3d1[0],
                views2=inputs2,
                meta2=meta2,
                targets_2d2=targets_2d2,
                weights_2d2=weights_2d2,
                targets_3d2=targets_3d2[0],
                views3=inputs3,
                meta3=meta3,
                targets_2d3=targets_2d3,
                weights_2d3=weights_2d3,
                targets_3d3=targets_3d3[0],
                epoch=epoch,
            )
        else:
            print("do it later")
            assert False

        loss = sum([v.mean() for v in loss_dict.values() if v.requires_grad])
        for k, v in loss_dict.items():
            losses_meter[k].update(v.mean())
        losses_meter["losses"].update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if loss_cord > 0:
        #     optimizer.zero_grad()
        #     (loss_2d + loss_cord).backward()
        #     optimizer.step()

        # if accu_loss_3d > 0 and (i + 1) % accumulation_steps == 0:
        #     optimizer.zero_grad()
        #     accu_loss_3d.backward()
        #     optimizer.step()
        #     accu_loss_3d = 0.0
        # else:
        #     accu_loss_3d += loss_3d / accumulation_steps

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0)
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                "Speed: {speed:.1f} samples/s\t"
                "Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                "Loss: {loss.val:.6f} ({loss.avg:.6f})\t"
                "Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t"
                "loss_root_syn: {loss_root_syn.val:.7f} ({loss_root_syn.avg:.7f})\t"
                "loss_root_reg: {loss_root_reg.val:.7f} ({loss_root_reg.avg:.7f})\t"
                "loss_pose3d_ssv: {loss_pose3d_ssv.val:.6f} ({loss_pose3d_ssv.avg:.6f})\t"
                "loss_attn_ssv: {loss_attn_ssv.val:.6f} ({loss_attn_ssv.avg:.6f})\t"
                "loss_pose3d_l1_ssv: {loss_pose3d_l1_ssv.val:.6f} ({loss_pose3d_l1_ssv.avg:.6f})\t"
                "missed countes: {mis_count}\t"
                "Memory {memory:.1f}".format(
                    epoch,
                    i,
                    len(loader),
                    batch_time=batch_time,
                    speed=len(inputs1) * inputs1[0].size(0) / batch_time.val,
                    data_time=data_time,
                    loss=losses_meter["losses"],
                    loss_2d=losses_meter["loss_2d"],
                    loss_root_syn=losses_meter["loss_root_syn"],
                    loss_root_reg=losses_meter["loss_root_reg"],
                    loss_pose3d_ssv=losses_meter["loss_pose3d_ssv"],
                    loss_attn_ssv=losses_meter["loss_attn_ssv"],
                    loss_pose3d_l1_ssv=losses_meter["loss_pose3d_l1_ssv"],
                    mis_count=meta1[0]["mis_count"].sum().item(),
                    memory=gpu_memory_usage,
                )
            )
            logger.info(msg)

            writer = writer_dict["writer"]
            global_steps = writer_dict["train_global_steps"]
            writer.add_scalar("train_loss_2d", losses_meter["loss_2d"].val, global_steps)
            writer.add_scalar(
                "train_loss_root",
                losses_meter["loss_root_syn"].val + losses_meter["loss_root_reg"].val,
                global_steps,
            )
            writer.add_scalar(
                "train_loss_pose3d_ssv",
                losses_meter["loss_pose3d_ssv"].val,
                global_steps,
            )
            writer.add_scalar(
                "train_loss_attn_ssv",
                losses_meter["loss_attn_ssv"].val,
                global_steps,
            )
            writer.add_scalar("train_loss", losses_meter["losses"].val, global_steps)
            writer_dict["train_global_steps"] = global_steps + 1

            for k in range(len(inputs3)):
                view_name = "view_{}".format(k + 1)
                prefix = "{}_{:08}_{}".format(os.path.join(output_dir, "train"), i, view_name)
                save_debug_images_multi(
                    config, inputs3[k], meta3[k], targets_2d3[k], heatmaps3[k], prefix
                )

            prefix = "{}_{:03}".format(os.path.join(output_dir, "train"), i)
            if config.DEBUG.SAVE_3D_ROOTS:
                if 'uncalibrated' in config.DATASET.TRAIN_DATASET:
                    save_debug_3d_cubes(config, meta3[0], grid_centers, prefix, True)
                else:
                    save_debug_3d_cubes(config, meta3[0], grid_centers, prefix)
            if config.DEBUG.SAVE_3D_POSES:
                if 'uncalibrated' in config.DATASET.TRAIN_DATASET:
                    save_debug_3d_images(config, meta2[0], pred2, prefix, True)
                else:
                    save_debug_3d_images(config, meta2[0], pred2, prefix)

            # for ii, (meta, grid_centers, pred,) in enumerate(
            #     zip(
            #         (meta1, meta2),
            #         grid_centers_all,
            #         pred_all,
            #     )
            # ):
            # for k in range(len(inputs1)):
            #     view_name = "view_{}".format(k + 1)
            #
            #     prefix = "{}_{:03}_{:08}_{}_{}".format(
            #         os.path.join(output_dir, "train"), epoch, i, view_name, aug_name
            #     )
            #     save_debug_images_multi(
            #         config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix
            #     )
            # aug_name = "aug_{}".format(ii + 1)
            # prefix2 = "{}_{:03}_{:08}_{}".format(os.path.join(output_dir, "train"), epoch, i, aug_name)
            # save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
            # save_debug_3d_images(config, meta[0], pred, prefix2)
            # save_debug_3d_images_all(
            #     config, meta, pred, inputs, targets_2d, heatmaps, prefix
            # )

def train_3d(config, model, optimizer, loader, epoch, output_dir, writer_dict):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_2d = AverageMeter()
    losses_3d = AverageMeter()
    losses_cord = AverageMeter()

    model.train()
    if not config.NETWORK.TRAIN_BACKBONE:
        if model.module.backbone is not None:
            model.module.backbone.eval()  # Comment out this line if you want to train 2D

    accumulation_steps = 4
    accu_loss_3d = 0

    end = time.time()
    for i, (
        inputs,
        targets_2d,
        weights_2d,
        targets_3d,
        meta,
        input_heatmap,
    ) in enumerate(loader):
        data_time.update(time.time() - end)

        if "panoptic" in config.DATASET.TEST_DATASET or "shelf" in config.DATASET.TEST_DATASET:
            if config.NETWORK.TRAIN_ONLY_2D:
                loss_2d, heatmaps = model(
                    views=inputs,
                    meta=meta,
                    targets_2d=targets_2d,
                    weights_2d=weights_2d,
                    targets_3d=None,
                )
            else:
                pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(
                    views=inputs,
                    meta=meta,
                    targets_2d=targets_2d,
                    weights_2d=weights_2d,
                    targets_3d=targets_3d[0],
                )
        elif "campus" in config.DATASET.TEST_DATASET:
            pred, heatmaps, grid_centers, loss_2d, loss_3d, loss_cord = model(
                meta=meta, targets_3d=targets_3d[0], input_heatmaps=input_heatmap
            )

        if config.NETWORK.TRAIN_ONLY_2D:
            loss_2d = loss_2d.mean()
            losses_2d.update(loss_2d.item())
            loss = loss_2d
            losses.update(loss.item())
        else:
            loss_2d = loss_2d.mean()
            loss_3d = loss_3d.mean()
            loss_cord = loss_cord.mean()
            losses_3d.update(loss_3d.item())
            losses_cord.update(loss_cord.item())
            loss = loss_2d + loss_3d + loss_cord
            losses.update(loss.item())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if loss_cord > 0:
        #     optimizer.zero_grad()
        #     (loss_2d + loss_cord).backward()
        #     optimizer.step()

        # if accu_loss_3d > 0 and (i + 1) % accumulation_steps == 0:
        #     optimizer.zero_grad()
        #     accu_loss_3d.backward()
        #     optimizer.step()
        #     accu_loss_3d = 0.0
        # else:
        #     accu_loss_3d += loss_3d / accumulation_steps

        batch_time.update(time.time() - end)
        end = time.time()

        if i % config.PRINT_FREQ == 0:
            gpu_memory_usage = torch.cuda.memory_allocated(0) / 1024.0 / 1024.0 / 1024.0
            msg = (
                "Epoch: [{0}][{1}/{2}]\t"
                "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                "Speed: {speed:.1f} samples/s\t"
                "Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                "Loss: {loss.val:.6f} ({loss.avg:.6f})\t"
                "Loss_2d: {loss_2d.val:.7f} ({loss_2d.avg:.7f})\t"
                "Loss_3d: {loss_3d.val:.7f} ({loss_3d.avg:.7f})\t"
                "Loss_cord: {loss_cord.val:.6f} ({loss_cord.avg:.6f})\t"
                "Memory {memory:.1f} GB".format(
                    epoch,
                    i,
                    len(loader),
                    batch_time=batch_time,
                    speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                    data_time=data_time,
                    loss=losses,
                    loss_2d=losses_2d,
                    loss_3d=losses_3d,
                    loss_cord=losses_cord,
                    memory=gpu_memory_usage,
                )
            )
            logger.info(msg)

            writer = writer_dict["writer"]
            global_steps = writer_dict["train_global_steps"]
            writer.add_scalar("train_loss_3d", losses_3d.val, global_steps)
            writer.add_scalar("train_loss_cord", losses_cord.val, global_steps)
            writer.add_scalar("train_loss", losses.val, global_steps)
            writer_dict["train_global_steps"] = global_steps + 1

            for k in range(len(inputs)):
                view_name = "view_{}".format(k + 1)
                prefix = "{}_{:08}_{}".format(os.path.join(output_dir, "train"), i, view_name)
                save_debug_images_multi(
                    config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix
                )
            prefix2 = "{}_{:08}".format(os.path.join(output_dir, "train"), i)

            if not config.NETWORK.TRAIN_ONLY_2D:
                save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
                save_debug_3d_images(config, meta[0], pred, prefix2)
            # save_debug_3d_images_all(
            #    config, meta, pred, inputs, targets_2d, heatmaps, prefix
            # )


def validate_3d(config, model, loader, epoch, output_dir, with_ssv=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    model.eval()
    preds = []
    roots = []
    with torch.no_grad():
        end = time.time()
        for i, (
            inputs,
            targets_2d,
            weights_2d,
            targets_3d,
            meta,
            input_heatmap,
        ) in enumerate(loader):
            data_time.update(time.time() - end)
            if "panoptic" in config.DATASET.TEST_DATASET or "shelf" in config.DATASET.TEST_DATASET or "campus" in config.DATASET.TEST_DATASET:
                if with_ssv:
                    pred, heatmaps, grid_centers = model(
                        views1=inputs,
                        meta1=meta,
                        input_heatmaps1=input_heatmap,
                        inference=True,
                    )
                else:
                    if config.NETWORK.TRAIN_ONLY_2D:
                        _, heatmaps = model(
                            views=inputs,
                            meta=meta,
                            targets_2d=targets_2d,
                            weights_2d=weights_2d,
                            targets_3d=None,
                        )
                    else:
                        pred, heatmaps, grid_centers, _, _, _ = model(
                            views=inputs,
                            meta=meta,
                            targets_2d=targets_2d,
                            weights_2d=weights_2d,
                            targets_3d=targets_3d[0],
                        )
            if not config.NETWORK.TRAIN_ONLY_2D:
                for b in range(pred.shape[0]):
                    preds.append(pred[b].detach().cpu().numpy().copy())
                    roots.append(grid_centers[b].detach().cpu().numpy().copy())

            batch_time.update(time.time() - end)
            end = time.time()
            if i % config.PRINT_FREQ == 0 or i == len(loader) - 1:
                gpu_memory_usage = torch.cuda.memory_allocated(0)
                msg = (
                    "Test: [{0}/{1}]\t"
                    "Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t"
                    "Speed: {speed:.1f} samples/s\t"
                    "Data: {data_time.val:.3f}s ({data_time.avg:.3f}s)\t"
                    "Memory {memory:.1f}".format(
                        i,
                        len(loader),
                        batch_time=batch_time,
                        speed=len(inputs) * inputs[0].size(0) / batch_time.val,
                        data_time=data_time,
                        memory=gpu_memory_usage,
                    )
                )
                logger.info(msg)

                for k in range(len(inputs)):
                    view_name = "view_{}".format(k + 1)
                    prefix = "{}_{:08}_{}".format(
                        os.path.join(output_dir, "validation"), i, view_name
                    )
                    save_debug_images_multi(
                        config, inputs[k], meta[k], targets_2d[k], heatmaps[k], prefix
                    )
                prefix2 = "{}_{:03}_{:08}".format(os.path.join(output_dir, "validation"), epoch, i)
                if not config.NETWORK.TRAIN_ONLY_2D:
                    save_debug_3d_cubes(config, meta[0], grid_centers, prefix2)
                    save_debug_3d_images(config, meta[0], pred, prefix2)
                # save_debug_3d_images_all(
                #     config, meta, pred, inputs, targets_2d, heatmaps, prefix
                # )

    metric = None
    if config.NETWORK.TRAIN_ONLY_2D:
        msg = "training only the backbone; no evaluation for this part"
        logger.info(msg)
    else:
        if "panoptic" in config.DATASET.TEST_DATASET:
            aps_all, _, mpjpe_all, recall_all = loader.dataset.evaluate(preds, roots, output_dir)
            aps, aps_root = aps_all[0], aps_all[1]
            mpjpe, mpjpe_root = mpjpe_all[0], mpjpe_all[1]
            recall, recall_root = recall_all[0], recall_all[1]
            msg = (
                "ap@25: {aps_25:.4f}\tap@50: {aps_50:.4f}\tap@75: {aps_75:.4f}\t"
                "ap@100: {aps_100:.4f}\tap@125: {aps_125:.4f}\tap@150: {aps_150:.4f}\t"
                "recall@500mm: {recall:.4f}\tmpjpe@500mm: {mpjpe:.3f}".format(
                    aps_25=aps[0],
                    aps_50=aps[1],
                    aps_75=aps[2],
                    aps_100=aps[3],
                    aps_125=aps[4],
                    aps_150=aps[5],
                    recall=recall,
                    mpjpe=mpjpe,
                )
            )
            logger.info(msg)
            msg = (
                "Root eval\n ap_root@25: {aps_25:.4f}\tap_root@50: {aps_50:.4f}\tap_root@75: {aps_75:.4f}\t"
                "ap_root@100: {aps_100:.4f}\tap_root@125: {aps_125:.4f}\tap_root@150: {aps_150:.4f}\t"
                "recall_root@500mm: {recall:.4f}\tmpjpe_root@500mm: {mpjpe:.3f}".format(
                    aps_25=aps_root[0],
                    aps_50=aps_root[1],
                    aps_75=aps_root[2],
                    aps_100=aps_root[3],
                    aps_125=aps_root[4],
                    aps_150=aps_root[5],
                    recall=recall_root,
                    mpjpe=mpjpe_root,
                )
            )
            logger.info(msg)
            metric = np.mean(aps)

        elif "campus" in config.DATASET.TEST_DATASET or "shelf" in config.DATASET.TEST_DATASET:
            actor_pcp, avg_pcp, _, recall = loader.dataset.evaluate(preds)
            msg = "     | Actor 1 | Actor 2 | Actor 3 | Average | \n" " PCP |  {pcp_1:.2f}  |  {pcp_2:.2f}  |  {pcp_3:.2f}  |  {pcp_avg:.2f}  |\t Recall@500mm: {recall:.4f}".format(
                pcp_1=actor_pcp[0] * 100,
                pcp_2=actor_pcp[1] * 100,
                pcp_3=actor_pcp[2] * 100,
                pcp_avg=avg_pcp * 100,
                recall=recall,
            )
            logger.info(msg)
            metric = np.mean(avg_pcp)

    return metric


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
