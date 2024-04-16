#!/usr/bin/env bash

# build HR-Net from https://github.com/HRNet/HRNet-Human-Pose-Estimation
HRNET_PATH=/path/to/hrnet
cd ${/path/to/hrnet}

CONFIG_FILE=experiments/coco/hrnet/w48_384x288_adam_lr1e-3.yaml
MODEL_WEIGHTS=/path/to/hrnet_weights/pose_hrnet_w48_384x288.pth


TEST_JSON="./pseudo_labels/pseudo_bboxes_panoptic.json"
OUTPUT_DIR="./pseudo_labels/kpt2d_inference"
TEST_IMG_DIR="./voxelpose-pytorch/data"

mkdir -p ${OUTPUT_DIR}
LOG=${OUTPUT_DIR}/eval.log
python tools/test.py \
    --cfg ${CONFIG_FILE} \
    TEST.MODEL_FILE ${MODEL_WEIGHTS}  \
    TEST.USE_GT_BBOX True \
    DATASET.TEST_JSON ${TEST_JSON} \
    DATASET.TEST_IMGDIR ${TEST_IMG_DIR}\
    GPUS 0, \
    OUTPUT_DIR ${OUTPUT_DIR} 2>&1 | tee ${LOG}