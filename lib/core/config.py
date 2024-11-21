'''
Project: SelfPose3d
-----
Copyright (c) University of Strasbourg, All Rights Reserved.
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict

config = edict()

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.BACKBONE_MODEL = 'pose_resnet'
config.MODEL = 'multi_person_posenet'
config.GPUS = '0,1'
config.WORKERS = 8
config.PRINT_FREQ = 100
config.WITH_SSV = False
config.WITH_ATTN = False
config.ATTN_WEIGHT = 0.1
config.ATTN_NUM_LAYERS = 18
config.USE_L1 = False
config.L1_WEIGHT = 0.1
config.L1_ATTN = False
config.MIN_VIEWS_CHECK = 1
config.EVAL_ROOTNET_ONLY = False
config.COCO_TO_PANOPTIC_MAPPING = [5, 0, 11, 5, 7, 9, 11, 13, 15, 6, 8, 10, 12, 14, 16]
# higherhrnet definition
config.MODEL_EXTRA = edict()
config.MODEL_EXTRA.PRETRAINED_LAYERS = ['*']
config.MODEL_EXTRA.FINAL_CONV_KERNEL = 1
config.MODEL_EXTRA.STEM_INPLANES = 64

config.MODEL_EXTRA.STAGE2 = edict()
config.MODEL_EXTRA.STAGE2.NUM_MODULES = 1
config.MODEL_EXTRA.STAGE2.NUM_BRANCHES= 2
config.MODEL_EXTRA.STAGE2.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE2.NUM_BLOCKS = [4, 4]
config.MODEL_EXTRA.STAGE2.NUM_CHANNELS = [48, 96]
config.MODEL_EXTRA.STAGE2.FUSE_METHOD = 'SUM'


config.MODEL_EXTRA.STAGE3 = edict()
config.MODEL_EXTRA.STAGE3.NUM_MODULES = 4
config.MODEL_EXTRA.STAGE3.NUM_BRANCHES = 3
config.MODEL_EXTRA.STAGE3.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE3.NUM_BLOCKS = [4, 4, 4]
config.MODEL_EXTRA.STAGE3.NUM_CHANNELS = [48, 96, 192]
config.MODEL_EXTRA.STAGE3.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.STAGE4 = edict()
config.MODEL_EXTRA.STAGE4.NUM_MODULES = 3
config.MODEL_EXTRA.STAGE4.NUM_BRANCHES = 4
config.MODEL_EXTRA.STAGE4.BLOCK = 'BASIC'
config.MODEL_EXTRA.STAGE4.NUM_BLOCKS = [4, 4, 4, 4]
config.MODEL_EXTRA.STAGE4.NUM_CHANNELS = [48, 96, 192, 384]
config.MODEL_EXTRA.STAGE4.FUSE_METHOD = 'SUM'

config.MODEL_EXTRA.DECONV = edict()
config.MODEL_EXTRA.DECONV.NUM_DECONVS = 1
config.MODEL_EXTRA.DECONV.NUM_CHANNELS = 32
config.MODEL_EXTRA.DECONV.KERNEL_SIZE = 4
config.MODEL_EXTRA.DECONV.NUM_BASIC_BLOCKS = 4
config.MODEL_EXTRA.DECONV.CAT_OUTPUT = True

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = edict()
config.NETWORK.PRETRAINED = 'models/pytorch/imagenet/resnet50-19c8e357.pth'
config.NETWORK.PRETRAINED_BACKBONE = ''
config.NETWORK.PRETRAINED_BACKBONE_PSEUDOGT = False
config.NETWORK.TRAIN_BACKBONE = False
config.NETWORK.TRAIN_ONLY_2D = False
config.NETWORK.NUM_JOINTS = 20
config.NETWORK.INPUT_SIZE = 512
config.NETWORK.HEATMAP_SIZE = np.array([80, 80])
config.NETWORK.IMAGE_SIZE = np.array([320, 320])
config.NETWORK.IMAGE_SIZE_ORIG = np.array([1920, 1080])
config.NETWORK.SIGMA = 2
config.NETWORK.TARGET_TYPE = 'gaussian'
config.NETWORK.AGGRE = True
config.NETWORK.USE_GT = False
config.NETWORK.BETA = 100.0
config.NETWORK.ROOTNET_ROOTHM = False
config.NETWORK.ROOTNET_TRAIN_SYNTH = False
config.NETWORK.INIT_TRAIN_EPOCHS_ROOTNET = 0
config.NETWORK.INIT_ROOTNET = ""
config.NETWORK.TRAIN_ONLY_ROOTNET = False
config.NETWORK.ROOTNET_BUFFER_SIZE = 5000
config.NETWORK.FREEZE_ROOTNET = False
config.NETWORK.INIT_ALL = ""
config.NETWORK.SINGLE_AUG_TRAINING_POSENET = False
config.NETWORK.ROOT_CONSISTENCY_LOSS = True
config.NETWORK.WEIGHT_ROOT_SYN = 100.0
config.NETWORK.WEIGHT_ROOT_REG = 1.0
config.NETWORK.ROOTNET_SYN_RANGE = np.array([[2500.0, -2000.0], [1500.0, -1500.0], [250.0, -300.0]])

# pose_resnet related params
config.POSE_RESNET = edict()
config.POSE_RESNET.NUM_LAYERS = 50
config.POSE_RESNET.DECONV_WITH_BIAS = False
config.POSE_RESNET.NUM_DECONV_LAYERS = 3
config.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
config.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
config.POSE_RESNET.FINAL_CONV_KERNEL = 1

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.USE_DIFFERENT_JOINTS_WEIGHT = False

# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = '../data/h36m/'
config.DATASET.TRAIN_DATASET = 'mixed_dataset'
config.DATASET.TEST_DATASET = 'multi_view_h36m'
config.DATASET.TRAIN_SUBSET = 'train'
config.DATASET.TEST_SUBSET = 'validation'
config.DATASET.ROOTIDX = 2
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.BBOX = 2000
config.DATASET.CROP = True
config.DATASET.COLOR_RGB = False
config.DATASET.FLIP = True
config.DATASET.DATA_AUGMENTATION = True
config.DATASET.CAMERA_NUM = 5
config.DATASET.CAMERAS = [0,1,2,3,4]
config.DATASET.CAMERA_NUM_TOTAL = 5

# training data augmentation
config.DATASET.SCALE_FACTOR = 0
config.DATASET.SCALE_FACTOR1 = 0
config.DATASET.SCALE_FACTOR2 = 0
config.DATASET.ROT_FACTOR = 0
config.DATASET.ROT_FACTOR1 = 0
config.DATASET.ROT_FACTOR2 = 0
config.DATASET.APPLY_CUTOUT = False
config.DATASET.APPLY_RANDAUG = False
config.DATASET.SUFFIX = "sub"
config.DATASET.GT_3D_FILE = "panoptic_training_pose.pkl"
config.DATASET.TRAIN_PSEUDO_GT3D = False
config.DATASET.ROOTIDX_PSEUDO = 2

# train
config.TRAIN = edict()
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001
config.TRAIN.L1_EPOCH = 5

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = False

config.TRAIN.BATCH_SIZE = 8
config.TRAIN.SHUFFLE = True

# testing
config.TEST = edict()
config.TEST.BATCH_SIZE = 8
config.TEST.STATE = 'best'
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = False
config.TEST.SHIFT_HEATMAP = False
config.TEST.USE_GT_BBOX = False
config.TEST.IMAGE_THRE = 0.1
config.TEST.NMS_THRE = 0.6
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MATCH_IOU_THRE = 0.3
config.TEST.DETECTOR = 'fpn_dcn'
config.TEST.DETECTOR_DIR = ''
config.TEST.MODEL_FILE = ''
config.TEST.HEATMAP_LOCATION_FILE = 'predicted_heatmaps.h5'

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES_GT = True
config.DEBUG.SAVE_BATCH_IMAGES_PRED = True
config.DEBUG.SAVE_HEATMAPS_GT = True
config.DEBUG.SAVE_HEATMAPS_PRED = True
config.DEBUG.SAVE_3D_POSES = False
config.DEBUG.SAVE_3D_ROOTS = False

# pictorial structure
config.PICT_STRUCT = edict()
config.PICT_STRUCT.FIRST_NBINS = 16
config.PICT_STRUCT.PAIRWISE_FILE = ''
config.PICT_STRUCT.RECUR_NBINS = 2
config.PICT_STRUCT.RECUR_DEPTH = 10
config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE = 150
config.PICT_STRUCT.GRID_SIZE = np.array([2000.0, 2000.0, 2000.0])
config.PICT_STRUCT.CUBE_SIZE = np.array([64, 64, 64])
config.PICT_STRUCT.DEBUG = False
config.PICT_STRUCT.TEST_PAIRWISE = False
config.PICT_STRUCT.SHOW_ORIIMG = False
config.PICT_STRUCT.SHOW_CROPIMG = False
config.PICT_STRUCT.SHOW_HEATIMG = False

config.MULTI_PERSON = edict()
config.MULTI_PERSON.SPACE_SIZE = np.array([4000.0, 5200.0, 2400.0])
config.MULTI_PERSON.SPACE_CENTER = np.array([300.0, 300.0, 300.0])
config.MULTI_PERSON.ESTIMATED_SPACE_CENTER = np.array([300.0, 300.0, 300.0])
config.MULTI_PERSON.INITIAL_CUBE_SIZE = np.array([24, 32, 16])
config.MULTI_PERSON.MAX_PEOPLE_NUM = 10
config.MULTI_PERSON.THRESHOLD = 0.1


def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array(
                [eval(x) if isinstance(x, str) else x for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array(
                    [v['HEATMAP_SIZE'], v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.BBOX_FILE = os.path.join(config.DATA_DIR, config.TEST.BBOX_FILE)

    config.NETWORK.PRETRAINED = os.path.join(config.DATA_DIR,
                                             config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    name = '{model}_{num_layers}'.format(
        model=cfg.MODEL, num_layers=cfg.POSE_RESNET.NUM_LAYERS)
    deconv_suffix = ''.join(
        'd{}'.format(num_filters)
        for num_filters in cfg.POSE_RESNET.NUM_DECONV_FILTERS)
    full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
        height=cfg.NETWORK.IMAGE_SIZE[1],
        width=cfg.NETWORK.IMAGE_SIZE[0],
        name=name,
        deconv_suffix=deconv_suffix)

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
