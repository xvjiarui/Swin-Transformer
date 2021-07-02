# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------'

import os
import yaml
from yacs.config import CfgNode as CN

_C = CN()

# Base config files
_C.BASE = ['']

# -----------------------------------------------------------------------------
# Data settings
# -----------------------------------------------------------------------------
_C.DATA = CN()
# Batch size for a single GPU, could be overwritten by command line argument
_C.DATA.BATCH_SIZE = 128
# Path to dataset, could be overwritten by command line argument
_C.DATA.DATA_PATH = ''
# Dataset name
_C.DATA.DATASET = 'imagenet'
# Input image size
_C.DATA.IMG_SIZE = 224
# Interpolation to resize image (random, bilinear, bicubic)
_C.DATA.INTERPOLATION = 'bicubic'
# Use zipped dataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.ZIP_MODE = False
# Use WebDataset instead of folder dataset
# could be overwritten by command line argument
_C.DATA.WEB_MODE = False
# Thomas said it should be at least about 5-10x your batch size; beyond that,
# the differences become academic.
_C.DATA.SHUFFLE_BUFFER = 10000
# Cache Data in Memory, could be overwritten by command line argument
_C.DATA.CACHE_MODE = 'part'
# Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.
_C.DATA.PIN_MEMORY = True
# Number of data loading threads
_C.DATA.NUM_WORKERS = 8

# -----------------------------------------------------------------------------
# Model settings
# -----------------------------------------------------------------------------
_C.MODEL = CN()
# Model type
_C.MODEL.TYPE = ''
# Model name
_C.MODEL.NAME = ''
# Checkpoint to resume, could be overwritten by command line argument
_C.MODEL.RESUME = ''
# Number of classes, overwritten in data preparation
_C.MODEL.NUM_CLASSES = 1000
# Dropout rate
_C.MODEL.DROP_RATE = 0.0
# Drop path rate
_C.MODEL.DROP_PATH_RATE = 0.1
# Label Smoothing
_C.MODEL.LABEL_SMOOTHING = 0.1

# self distillation
# use distillation if alpha is greater than 0
_C.MODEL.SELF_DIST_ALPHA = 0.
_C.MODEL.SELF_DIST_TYPE = 'soft'
_C.MODEL.SELF_DIST_TAU = 1.

# Swin Transformer parameters
_C.MODEL.SWIN = CN()
_C.MODEL.SWIN.PATCH_SIZE = 4
_C.MODEL.SWIN.IN_CHANS = 3
_C.MODEL.SWIN.EMBED_DIM = 96
_C.MODEL.SWIN.DEPTHS = [2, 2, 6, 2]
_C.MODEL.SWIN.NUM_HEADS = [3, 6, 12, 24]
_C.MODEL.SWIN.WINDOW_SIZE = 7
_C.MODEL.SWIN.MLP_RATIO = 4.
_C.MODEL.SWIN.QKV_BIAS = True
_C.MODEL.SWIN.QK_SCALE = None
_C.MODEL.SWIN.APE = False
_C.MODEL.SWIN.PATCH_NORM = True

# Group Transformer parameters
_C.MODEL.GVIT = CN()
_C.MODEL.GVIT.IN_CHANS = 3
_C.MODEL.GVIT.BASE_DIM = 96
_C.MODEL.GVIT.DIM_PER_HEAD = 96
_C.MODEL.GVIT.STAGE_BLOCKS = [1, 2, 11, 2]
_C.MODEL.GVIT.MLP_RATIO = 4.
_C.MODEL.GVIT.QKV_BIAS = True
_C.MODEL.GVIT.QK_SCALE = None
_C.MODEL.GVIT.PATCH_NORM = True
_C.MODEL.GVIT.CLUSTER_TOKENS = [64, 32, 16, 8]
_C.MODEL.GVIT.DOWNSAMPLE_TYPE='token_assign'

# Vanilla ViT
_C.MODEL.VIT = CN()
_C.MODEL.VIT.PATCH_SIZE = 16
_C.MODEL.VIT.IN_CHANS = 3
_C.MODEL.VIT.EMBED_DIM = 768
_C.MODEL.VIT.DEPTH = 12
_C.MODEL.VIT.NUM_HEADS = 12
_C.MODEL.VIT.MLP_RATIO = 4.
_C.MODEL.VIT.QKV_BIAS = True
_C.MODEL.VIT.QK_SCALE = None
_C.MODEL.VIT.PATCH_NORM = True
_C.MODEL.VIT.WITH_GAP = False
_C.MODEL.VIT.WITH_PEG = False

# Recurrent ViT
_C.MODEL.RVIT = CN()
_C.MODEL.RVIT.RECURRENCES = [12]
_C.MODEL.RVIT.PATCH_SIZE = 16
_C.MODEL.RVIT.IN_CHANS = 3
_C.MODEL.RVIT.EMBED_DIM = 768
_C.MODEL.RVIT.DEPTH = 1
_C.MODEL.RVIT.NUM_HEADS = 12
_C.MODEL.RVIT.MLP_RATIO = 4.
_C.MODEL.RVIT.QKV_BIAS = True
_C.MODEL.RVIT.QK_SCALE = None
_C.MODEL.RVIT.PATCH_NORM = True
_C.MODEL.RVIT.WITH_GAP = False
_C.MODEL.RVIT.WITH_PEG = False

# MViT
_C.MODEL.MVIT = CN()
_C.MODEL.MVIT.IN_CHANS = 3
_C.MODEL.MVIT.EMBED_DIM = 96
_C.MODEL.MVIT.DEPTHS = [1, 1, 10, 1]
_C.MODEL.MVIT.DIM_PER_HEAD = 96
_C.MODEL.MVIT.MLP_RATIO = 4.
_C.MODEL.MVIT.QKV_BIAS = True
_C.MODEL.MVIT.QK_SCALE = None
_C.MODEL.MVIT.PATCH_NORM = True

# RecurrentMViT parameters
_C.MODEL.RMVIT = CN()
_C.MODEL.RMVIT.IN_CHANS = 3
_C.MODEL.RMVIT.EMBED_DIM = 96
_C.MODEL.RMVIT.DEPTHS = [1, 1, 1, 1]
_C.MODEL.RMVIT.RECURRENCES = [1, 1, 10, 1]
_C.MODEL.RMVIT.DIM_PER_HEAD = 96
_C.MODEL.RMVIT.MLP_RATIO = 4.
_C.MODEL.RMVIT.QKV_BIAS = True
_C.MODEL.RMVIT.QK_SCALE = None
_C.MODEL.RMVIT.PATCH_NORM = True

# CMViT
_C.MODEL.CMVIT = CN()
_C.MODEL.CMVIT.IN_CHANS = 3
_C.MODEL.CMVIT.PATCH_SIZE = 4
_C.MODEL.CMVIT.EMBED_DIM = 96
_C.MODEL.CMVIT.EMBED_FACTORS = [1, 2, 4, 8]
_C.MODEL.CMVIT.DEPTHS = [1, 1, 10, 1]
_C.MODEL.CMVIT.NUM_CLUSTERS = [64, 32, 16, 8]
_C.MODEL.CMVIT.NUM_ANCHORS = [0, 0, 0, 0]
_C.MODEL.CMVIT.POOL_MODE = 'depth-conv'
_C.MODEL.CMVIT.POOL_STAGES = [0, 1, 2]
_C.MODEL.CMVIT.DIM_PER_HEAD = 96
_C.MODEL.CMVIT.MLP_RATIO = 4.
_C.MODEL.CMVIT.QKV_BIAS = True
_C.MODEL.CMVIT.QK_SCALE = None
_C.MODEL.CMVIT.PATCH_NORM = True
_C.MODEL.CMVIT.ASSIGN_TYPE = ['gumbel', 'hard', 'inv']
_C.MODEL.CMVIT.NUM_ASSIGN = [-1, -1, -1]
_C.MODEL.CMVIT.INTER_MODE = 'attn'
_C.MODEL.CMVIT.ASSIGN_SKIP = True
_C.MODEL.CMVIT.WITH_MLP_INTER = False
_C.MODEL.CMVIT.WITH_CLUSTER_ATTN = True
_C.MODEL.CMVIT.DECOUPLE_CLUSTER_ATTN = False
_C.MODEL.CMVIT.WITH_GAP = False
_C.MODEL.CMVIT.WITH_PEG = [0, 0, 0, 0]
_C.MODEL.CMVIT.POS_EMBED_TYPE = 'simple'
_C.MODEL.CMVIT.CLUSTER_MLP_TYPE = []
_C.MODEL.CMVIT.CLUSTER_TOKEN_WD = False
_C.MODEL.CMVIT.PATCH_EMBED_TYPE = 'simple'
_C.MODEL.CMVIT.I2C_MLP_RATIO = 4.
_C.MODEL.CMVIT.CLUSTER_HEAD_TYPE = 'none'
_C.MODEL.CMVIT.WITH_CLUSTER_PROJ = False
_C.MODEL.CMVIT.WITH_CLUSTER_NORM = False

# RecurrentCMViT
_C.MODEL.RCMVIT = CN()
_C.MODEL.RCMVIT.IN_CHANS = 3
_C.MODEL.RCMVIT.EMBED_DIM = 96
_C.MODEL.RCMVIT.DEPTHS = [1, 1, 1, 1]
_C.MODEL.RCMVIT.RECURRENCES = [1, 1, 10, 1]
_C.MODEL.RCMVIT.NUM_CLUSTERS = [64, 32, 16, 8]
_C.MODEL.RCMVIT.POOL_MODE = 'depth-conv'
_C.MODEL.RCMVIT.POOL_STAGES = [0, 1, 2]
_C.MODEL.RCMVIT.DIM_PER_HEAD = 96
_C.MODEL.RCMVIT.MLP_RATIO = 4.
_C.MODEL.RCMVIT.QKV_BIAS = True
_C.MODEL.RCMVIT.QK_SCALE = None
_C.MODEL.RCMVIT.PATCH_NORM = True
_C.MODEL.RCMVIT.WITH_GAP = False
_C.MODEL.RCMVIT.WITH_PEG = [0, 0, 0, 0]

# -----------------------------------------------------------------------------
# Training settings
# -----------------------------------------------------------------------------
_C.TRAIN = CN()
_C.TRAIN.START_EPOCH = 0
_C.TRAIN.EPOCHS = 300
_C.TRAIN.WARMUP_EPOCHS = 20
_C.TRAIN.WEIGHT_DECAY = 0.05
_C.TRAIN.BASE_LR = 5e-4
_C.TRAIN.WARMUP_LR = 5e-7
_C.TRAIN.MIN_LR = 5e-6
# Clip gradient norm
_C.TRAIN.CLIP_GRAD = 5.0
# Auto resume from latest checkpoint
_C.TRAIN.AUTO_RESUME = True
# Gradient accumulation steps
# could be overwritten by command line argument
_C.TRAIN.ACCUMULATION_STEPS = 0
# Whether to use gradient checkpointing to save memory
# could be overwritten by command line argument
_C.TRAIN.USE_CHECKPOINT = False

# LR scheduler
_C.TRAIN.LR_SCHEDULER = CN()
_C.TRAIN.LR_SCHEDULER.NAME = 'cosine'
# Epoch interval to decay LR, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_EPOCHS = 30
# LR decay rate, used in StepLRScheduler
_C.TRAIN.LR_SCHEDULER.DECAY_RATE = 0.1

# Optimizer
_C.TRAIN.OPTIMIZER = CN()
_C.TRAIN.OPTIMIZER.NAME = 'adamw'
# Optimizer Epsilon
_C.TRAIN.OPTIMIZER.EPS = 1e-8
# Optimizer Betas
_C.TRAIN.OPTIMIZER.BETAS = (0.9, 0.999)
# SGD momentum
_C.TRAIN.OPTIMIZER.MOMENTUM = 0.9

# -----------------------------------------------------------------------------
# Augmentation settings
# -----------------------------------------------------------------------------
_C.AUG = CN()
# Color jitter factor
_C.AUG.COLOR_JITTER = 0.4
# Use AutoAugment policy. "v0" or "original"
_C.AUG.AUTO_AUGMENT = 'rand-m9-mstd0.5-inc1'
# Random erase prob
_C.AUG.REPROB = 0.25
# Random erase mode
_C.AUG.REMODE = 'pixel'
# Random erase count
_C.AUG.RECOUNT = 1
# Mixup alpha, mixup enabled if > 0
_C.AUG.MIXUP = 0.8
# Cutmix alpha, cutmix enabled if > 0
_C.AUG.CUTMIX = 1.0
# Cutmix min/max ratio, overrides alpha and enables cutmix if set
_C.AUG.CUTMIX_MINMAX = None
# Probability of performing mixup or cutmix when either/both is enabled
_C.AUG.MIXUP_PROB = 1.0
# Probability of switching to cutmix when both mixup and cutmix enabled
_C.AUG.MIXUP_SWITCH_PROB = 0.5
# How to apply mixup/cutmix params. Per "batch", "pair", or "elem"
_C.AUG.MIXUP_MODE = 'batch'

# -----------------------------------------------------------------------------
# Testing settings
# -----------------------------------------------------------------------------
_C.TEST = CN()
# Whether to use center crop when testing
_C.TEST.CROP = True

# -----------------------------------------------------------------------------
# Misc
# -----------------------------------------------------------------------------
# Mixed precision opt level, if O0, no amp is used ('O0', 'O1', 'O2')
# overwritten by command line argument
_C.AMP_OPT_LEVEL = ''
# Path to output folder, overwritten by command line argument
_C.OUTPUT = ''
# Tag of experiment, overwritten by command line argument
_C.TAG = 'default'
# Frequency to save checkpoint
_C.SAVE_FREQ = 1
_C.MAX_KEPT_CKPT = -1
# Frequency to logging info
_C.PRINT_FREQ = 10
# Fixed random seed
_C.SEED = 0
# Perform evaluation only, overwritten by command line argument
_C.EVAL_MODE = False
# Test throughput only, overwritten by command line argument
_C.THROUGHPUT_MODE = False
# Log experiments with W&B
_C.WANDB = False
# local rank for DistributedDataParallel, given by command line argument
_C.LOCAL_RANK = 0


def _update_config_from_file(config, cfg_file):
    config.defrost()
    with open(cfg_file, 'r') as f:
        yaml_cfg = yaml.load(f, Loader=yaml.FullLoader)

    for cfg in yaml_cfg.setdefault('BASE', ['']):
        if cfg:
            _update_config_from_file(
                config, os.path.join(os.path.dirname(cfg_file), cfg)
            )
    print('=> merge config from {}'.format(cfg_file))
    config.merge_from_file(cfg_file)
    config.freeze()


def update_config(config, args):
    _update_config_from_file(config, args.cfg)

    config.defrost()
    if args.opts:
        config.merge_from_list(args.opts)

    # merge from specific arguments
    if args.batch_size:
        config.DATA.BATCH_SIZE = args.batch_size
    if args.data_path:
        config.DATA.DATA_PATH = args.data_path
    if args.zip:
        config.DATA.ZIP_MODE = True
    if args.web:
        config.DATA.WEB_MODE = True
    if args.cache_mode:
        config.DATA.CACHE_MODE = args.cache_mode
    if args.resume:
        config.MODEL.RESUME = args.resume
    if args.accumulation_steps:
        config.TRAIN.ACCUMULATION_STEPS = args.accumulation_steps
    if args.use_checkpoint:
        config.TRAIN.USE_CHECKPOINT = True
    if args.amp_opt_level:
        config.AMP_OPT_LEVEL = args.amp_opt_level
    if args.output:
        config.OUTPUT = args.output
    if args.tag:
        config.TAG = args.tag
    if args.eval:
        config.EVAL_MODE = True
    if args.throughput:
        config.THROUGHPUT_MODE = True
    if args.wandb:
        config.WANDB = True
    if args.keep:
        config.MAX_KEPT_CKPT = args.keep

    # set local rank for distributed training
    config.LOCAL_RANK = args.local_rank

    # set default name
    if not len(config.MODEL.NAME):
        config.MODEL.NAME = os.path.splitext(os.path.basename(args.cfg))[0]

    world_size = int(os.environ['WORLD_SIZE'])

    config.MODEL.NAME = config.MODEL.NAME+f'_bs{config.DATA.BATCH_SIZE}x{world_size}'

    # output folder
    config.OUTPUT = os.path.join(config.OUTPUT, config.MODEL.NAME, config.TAG)

    config.freeze()


def get_config(args):
    """Get a yacs CfgNode object with default values."""
    # Return a clone so that the defaults will not be altered
    # This is for the "local variable" use pattern
    config = _C.clone()
    update_config(config, args)

    return config
