# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .gvit import GroupingVisionTransformer


def build_model(config):
    model_type = config.MODEL.TYPE
    if model_type == 'swin':
        model = SwinTransformer(img_size=config.DATA.IMG_SIZE,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                num_classes=config.MODEL.NUM_CLASSES,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'gvit':
        model = GroupingVisionTransformer(img_size=config.DATA.IMG_SIZE,
                                          in_chans=config.MODEL.GVIT.IN_CHANS,
                                          num_classes=config.MODEL.NUM_CLASSES,
                                          base_dim=config.MODEL.GVIT.BASE_DIM,
                                          stage_blocks=config.MODEL.GVIT.STAGE_BLOCKS,
                                          dim_per_head=config.MODEL.GVIT.DIM_PER_HEAD,
                                          mlp_ratio=config.MODEL.GVIT.MLP_RATIO,
                                          qkv_bias=config.MODEL.GVIT.QKV_BIAS,
                                          qk_scale=config.MODEL.GVIT.QK_SCALE,
                                          drop_rate=config.MODEL.DROP_RATE,
                                          drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                          patch_norm=config.MODEL.GVIT.PATCH_NORM,
                                          cluster_tokens=config.MODEL.GVIT.CLUSTER_TOKENS,
                                          downsample_type=config.MODEL.GVIT.DOWNSAMPLE_TYPE,
                                          use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    else:
        raise NotImplementedError(f"Unkown model: {model_type}")

    return model
