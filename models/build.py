# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

from .swin_transformer import SwinTransformer
from .gvit import GroupingVisionTransformer
from .vit import VisionTransformer, RecurrentVisionTransformer
from .mvit import MViT, RecurrentMViT
from .cmvit import CMViT, RecurrentCMViT


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
    elif model_type == 'vit':
        model = VisionTransformer(img_size=config.DATA.IMG_SIZE,
                                  patch_size=config.MODEL.VIT.PATCH_SIZE,
                                  in_chans=config.MODEL.VIT.IN_CHANS,
                                  num_classes=config.MODEL.NUM_CLASSES,
                                  embed_dim=config.MODEL.VIT.EMBED_DIM,
                                  depth=config.MODEL.VIT.DEPTH,
                                  num_heads=config.MODEL.VIT.NUM_HEADS,
                                  mlp_ratio=config.MODEL.VIT.MLP_RATIO,
                                  qkv_bias=config.MODEL.VIT.QKV_BIAS,
                                  qk_scale=config.MODEL.VIT.QK_SCALE,
                                  drop_rate=config.MODEL.DROP_RATE,
                                  drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                  patch_norm=config.MODEL.VIT.PATCH_NORM,
                                  use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                  with_gap=config.MODEL.VIT.WITH_GAP,
                                  with_peg=config.MODEL.VIT.WITH_PEG)
    elif model_type == 'mvit':
        model = MViT(img_size=config.DATA.IMG_SIZE,
                     in_chans=config.MODEL.MVIT.IN_CHANS,
                     num_classes=config.MODEL.NUM_CLASSES,
                     embed_dim=config.MODEL.MVIT.EMBED_DIM,
                     depths=config.MODEL.MVIT.DEPTHS,
                     dim_per_head=config.MODEL.MVIT.DIM_PER_HEAD,
                     mlp_ratio=config.MODEL.MVIT.MLP_RATIO,
                     qkv_bias=config.MODEL.MVIT.QKV_BIAS,
                     qk_scale=config.MODEL.MVIT.QK_SCALE,
                     drop_rate=config.MODEL.DROP_RATE,
                     drop_path_rate=config.MODEL.DROP_PATH_RATE,
                     patch_norm=config.MODEL.MVIT.PATCH_NORM,
                     use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'cmvit':
        model = CMViT(img_size=config.DATA.IMG_SIZE,
                      in_chans=config.MODEL.CMVIT.IN_CHANS,
                      num_classes=config.MODEL.NUM_CLASSES,
                      embed_dim=config.MODEL.CMVIT.EMBED_DIM,
                      depths=config.MODEL.CMVIT.DEPTHS,
                      num_clusters=config.MODEL.CMVIT.NUM_CLUSTERS,
                      num_anchors=config.MODEL.CMVIT.NUM_ANCHORS,
                      pool_mode=config.MODEL.CMVIT.POOL_MODE,
                      pool_stages=config.MODEL.CMVIT.POOL_STAGES,
                      dim_per_head=config.MODEL.CMVIT.DIM_PER_HEAD,
                      mlp_ratio=config.MODEL.CMVIT.MLP_RATIO,
                      qkv_bias=config.MODEL.CMVIT.QKV_BIAS,
                      qk_scale=config.MODEL.CMVIT.QK_SCALE,
                      drop_rate=config.MODEL.DROP_RATE,
                      drop_path_rate=config.MODEL.DROP_PATH_RATE,
                      patch_norm=config.MODEL.CMVIT.PATCH_NORM,
                      use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                      assign_type=config.MODEL.CMVIT.ASSIGN_TYPE,
                      num_assign=config.MODEL.CMVIT.NUM_ASSIGN,
                      inter_mode=config.MODEL.CMVIT.INTER_MODE,
                      with_mlp_inter=config.MODEL.CMVIT.WITH_MLP_INTER,
                      with_cluster_attn=config.MODEL.CMVIT.WITH_CLUSTER_ATTN,
                      with_gap=config.MODEL.CMVIT.WITH_GAP,
                      with_peg=config.MODEL.CMVIT.WITH_PEG)
    elif model_type == 'rvit':
        model = RecurrentVisionTransformer(img_size=config.DATA.IMG_SIZE,
                                           patch_size=config.MODEL.RVIT.PATCH_SIZE,
                                           in_chans=config.MODEL.RVIT.IN_CHANS,
                                           num_classes=config.MODEL.NUM_CLASSES,
                                           embed_dim=config.MODEL.RVIT.EMBED_DIM,
                                           depth=config.MODEL.RVIT.DEPTH,
                                           recurrences=config.MODEL.RVIT.RECURRENCES,
                                           num_heads=config.MODEL.RVIT.NUM_HEADS,
                                           mlp_ratio=config.MODEL.RVIT.MLP_RATIO,
                                           qkv_bias=config.MODEL.RVIT.QKV_BIAS,
                                           qk_scale=config.MODEL.RVIT.QK_SCALE,
                                           drop_rate=config.MODEL.DROP_RATE,
                                           drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                           patch_norm=config.MODEL.RVIT.PATCH_NORM,
                                           use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                           with_gap=config.MODEL.RVIT.WITH_GAP,
                                           with_peg=config.MODEL.RVIT.WITH_PEG)
    elif model_type == 'rmvit':
        model = RecurrentMViT(img_size=config.DATA.IMG_SIZE,
                              in_chans=config.MODEL.RMVIT.IN_CHANS,
                              num_classes=config.MODEL.NUM_CLASSES,
                              embed_dim=config.MODEL.RMVIT.EMBED_DIM,
                              depths=config.MODEL.RMVIT.DEPTHS,
                              recurrences=config.MODEL.RMVIT.RECURRENCES,
                              dim_per_head=config.MODEL.RMVIT.DIM_PER_HEAD,
                              mlp_ratio=config.MODEL.RMVIT.MLP_RATIO,
                              qkv_bias=config.MODEL.RMVIT.QKV_BIAS,
                              qk_scale=config.MODEL.RMVIT.QK_SCALE,
                              drop_rate=config.MODEL.DROP_RATE,
                              drop_path_rate=config.MODEL.DROP_PATH_RATE,
                              patch_norm=config.MODEL.RMVIT.PATCH_NORM,
                              use_checkpoint=config.TRAIN.USE_CHECKPOINT)
    elif model_type == 'rcmvit':
        model = RecurrentCMViT(img_size=config.DATA.IMG_SIZE,
                               in_chans=config.MODEL.RCMVIT.IN_CHANS,
                               num_classes=config.MODEL.NUM_CLASSES,
                               embed_dim=config.MODEL.RCMVIT.EMBED_DIM,
                               depths=config.MODEL.RCMVIT.DEPTHS,
                               recurrences=config.MODEL.RCMVIT.RECURRENCES,
                               num_clusters=config.MODEL.RCMVIT.NUM_CLUSTERS,
                               pool_mode=config.MODEL.RCMVIT.POOL_MODE,
                               pool_stages=config.MODEL.RCMVIT.POOL_STAGES,
                               dim_per_head=config.MODEL.RCMVIT.DIM_PER_HEAD,
                               mlp_ratio=config.MODEL.RCMVIT.MLP_RATIO,
                               qkv_bias=config.MODEL.RCMVIT.QKV_BIAS,
                               qk_scale=config.MODEL.RCMVIT.QK_SCALE,
                               drop_rate=config.MODEL.DROP_RATE,
                               drop_path_rate=config.MODEL.DROP_PATH_RATE,
                               patch_norm=config.MODEL.RCMVIT.PATCH_NORM,
                               use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                               with_gap=config.MODEL.RCMVIT.WITH_GAP,
                               with_peg=config.MODEL.RCMVIT.WITH_PEG)
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
