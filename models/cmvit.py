# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import to_2tuple, trunc_normal_, drop_path
from einops import rearrange


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

def attention_pool(tensor, pool, hw_shape, has_cls_embed=True, norm=None):
    if pool is None:
        return tensor, hw_shape
    tensor_dim = tensor.ndim
    if tensor_dim == 4:
        pass
    elif tensor_dim == 3:
        tensor = tensor.unsqueeze(1)
    else:
        raise NotImplementedError(f"Unsupported input dimension {tensor.shape}")

    if has_cls_embed:
        cls_tok, tensor = tensor[:, :, :1, :], tensor[:, :, 1:, :]

    B, N, L, C = tensor.shape
    H, W = hw_shape
    tensor = rearrange(tensor, 'b n (h w) c -> (b n) c h w', h=H, w=W, b=B, n=N, c=C)
    tensor = tensor.contiguous()

    tensor = pool(tensor)

    hw_shape = [tensor.shape[2], tensor.shape[3]]
    L_pooled = tensor.shape[2] * tensor.shape[3]
    # [B, N, L_pooled, C]
    tensor = tensor.reshape(B, N, C, L_pooled).transpose(2, 3)
    if has_cls_embed:
        tensor = torch.cat((cls_tok, tensor), dim=2)
    if norm is not None:
        tensor = norm(tensor)
    # Assert tensor_dim in [3, 4]
    if tensor_dim == 4:
        pass
    else:  #  tensor_dim == 3:
        tensor = tensor.squeeze(1)
    return tensor, hw_shape


class SpatialPool(nn.Module):
    def __init__(self, mode, dim, out_dim, hw_shape, with_cls_token, norm_layer):
        super(SpatialPool, self).__init__()
        self.hw_shape = hw_shape
        self.with_cls_token = with_cls_token
        assert mode in ['depth-conv', 'conv', 'max', 'avg']
        if mode == 'conv' or mode == 'depth-conv':
            self.pool = nn.Conv2d(
                    dim,
                    dim,
                    (2, 2),
                    stride=(2, 2),
                    groups=dim if mode == 'depth-conv' else 1,
                    bias=False)
        elif mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        elif mode == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        else:
            raise NotImplementedError

        self.norm = norm_layer(dim)
        self.reduction = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
        """
        x, hw_shape = attention_pool(x, self.pool, self.hw_shape, self.with_cls_token, self.norm)
        x = self.reduction(x)

        return x, hw_shape


def hard_softmax(logits, dim):
    y_soft = logits.softmax(dim)
    # Straight through.
    index = y_soft.max(dim, keepdim=True)[1]
    y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    return ret

class AssignAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., hard=True, inv_attn=True):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.hard = hard
        self.inv_attn = inv_attn

    def get_attn(self, attn):
        if self.inv_attn:
            attn_dim = -2
        else:
            attn_dim = -1
        if self.hard:
            attn = hard_softmax(attn, dim=attn_dim)
        else:
            attn = F.softmax(attn, dim=attn_dim)

        return attn

    def forward(self, query, *, key=None, value=None, attn_weight=None):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c',
                      h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads,
                      b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c',
                      h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_weight is not None:
            assert attn_weight.shape == attn.shape
            # use log attn to match the scale of attn_weight
            attn = attn_weight + attn
        attn = self.get_attn(attn)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] <- [B, nh, N, S] @ [B, nh, S, C//nh]
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B,
                        n=N, c=C // self.num_heads)

        out = self.proj(out)
        out = self.proj_drop(out)
        return out

    def extra_repr(self) -> str:
        return f'hard: {self.hard}, \n' \
               f'inv_attn: {self.inv_attn}'

class TokenAssign(nn.Module):

    def __init__(self, dim, out_dim, num_heads, num_cluster,  out_seq_len, with_cls_token, norm_layer,
                 mlp_ratio=(0.5, 4.0), hard=True, inv_attn=True):
        super(TokenAssign, self).__init__()
        self.norm1 = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        self.mlp_tokens = Mlp(num_cluster, tokens_dim, out_seq_len)
        self.norm2 = norm_layer(dim)
        self.norm3 = norm_layer(dim)
        self.assign = AssignAttention(dim=dim, num_heads=num_heads,
                                      qkv_bias=True, hard=hard,
                                      inv_attn=inv_attn)
        self.norm4 = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(norm_layer(dim),
                                           nn.Linear(dim, out_dim,
                                                     bias=False))
        else:
            self.reduction = nn.Identity()
        self.hard = hard
        self.with_cls_token = with_cls_token

    def forward(self, x, cluster_tokens):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            cluster_tokens (torch.Tensor): cluster tokens, [B, L_1, C]
        """
        # [B, L_2, C] <- [B, L_1, C]
        merged_cluster_tokens = self.mlp_tokens(self.norm1(cluster_tokens).transpose(1, 2)).transpose(1, 2)
        # [B, L_2, C]
        x = self.norm2(x)
        merged_cluster_tokens = self.norm3(merged_cluster_tokens)
        # merged_x = merged_cluster_tokens + self.assign(query=merged_cluster_tokens, key=x)
        if self.with_cls_token:
            merged_x = self.assign(query=merged_cluster_tokens, key=x[:, 1:])
        else:
            merged_x = self.assign(query=merged_cluster_tokens, key=x)
        merged_x += merged_cluster_tokens

        if self.with_cls_token:
            merged_x = torch.cat((x[:, :1], merged_x), dim=1)

        merged_x = self.reduction(merged_x) + self.mlp_channels(self.norm4(merged_x))

        return merged_x, None


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, query, *, key=None, value=None, attn_weight=None,
                return_attn=False):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c',
                      h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads,
                      b=B, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        v = rearrange(self.v_proj(value), 'b n (h c)-> b h n c',
                      h=self.num_heads, b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        log_attn = F.log_softmax(attn, dim=-1)
        if attn_weight is not None:
            assert attn_weight.shape == log_attn.shape
            # use log attn to match the scale of attn_weight
            attn = attn_weight + log_attn
            log_attn = attn
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B,
                        n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        if return_attn:
            return out, log_attn
        else:
            return out


class CrossAttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 out_dim=None, with_mlp=True):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path_prob = drop_path
        self.with_mlp = with_mlp
        if self.with_mlp:
            self.norm2 = norm_layer(dim)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                           act_layer=act_layer, drop=drop, out_features=out_dim)
            if out_dim is not None and dim != out_dim:
                self.reduction = nn.Sequential(norm_layer(dim),
                                               nn.Linear(dim, out_dim,
                                                         bias=False))
            else:
                self.reduction = nn.Identity()

    def drop_path(self, x, drop_prob):
        if drop_prob is None:
            drop_prob = self.drop_path_prob
        return drop_path(x, drop_prob, self.training)

    def forward(self, query, key, *, drop_prob=None):
        x = query
        out = self.attn(self.norm_q(query), key=self.norm_k(key))
        x = x + self.drop_path(out, drop_prob=drop_prob)
        if self.with_mlp:
            x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)), drop_prob=drop_prob)
        return x


class BasicLayer(nn.Module):
    """ A basic Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, depth, num_heads, num_cluster,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, with_cls_token=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.cluster_token = nn.Parameter(torch.zeros(1, num_cluster, dim))

        # build blocks
        self.depth = depth
        i2c_attn_blocks = []
        c2i_attn_blocks = []
        for blk_idx in range(depth):
            i2c_attn_blocks.append(
                CrossAttnBlock(
                    dim=dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[blk_idx],
                    norm_layer=norm_layer,
                    with_mlp=False))
            c2i_attn_blocks.append(
                CrossAttnBlock(
                    dim=dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop,
                    attn_drop=attn_drop,
                    drop_path=drop_path[blk_idx],
                    norm_layer=norm_layer))
        # self.coord_proj = Mlp(in_features=coord_dim, out_features=out_dim)
        self.i2c_attn_blocks = nn.ModuleList(i2c_attn_blocks)
        self.c2i_attn_blocks = nn.ModuleList(c2i_attn_blocks)
        trunc_normal_(self.cluster_token, std=.02)
        self.downsample = downsample
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        self.downsample = downsample

    def forward(self, x):
        cluster_token = self.cluster_token.expand(x.size(0), -1, -1)

        B, L, C = x.shape
        # coord_embed = self.coord_proj(coord)
        for blk_idx in range(self.depth):
            i2c_blk = self.i2c_attn_blocks[blk_idx]
            c2i_blk = self.c2i_attn_blocks[blk_idx]
            if self.use_checkpoint:
                cluster_token = checkpoint.checkpoint(i2c_blk, cluster_token, torch.cat((cluster_token, x), dim=1))
                x = checkpoint.checkpoint(c2i_blk, x, cluster_token)
            else:
                cluster_token= i2c_blk(
                    cluster_token,
                    torch.cat((cluster_token, x), dim=1))
                x = c2i_blk(x, cluster_token)

        if self.downsample is not None:
            if isinstance(self.downsample, TokenAssign):
                x, hw_shape = self.downsample(x, cluster_token)
            else:
                x, hw_shape = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2,
                 in_chans=3, embed_dim=96, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        kernel_size = to_2tuple(kernel_size)
        stride = to_2tuple(stride)
        padding = to_2tuple(padding)
        self.img_size = img_size
        self.patches_resolution = (
            int((img_size[1] + 2 * padding[1] - kernel_size[1]) / stride[
                1] + 1),
            int((img_size[0] + 2 * padding[0] - kernel_size[0]) / stride[
                0] + 1))

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            # FIXME look at relaxing size constraints
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape


class CMViT(nn.Module):
    r""" Swin Transformer
        A PyTorch impl of : `Swin Transformer: Hierarchical Vision Transformer using Shifted Windows`  -
          https://arxiv.org/pdf/2103.14030

    Args:
        img_size (int | tuple(int)): Input image size. Default 224
        patch_size (int | tuple(int)): Patch size. Default: 4
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        embed_dim (int): Patch embedding dimension. Default: 96
        depths (tuple(int)): Depth of each Swin Transformer layer.
        num_heads (tuple(int)): Number of attention heads in different layers.
        window_size (int): Window size. Default: 7
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim. Default: 4
        qkv_bias (bool): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float): Override default qk scale of head_dim ** -0.5 if set. Default: None
        drop_rate (float): Dropout rate. Default: 0
        attn_drop_rate (float): Attention dropout rate. Default: 0
        drop_path_rate (float): Stochastic depth rate. Default: 0.1
        norm_layer (nn.Module): Normalization layer. Default: nn.LayerNorm.
        ape (bool): If True, add absolute position embedding to the patch embedding. Default: False
        patch_norm (bool): If True, add normalization after patch embedding. Default: True
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False
    """

    def __init__(self, img_size=224, in_chans=3, num_classes=1000,
                 embed_dim=96, depths=[1, 1, 10, 1], dim_per_head=96,
                 num_clusters=(64, 32, 16, 8),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
                 pool_mode='depth-conv',
                 pool_stages=[0, 1, 2],
                 assign_ratio=[4, 4, 4],
                 assign_type=('hard', 'inv'),
                 with_gap=False):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.dim_per_head = dim_per_head
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * 2 ** (self.num_layers - 1))
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.with_gap = with_gap
        self.pool_mode = pool_mode
        self.num_clusters = num_clusters
        self.pool_stages = pool_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            img_size=img_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        if self.with_gap:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            out_dim = dim * 2
            hw_shape = (patches_resolution[0] // (2 ** i_layer),
                        patches_resolution[1] // (2 ** i_layer))
            downsample = None
            if i_layer < self.num_layers -1 :
                if i_layer in self.pool_stages:
                    downsample = SpatialPool(mode=pool_mode,
                                             dim=dim,
                                             out_dim=out_dim,
                                             hw_shape=hw_shape,
                                             with_cls_token=self.with_cls_token,
                                             norm_layer=norm_layer)
                else:
                    downsample = TokenAssign(dim=dim,
                                             out_dim=out_dim,
                                             num_heads=dim // dim_per_head,
                                             num_cluster=num_clusters[i_layer],
                                             out_seq_len=np.prod(hw_shape) //
                                                         assign_ratio[i_layer],
                                             norm_layer=norm_layer,
                                             hard='hard' in assign_type,
                                             inv_attn='inv' in assign_type,
                                             with_cls_token=self.with_cls_token)
            layer = BasicLayer(dim=dim,
                               input_resolution=hw_shape,
                               depth=depths[i_layer],
                               num_heads=dim // dim_per_head,
                               num_cluster=num_clusters[i_layer],
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=downsample,
                               use_checkpoint=use_checkpoint,
                               with_cls_token=self.with_cls_token)
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        if self.with_cls_token:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    @property
    def with_cls_token(self):
        return not self.with_gap

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'cluster_token'}

    def forward_features(self, x):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1,
                                               -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x)

        # [B, L, C]
        x = self.norm(x)
        if self.with_gap:
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
        else:
            x = x[:, 0]
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class RecurrentBasicLayer(nn.Module):
    """ A basic Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_resolution (tuple[int]): Input resolution.
        depth (int): Number of blocks.
        num_heads (int): Number of attention heads.
        window_size (int): Local window size.
        mlp_ratio (float): Ratio of mlp hidden dim to embedding dim.
        qkv_bias (bool, optional): If True, add a learnable bias to query, key, value. Default: True
        qk_scale (float | None, optional): Override default qk scale of head_dim ** -0.5 if set.
        drop (float, optional): Dropout rate. Default: 0.0
        attn_drop (float, optional): Attention dropout rate. Default: 0.0
        drop_path (float | tuple[float], optional): Stochastic depth rate. Default: 0.0
        norm_layer (nn.Module, optional): Normalization layer. Default: nn.LayerNorm
        downsample (nn.Module | None, optional): Downsample layer at the end of the layer. Default: None
        use_checkpoint (bool): Whether to use checkpointing to save memory. Default: False.
    """

    def __init__(self, dim, input_resolution, recurrence, num_heads, num_cluster,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, with_cls_token=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.recurrence = recurrence
        self.use_checkpoint = use_checkpoint
        self.cluster_token = nn.Parameter(torch.zeros(1, num_cluster, dim))

        # build blocks
        self.i2c_attn_block = CrossAttnBlock(
            dim=dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer,
            with_mlp=False)
        self.c2i_attn_block = CrossAttnBlock(
            dim=dim, num_heads=num_heads,
            mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
            qk_scale=qk_scale, drop=drop,
            attn_drop=attn_drop,
            norm_layer=norm_layer)
        trunc_normal_(self.cluster_token, std=.02)
        self.drop_path = drop_path
        self.downsample = downsample
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # patch merging layer
        self.downsample = downsample

    def forward(self, x):
        cluster_token = self.cluster_token.expand(x.size(0), -1, -1)

        B, L, C = x.shape
        i2c_blk = self.i2c_attn_block
        c2i_blk = self.c2i_attn_block
        for blk_idx in range(self.recurrence):
            if self.use_checkpoint:
                cluster_token = checkpoint.checkpoint(
                    i2c_blk, cluster_token,
                    torch.cat((cluster_token, x), dim=1),
                    drop_prob=self.drop_path[blk_idx])
                x = checkpoint.checkpoint(
                    c2i_blk, x, cluster_token,
                    self.drop_path[blk_idx],
                    drop_prob=self.drop_path[blk_idx])
            else:
                cluster_token = i2c_blk(
                    cluster_token,
                    torch.cat((cluster_token, x), dim=1),
                    drop_prob=self.drop_path[blk_idx])
                x = c2i_blk(x, cluster_token, drop_prob=self.drop_path[blk_idx])

        if self.downsample is not None:
            x, hw_shape = self.downsample(x)

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, recurrence={self.recurrence}"

class RecurrentCMViT(CMViT):
    def __init__(self, *args, recurrences, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(recurrences, (list, tuple))
        self.recurrences = recurrences
        embed_dim = self.embed_dim
        dim_per_head = self.dim_per_head
        patches_resolution = self.patches_resolution
        mlp_ratio = self.mlp_ratio
        qkv_bias = self.qkv_bias
        qk_scale = self.qk_scale
        drop_rate = self.drop_rate
        attn_drop_rate = self.attn_drop_rate
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                sum(recurrences))]  # stochastic depth decay rule
        cluster_tokens = self.num_clusters
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            out_dim = dim * 2
            hw_shape = (patches_resolution[0] // (2 ** i_layer),
                        patches_resolution[1] // (2 ** i_layer))
            if i_layer < self.num_layers -1 :
                downsample = SpatialPool(mode=self.pool_mode,
                                         dim=dim,
                                         out_dim=out_dim,
                                         hw_shape=hw_shape,
                                         with_cls_token=self.with_cls_token,
                                         norm_layer=nn.LayerNorm)
            else:
                downsample = None
            layer = RecurrentBasicLayer(dim=dim,
                                        input_resolution=hw_shape,
                                        recurrence=recurrences[i_layer],
                                        num_heads=dim // dim_per_head,
                                        num_cluster=cluster_tokens[i_layer],
                                        mlp_ratio=mlp_ratio,
                                        qkv_bias=qkv_bias, qk_scale=qk_scale,
                                        drop=drop_rate,
                                        attn_drop=attn_drop_rate,
                                        drop_path=dpr[sum(recurrences[:i_layer]):sum(
                                            recurrences[:i_layer + 1])],
                                        norm_layer=nn.LayerNorm,
                                        downsample=downsample,
                                        use_checkpoint=False,
                                        with_cls_token=self.with_cls_token)
            self.layers.append(layer)
        assert len(recurrences) == len(self.layers)

    def extra_repr(self):
        return f'recurrences={self.recurrences}'
