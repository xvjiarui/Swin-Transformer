# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import numpy as np
import torch
import torch.nn as nn
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


class MultiScaleAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        drop_rate=0.0,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        norm_layer=nn.LayerNorm,
        has_cls_embed=True,
        # Options include `conv`, `avg`, and `max`.
        mode="conv",
    ):
        super().__init__()
        self.drop_rate = drop_rate
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.has_cls_embed = has_cls_embed
        padding_q = [int(q // 2) for q in kernel_q]
        padding_kv = [int(kv // 2) for kv in kernel_kv]

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        if drop_rate > 0.0:
            self.proj_drop = nn.Dropout(drop_rate)

        # Skip pooling with kernel and stride size of (1, 1, 1).
        if np.prod(kernel_q) == 1 and np.prod(stride_q) == 1:
            kernel_q = ()
        if np.prod(kernel_kv) == 1 and np.prod(stride_kv) == 1:
            kernel_kv = ()

        if mode == "avg":
            self.pool_q = (
                nn.AvgPool2d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.AvgPool2d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.AvgPool2d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "max":
            self.pool_q = (
                nn.MaxPool2d(kernel_q, stride_q, padding_q, ceil_mode=False)
                if len(kernel_q) > 0
                else None
            )
            self.pool_k = (
                nn.MaxPool2d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
            self.pool_v = (
                nn.MaxPool2d(kernel_kv, stride_kv, padding_kv, ceil_mode=False)
                if len(kernel_kv) > 0
                else None
            )
        elif mode == "conv":
            self.pool_q = (
                nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_q,
                    stride=stride_q,
                    padding=padding_q,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_q) > 0
                else None
            )
            self.norm_q = norm_layer(head_dim) if len(kernel_q) > 0 else None
            self.pool_k = (
                nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_k = norm_layer(head_dim) if len(kernel_kv) > 0 else None
            self.pool_v = (
                nn.Conv2d(
                    head_dim,
                    head_dim,
                    kernel_kv,
                    stride=stride_kv,
                    padding=padding_kv,
                    groups=head_dim,
                    bias=False,
                )
                if len(kernel_kv) > 0
                else None
            )
            self.norm_v = norm_layer(head_dim) if len(kernel_kv) > 0 else None
        else:
            raise NotImplementedError(f"Unsupported model {mode}")

    def forward(self, x, hw_shape):
        B, N, C = x.shape

        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        q, out_shape = attention_pool(
            q,
            self.pool_q,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_q if hasattr(self, "norm_q") else None,
        )
        k, _ = attention_pool(
            k,
            self.pool_k,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_k if hasattr(self, "norm_k") else None,
        )
        v, _ = attention_pool(
            v,
            self.pool_v,
            hw_shape,
            has_cls_embed=self.has_cls_embed,
            norm=self.norm_v if hasattr(self, "norm_v") else None,
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        N = q.shape[2]
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        if self.drop_rate > 0.0:
            x = self.proj_drop(x)
        return x, out_shape

class MultiScaleBlock(nn.Module):
    def __init__(
        self,
        dim,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        up_rate=None,
        kernel_q=(1, 1),
        kernel_kv=(1, 1),
        stride_q=(1, 1),
        stride_kv=(1, 1),
        mode="conv",
        has_cls_embed=True,
    ):
        super().__init__()
        self.dim = dim
        self.dim_out = dim_out
        self.norm1 = norm_layer(dim)
        kernel_skip = [s + 1 if s > 1 else s for s in stride_q]
        stride_skip = stride_q
        padding_skip = [int(skip // 2) for skip in kernel_skip]
        self.attn = MultiScaleAttention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            drop_rate=attn_drop_rate,
            kernel_q=kernel_q,
            kernel_kv=kernel_kv,
            stride_q=stride_q,
            stride_kv=stride_kv,
            norm_layer=nn.LayerNorm,
            has_cls_embed=has_cls_embed,
            mode=mode,
        )

        self.drop_path_prob = drop_path

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.has_cls_embed = has_cls_embed
        # TODO: check the use case for up_rate, and merge the following lines
        if up_rate is not None and up_rate > 1:
            mlp_dim_out = dim * up_rate
        else:
            mlp_dim_out = dim_out
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=mlp_dim_out,
            act_layer=act_layer,
            drop=drop_rate,
        )
        if dim != dim_out:
            self.proj = nn.Linear(dim, dim_out)

        self.pool_skip = (
            nn.MaxPool2d(
                kernel_skip, stride_skip, padding_skip, ceil_mode=False
            )
            if len(kernel_skip) > 0
            else None
        )

    def drop_path(self, x, drop_prob):
        if drop_prob is None:
            drop_prob = self.drop_path_prob
        return drop_path(x, drop_prob, self.training)

    def forward(self, x, hw_shape, drop_prob=None):
        x_block, hw_shape_new = self.attn(self.norm1(x), hw_shape)
        x_res, _ = attention_pool(
            x, self.pool_skip, hw_shape, has_cls_embed=self.has_cls_embed
        )
        x = x_res + self.drop_path(x_block, drop_prob=drop_prob)
        x_norm = self.norm2(x)
        x_mlp = self.mlp(x_norm)
        if self.dim != self.dim_out:
            x = self.proj(x_norm)
        x = x + self.drop_path(x_mlp, drop_prob=drop_prob)
        return x, hw_shape_new

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

    def __init__(self, dim, input_resolution, depth, num_heads, attn_stride, attn_kernel=(3, 3),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, with_cls_token=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            MultiScaleBlock(dim=dim,
                            dim_out=dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop_rate=drop,
                            attn_drop_rate=attn_drop,
                            drop_path=drop_path[i],
                            norm_layer=norm_layer,
                            stride_kv=attn_stride,
                            kernel_kv=attn_kernel,
                            stride_q=[],
                            kernel_q=[],
                            has_cls_embed=with_cls_token)
            for i in range(depth)])

        # patch merging layer
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for blk in self.blocks:
            if self.use_checkpoint:
                x, hw_shape = checkpoint.checkpoint(blk, x, hw_shape)
            else:
                x, hw_shape = blk(x, hw_shape)
        if self.downsample is not None:
            x, hw_shape = self.downsample(x, hw_shape)
        return x, hw_shape

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


class MViT(nn.Module):
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
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 use_checkpoint=False,
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
            if i_layer < self.num_layers -1 :
                downsample = MultiScaleBlock(dim=dim,
                                             dim_out=out_dim,
                                             num_heads=out_dim//dim_per_head,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop_rate=drop_rate,
                                             attn_drop_rate=attn_drop_rate,
                                             norm_layer=norm_layer,
                                             kernel_q=to_2tuple(3),
                                             stride_q=to_2tuple(2),
                                             kernel_kv=to_2tuple(3),
                                             stride_kv=to_2tuple(2 ** (self.num_layers-2 - i_layer)),
                                             has_cls_embed=self.with_cls_token)
            else:
                downsample = None
            layer = BasicLayer(dim=dim,
                               input_resolution=(
                               patches_resolution[0] // (2 ** i_layer),
                               patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=dim // dim_per_head,
                               attn_stride=to_2tuple(
                                   2 ** (self.num_layers - 1 - i_layer)),
                               attn_kernel=to_2tuple(
                                   3) if i_layer < self.num_layers - 1 else to_2tuple(
                                   1),
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(depths[:i_layer]):sum(
                                   depths[:i_layer + 1])],
                               norm_layer=norm_layer,
                               downsample=downsample if (
                                           i_layer < self.num_layers - 1) else None,
                               use_checkpoint=use_checkpoint)
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
            x, hw_shape = layer(x, hw_shape)

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

    def __init__(self, dim, input_resolution, recurrence, num_heads, attn_stride, attn_kernel=(3, 3),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., norm_layer=nn.LayerNorm, downsample=None, use_checkpoint=False, with_cls_token=True):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.recurrence = recurrence
        self.use_checkpoint = use_checkpoint
        self.block = MultiScaleBlock(dim=dim,
                                     dim_out=dim,
                                     num_heads=num_heads,
                                     mlp_ratio=mlp_ratio,
                                     qkv_bias=qkv_bias,
                                     qk_scale=qk_scale,
                                     drop_rate=drop,
                                     attn_drop_rate=attn_drop,
                                     norm_layer=norm_layer,
                                     stride_kv=attn_stride,
                                     kernel_kv=attn_kernel,
                                     has_cls_embed=with_cls_token)
        self.drop_path = drop_path

        # patch merging layer
        self.downsample = downsample

    def forward(self, x, hw_shape):
        for i in range(self.recurrence):
            blk = self.block
            if self.use_checkpoint:
                x, hw_shape = checkpoint.checkpoint(blk, x, hw_shape, self.drop_path[i])
            else:
                x, hw_shape = blk(x, hw_shape, self.drop_path[i])
        if self.downsample is not None:
            x, hw_shape = self.downsample(x, hw_shape)
        return x, hw_shape

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, recurrence={self.recurrence}"

class RecurrentMViT(MViT):
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
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * 2 ** i_layer)
            out_dim = dim * 2
            if i_layer < self.num_layers -1 :
                downsample = MultiScaleBlock(dim=dim,
                                             dim_out=out_dim,
                                             num_heads=out_dim//dim_per_head,
                                             mlp_ratio=mlp_ratio,
                                             qkv_bias=qkv_bias,
                                             qk_scale=qk_scale,
                                             drop_rate=drop_rate,
                                             attn_drop_rate=attn_drop_rate,
                                             norm_layer=nn.LayerNorm,
                                             kernel_q=to_2tuple(3),
                                             stride_q=to_2tuple(2),
                                             kernel_kv=to_2tuple(3),
                                             stride_kv=to_2tuple(2 ** (self.num_layers-2 - i_layer)),
                                             has_cls_embed=self.with_cls_token)
            else:
                downsample = None
            layer = RecurrentBasicLayer(dim=dim,
                               input_resolution=(
                                   patches_resolution[0] // (2 ** i_layer),
                                   patches_resolution[1] // (2 ** i_layer)),
                               recurrence=recurrences[i_layer],
                               num_heads=dim // dim_per_head,
                               attn_stride=to_2tuple(
                                   2 ** (self.num_layers - 1 - i_layer)),
                               attn_kernel=to_2tuple(
                                   3) if i_layer < self.num_layers - 1 else to_2tuple(
                                   1),
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               drop=drop_rate, attn_drop=attn_drop_rate,
                               drop_path=dpr[sum(recurrences[:i_layer]):sum(
                                   recurrences[:i_layer + 1])],
                               norm_layer=nn.LayerNorm,
                               downsample=downsample if (
                                       i_layer < self.num_layers - 1) else None,
                               use_checkpoint=False)
            self.layers.append(layer)
        assert len(recurrences) == len(self.layers)

    def extra_repr(self):
        return f'recurrences={self.recurrences}'
