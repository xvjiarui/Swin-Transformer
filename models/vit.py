import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
import numpy as np
from einops.layers.torch import Rearrange

from timm.models.layers import trunc_normal_, drop_path, to_2tuple

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x, drop_prob=None):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
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


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        # [3, B, nh, N, C//nh]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, nh, N, C//nh]
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)

        # [B, nh, N, N]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path_prob = drop_path
        # self.drop_path = DropPath(
        #     drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)

    def drop_path(self, x, drop_prob):
        if drop_prob is None:
            drop_prob = self.drop_path_prob
        return drop_path(x, drop_prob, self.training)


    def forward(self, x, drop_prob=None):
        x = x + self.drop_path(self.attn(self.norm1(x)), drop_prob=drop_prob)
        x = x + self.drop_path(self.mlp(self.norm2(x)), drop_prob=drop_prob)
        return x

class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        patches_resolution = [img_size[0] // patch_size[0], img_size[1] // patch_size[1]]
        self.img_size = img_size
        self.patch_size = patch_size
        self.patches_resolution = patches_resolution
        self.num_patches = patches_resolution[0] * patches_resolution[1]

        self.in_chans = in_chans
        self.embed_dim = embed_dim

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        if self.training:
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        if self.norm is not None:
            x = self.norm(x)
        return x, hw_shape

class DeepPatchEmbed(nn.Module):
    """ Deep stem from https://arxiv.org/pdf/2106.14881.pdf
    """

    def __init__(self, img_size=224, total_stride=16,
                 in_chans=3, embed_dim=768, norm_type='BN', depthwise=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.patches_resolution = (img_size[1]//total_stride, img_size[0]//total_stride)
        assert norm_type in ['LN', 'BN']

        num_convs = int(np.log2(total_stride))
        stem = []
        prev_out_channels = in_chans
        out_channels = embed_dim // 2**(num_convs -1)
        for i in range(num_convs):
            stem.append(nn.Conv2d(
                in_channels=prev_out_channels,
                out_channels=out_channels,
                kernel_size=(3, 3),
                stride=(2, 2),
                padding=(1, 1),
                groups=prev_out_channels if depthwise and i > 0 else 1,
                bias=False))
            if norm_type == 'BN':
                stem.append(nn.BatchNorm2d(out_channels))
            else:
                stem.extend([Rearrange('b c h w -> b h w c'),
                             nn.LayerNorm(out_channels),
                             Rearrange('b h w c -> b c h w')])
            stem.append(nn.ReLU(inplace=True))
            prev_out_channels = out_channels
            if i < num_convs -1:
                out_channels *= 2
        stem.append(nn.Conv2d(
            in_channels=out_channels,
            out_channels=embed_dim,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True))
        self.stem = nn.Sequential(*stem)

    @property
    def num_patches(self):
        return self.patches_resolution[1] * self.patches_resolution[0]

    def forward(self, x):
        B, C, H, W = x.shape
        if self.training:
            # FIXME look at relaxing size constraints
            assert H == self.img_size[0] and W == self.img_size[1], \
                f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.stem(x)
        hw_shape = x.shape[2:]
        x = x.flatten(2).transpose(1, 2)
        return x, hw_shape


class PEG(nn.Module):
    def __init__(self, dim, out_dim, k=(3, 3), with_gap=True):
        super(PEG, self).__init__()
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.peg_conv = nn.Conv2d(dim, out_dim, kernel_size=k, stride=(1, 1),
                                  padding=(k[0] // 2, k[1] // 2), groups=out_dim)
        self.with_gap = with_gap

    def forward(self, x, hw_shape):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            hw_shape (tuple[int]): (height, width)
        """
        x = self.norm(x)
        batch, length, channels = x.shape
        if self.with_gap:
            img_view = x
        else:
            img_view = x[:, 1:]
        img_view = img_view.transpose(1, 2).reshape(batch, channels, *hw_shape).contiguous()
        img_view = img_view + self.peg_conv(img_view)
        x_pos = img_view.flatten(2).transpose(1, 2)
        if not self.with_gap:
            x_pos = torch.cat((x[:, :1], x_pos), dim=1)
        return x_pos

class VisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 patch_norm=True,
                 with_gap=False,
                 with_peg=False,
                 patch_embed_type='simple'):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.drop_rate = drop_rate
        self.with_gap = with_gap
        self.drop_path_rate = drop_path_rate
        self.use_checkpoint = use_checkpoint
        self.with_peg = with_peg
        assert patch_embed_type in ['simple', 'stem-BN', 'stem-LN', 'stem-depth-BN', 'stem-depth-LN']

        if patch_embed_type == 'simple':
            self.patch_embed = PatchEmbed(
                img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                embed_dim=embed_dim, norm_layer=norm_layer if patch_norm else None)
        else:
            self.patch_embed = DeepPatchEmbed(
                img_size=img_size, total_stride=patch_size, embed_dim=embed_dim,
                depthwise='depth' in patch_embed_type,
                norm_type=patch_embed_type[-2:])

        num_patches = self.patch_embed.num_patches

        if self.with_gap:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches, embed_dim))
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        self.dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=self.dpr[i],
                norm_layer=norm_layer)
            for i in range(depth)])
        if self.with_peg:
            self.pegs = nn.ModuleList([
                PEG(dim=embed_dim, out_dim=embed_dim, with_gap=self.with_gap)
                for i in range(depth)])

        self.norm = norm_layer(embed_dim)

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(embed_dim,
                              num_classes) if num_classes > 0 else nn.Identity()

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
        return {'peg_conv'}

    def forward_blks(self, x, hw_shape):
        for i, blk in enumerate(self.blocks):
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
            if self.with_peg:
                x = self.pegs[i](x, hw_shape)

        return x

    def forward_features(self, x):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1,
                                               -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        x = self.forward_blks(x, hw_shape)

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


class RecurrentVisionTransformer(VisionTransformer):
    def __init__(self, *args, recurrences, **kwargs):
        super().__init__(*args, **kwargs)
        assert isinstance(recurrences, (list, tuple))
        assert len(recurrences) == len(self.blocks)
        self.recurrences = recurrences
        self.dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate,
                                                     sum(recurrences))]  # stochastic depth decay rule
        if self.with_peg:
            self.pegs = nn.ModuleList([
                PEG(dim=self.embed_dim, out_dim=self.embed_dim, with_gap=self.with_gap)
                for i in range(sum(recurrences))])

    def forward_blks(self, x, hw_shape):
        for i, blk in enumerate(self.blocks):
            for r in range(self.recurrences[i]):
                x = blk(x, drop_prob=self.dpr[sum(self.recurrences[:i])+r])
                if self.with_peg:
                    x = self.pegs[sum(self.recurrences[:i])+r](x, hw_shape)

        return x

    def extra_repr(self):
        return f'recurrences={self.recurrences}'
