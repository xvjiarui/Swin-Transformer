import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial
from collections import OrderedDict
import numpy as np
from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange, Reduce

from timm.models.layers import trunc_normal_, DropPath, to_2tuple
from .swin_transformer import PatchMerging
import torch.utils.checkpoint as checkpoint


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


class SelfAttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 out_dim=None, with_mlp=True):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
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

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        if self.with_mlp:
            x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
        return x


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
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
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
        # self.with_cp = True

    def forward(self, query, key, *, attn_weight=None, return_attn=False):
        x = query
        if return_attn:
            out, attn = self.attn(self.norm_q(query), key=self.norm_k(key),
                                  attn_weight=attn_weight,
                                  return_attn=return_attn)
        else:
            out = self.attn(self.norm_q(query), key=self.norm_k(key),
                            attn_weight=attn_weight,
                            return_attn=return_attn)
        x = x + self.drop_path(out)
        if self.with_mlp:
            x = self.reduction(x) + self.drop_path(self.mlp(self.norm2(x)))
        if return_attn:
            return x, attn
        else:
            return x


class EntropyDown(nn.Module):

    def __init__(self, ratio=4, depthwise=True):
        super(EntropyDown, self).__init__()
        self.ratio = ratio
        self.depthwise = depthwise

    def forward(self, x, coord, attn):
        """
        Args:
            x (torch.Tensor): tokens, [B, L, C]
            attn (torch.Tensor): attention map in log scale, [B, nH, L, S]
        """
        B, L, C = x.shape
        num_heads, S = attn.shape[1], attn.shape[3]
        num_output = L // self.ratio
        if self.depthwise:
            # [B, nH, L, C//nH]
            x = rearrange(x, 'b l (nh c)-> b nh l c',
                          nh=num_heads, b=B, l=L, c=C // num_heads)
            # [B, nH, L, C//nH]
            coord = rearrange(coord, 'b l (nh c)-> b nh l c',
                              nh=num_heads, b=B, l=L, c=C // num_heads)
            # [B, nH, L]
            neg_entropy = torch.sum(attn.exp() * attn, dim=-1)
            # [B, nH, oL]
            indices = neg_entropy.topk(dim=-1, sorted=False,
                                       k=num_output).indices
            # sort is faster than topk (https://github.com/pytorch/pytorch/issues/22812) or not?
            # indices = neg_entropy.sort(descending=True, dim=-1)[1][:, :, :num_output]
            # [B, nH, oL, C//nH]
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, C // num_heads)
            # [B, nH, oL, C//nH]
            x = x.gather(dim=2, index=indices)
            # [B, oL, C]
            x = rearrange(x, 'b nh ol c -> b ol (nh c)', nh=num_heads,
                          ol=num_output, c=C // num_heads, b=B)

            # [B, nH, oL, C//nH]
            coord = coord.gather(dim=2, index=indices)
            # [B, oL, C]
            coord = rearrange(coord, 'b nh ol c -> b ol (nh c)', nh=num_heads,
                              ol=num_output, c=C // num_heads, b=B)
        else:
            # [B, L, S]
            attn = attn.exp().mean(dim=1)
            # [B, L]
            neg_entropy = torch.sum(attn * attn.log(), dim=-1)
            # [B, oL]
            indices = neg_entropy.topk(dim=-1, sorted=False,
                                       k=num_output).indices
            # [B, oL, C]
            indices = indices.unsqueeze(-1).expand(-1, -1, C)
            # [B, oL, C]
            x = x.gather(dim=1, index=indices)
            # [B, oL, C]
            coord = coord.gather(dim=1, index=indices)

        return x, coord


class AttnStage(nn.Module):

    def __init__(self, input_resolution, dim, num_cluster, num_blocks,
                 num_heads, drop_path,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0.,
                 norm_layer=nn.LayerNorm, out_dim=None,
                 downsample=None,
                 use_checkpoint=False):
        super().__init__()
        self.cluster_token = nn.Parameter(torch.zeros(1, num_cluster, dim))
        self.num_blocks = num_blocks
        i2c_attn_blocks = []
        c2i_attn_blocks = []
        for blk_idx in range(num_blocks):
            i2c_attn_blocks.append(
                CrossAttnBlock(
                    dim=dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[blk_idx],
                    norm_layer=norm_layer,
                    with_mlp=False))
            c2i_attn_blocks.append(
                CrossAttnBlock(
                    dim=dim, num_heads=num_heads,
                    mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                    qk_scale=qk_scale, drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=drop_path[blk_idx],
                    norm_layer=norm_layer,
                    out_dim=out_dim if blk_idx == num_blocks - 1 else None))
        # self.coord_proj = Mlp(in_features=coord_dim, out_features=out_dim)
        self.i2c_attn_blocks = nn.ModuleList(i2c_attn_blocks)
        self.c2i_attn_blocks = nn.ModuleList(c2i_attn_blocks)
        trunc_normal_(self.cluster_token, std=.02)
        self.downsample = downsample
        self.input_resolution = input_resolution
        self.use_checkpoint = use_checkpoint

        # self.downsample = nn.Conv2d(in_channels=dim,
        #                             out_channels=out_dim,
        #                             groups=dim,
        #                             kernel_size=(2, 2), stride=(2, 2))

    def get_coord_attn(self, coord_embed, attn):
        """
        Args:
            coord_embed (torch.Tensor): [B, L, C]
            attn (torch.Tensor): [B, nH, S, L]
        """
        num_heads = attn.shape[1]
        # [B, nH, L, C//nH]
        coord_embed = rearrange(coord_embed, 'b l (nh c) -> b nh l c',
                                nh=num_heads)
        # [B, nH, S, nC//nH] <- [B, nH, S, L] @ [B, nH, L, nC//nH]
        cluster_coord_embed = attn @ coord_embed

        # [B, nH, L, S]
        attn_weight = F.normalize(coord_embed) @ F.normalize(
            cluster_coord_embed.transpose(-2, -1))/coord_embed.shape[-1]

        return F.log_softmax(attn_weight, dim=-1)

    def forward(self, x):
        # print(f'input: {x.shape}')

        output_token = x
        cluster_token = self.cluster_token.expand(x.size(0), -1, -1)

        B, L, C = output_token.shape
        # coord_embed = self.coord_proj(coord)
        for blk_idx in range(self.num_blocks):
            i2c_blk = self.i2c_attn_blocks[blk_idx]
            c2i_blk = self.c2i_attn_blocks[blk_idx]
            if self.use_checkpoint:
                cluster_token = checkpoint.checkpoint(i2c_blk, cluster_token, torch.cat((cluster_token, output_token), dim=1))
                output_token = checkpoint.checkpoint(c2i_blk, output_token, cluster_token)
            else:
                cluster_token= i2c_blk(
                    cluster_token,
                    torch.cat((cluster_token, output_token), dim=1))
                output_token = c2i_blk(output_token, cluster_token)

        # print(f'after cluster: {output_token.shape}')
        if self.downsample is not None:
            if isinstance(self.downsample, TokenAssign):
                output_token = self.downsample(output_token, cluster_token)
            else:
                output_token = self.downsample(output_token)
        # H, W = self.input_resolution
        # output_token = rearrange(self.downsample(rearrange(output_token, 'b (h w) c -> b c h w', h=H, w=W)), 'b c h w -> b (h w) c')
        # print(f'downsample: {output_token.shape}')

        return output_token
        # return output_token, coord_embed

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

    def __init__(self, dim, out_dim, num_heads, seq_len, num_cluster,  out_seq_len, norm_layer,
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
        merged_x = merged_cluster_tokens + self.assign(query=merged_cluster_tokens, key=x)
        merged_x = self.reduction(merged_x) + self.mlp_channels(self.norm4(merged_x))

        return merged_x


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, kernel_size=7, stride=4, padding=2,
                 in_chans=3, embed_dim=96):
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
        return x


class FixPosEmbed(nn.Module):

    def __init__(self, embed_dim, img_size=56, dropout=0.):
        super(FixPosEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim
        h, w = img_size

        div_term = torch.exp(torch.arange(0, embed_dim // 2, 2).float() * (
                -math.log(10000.0) / embed_dim * 2)).unsqueeze(1)

        h_pe = torch.zeros(embed_dim // 2, h)
        h_position = torch.arange(0, h, dtype=torch.float).unsqueeze(0)
        h_pe[0::2, :] = torch.sin(h_position * div_term)
        h_pe[1::2, :] = torch.cos(h_position * div_term)

        w_pe = torch.zeros(embed_dim // 2, w)
        w_position = torch.arange(0, w, dtype=torch.float).unsqueeze(0)
        w_pe[0::2, :] = torch.sin(w_position * div_term)
        w_pe[1::2, :] = torch.cos(w_position * div_term)

        pe = torch.cat([h_pe.unsqueeze(2).expand(-1, -1, w),
                        w_pe.unsqueeze(1).expand(-1, h, -1)])

        # [1, embed_dim, h, w]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image tensor, [B, C, H, W]
        """
        if not self.training:
            pe = F.interpolate(self.pe, size=x.shape[2:], mode='bicubic')
        else:
            pe = self.pe

        return x + self.dropout(pe)


class LearnablePosEmbed(nn.Module):

    def __init__(self, embed_dim, img_size=56, dropout=0.):
        super(LearnablePosEmbed, self).__init__()
        img_size = to_2tuple(img_size)
        self.dropout = nn.Dropout(p=dropout)
        self.embed_dim = embed_dim
        h, w = img_size

        self.pos_embed = nn.Parameter(
            torch.zeros(1, embed_dim, h, w))
        trunc_normal_(self.pos_embed, std=.02)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): image tensor, [B, C, H, W]
        """
        if not self.training:
            pos_embed = F.interpolate(self.pos_embed, size=x.shape[2:],
                                      mode='bicubic')
        else:
            pos_embed = self.pos_embed

        return x + self.dropout(pos_embed)


def build_spatial_downsample(downsample_type, H, W, dim, out_dim, norm_layer):
    assert downsample_type in ['max', 'avg', 'merge', 'conv']
    downsample = OrderedDict()
    if downsample_type == 'merge':
        return PatchMerging(input_resolution=(H, W), dim=dim,
                         norm_layer=norm_layer)
    downsample['to_spatial'] = Rearrange('b (h w) c -> b c h w', h=H, w=W)
    if downsample_type == 'max':
        downsample['downsample'] = nn.MaxPool2d(kernel_size=2, stride=2)
    elif downsample_type == 'avg':
        downsample['downsample'] = nn.AvgPool2d(kernel_size=2, stride=2)
    elif downsample_type == 'conv':
        downsample['downsample'] = nn.Conv2d(in_channels=dim,
                               out_channels=out_dim,
                               groups=dim,
                               kernel_size=(2, 2),
                               stride=(2, 2))
    else:
        raise NotImplementedError
    downsample['to_seq'] = Rearrange('b c h w -> b (h w) c', h=H//2, w=W//2)

    return nn.Sequential(downsample)


class GroupingVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """

    def __init__(self, img_size=224, in_chans=3,
                 num_classes=1000, base_dim=96,
                 dim_per_head=96,
                 downsample_type='token_assign',
                 pos_embed_type='learn',
                 stage_blocks=(1, 2, 11, 2),
                 cluster_tokens=(64, 32, 16, 8),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0.,
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 # with_coord=False,
                 use_checkpoint=False):
        super().__init__()
        self.num_classes = num_classes
        self.drop_rate = drop_rate
        self.stage_dims = tuple(
            base_dim * 2 ** i for i in range(len(stage_blocks)))
        self.depth = sum(stage_blocks)
        # self.with_coord = with_coord
        assert downsample_type in ['token_assign', 'max', 'avg', 'conv', 'merge']
        assert pos_embed_type in ['learn', 'fix']

        self.patch_embed = PatchEmbed(
            img_size=img_size, in_chans=in_chans,
            embed_dim=base_dim)
        if patch_norm:
            self.patch_norm = norm_layer(base_dim)
        else:
            self.patch_norm = nn.Identity()
        # self.patch_embed = PatchEmbed(
        #     img_size=img_size, in_chans=in_chans, embed_dim=base_dim,
        #     kernel_size=4, stride=4, padding=0
        # )
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution

        self.pos_embed = LearnablePosEmbed(base_dim, patches_resolution,
                                           dropout=drop_rate)
        # self.coord_embed = FixPosEmbed(base_dim, patches_resolution)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate,
                                                self.depth)]  # stochastic depth decay rule
        stages = []
        for stage_idx, num_blocks in enumerate(stage_blocks):
            H = patches_resolution[1] // 2 ** stage_idx
            W = patches_resolution[0] // 2 ** stage_idx
            stage_dim = self.stage_dims[stage_idx]
            num_heads = stage_dim // dim_per_head
            if stage_idx < len(stage_blocks) - 1:
                out_dim = self.stage_dims[stage_idx + 1]
                if downsample_type == 'token_assign':
                    downsample = TokenAssign(
                        dim=stage_dim,
                        out_dim=out_dim,
                        num_heads=num_heads,
                        seq_len=H * W,
                        num_cluster=cluster_tokens[stage_idx],
                        out_seq_len=H * W //4,
                        norm_layer=norm_layer,
                        hard=False,
                        inv_attn=True)
                else:
                    downsample = build_spatial_downsample(
                        downsample_type=downsample_type,
                        H=H, W=W, dim=stage_dim, out_dim=out_dim,
                        norm_layer=norm_layer)
                if stage_idx == 0:
                    downsample = build_spatial_downsample(
                        downsample_type='merge',
                        H=H, W=W, dim=stage_dim, out_dim=out_dim,
                        norm_layer=norm_layer)
            else:
                out_dim = None
                downsample = None
            stage_block = AttnStage(
                input_resolution=(H, W),
                dim=stage_dim, num_blocks=num_blocks,
                num_heads=num_heads,
                num_cluster=cluster_tokens[stage_idx],
                mlp_ratio=mlp_ratio, qkv_bias=qkv_bias,
                qk_scale=qk_scale,
                drop_path=dpr[sum(stage_blocks[:stage_idx]):sum(
                    stage_blocks[:stage_idx + 1])],
                drop_rate=drop_rate,
                attn_drop_rate=attn_drop_rate,
                norm_layer=norm_layer,
                out_dim=out_dim if downsample_type not in [
                    'merge', 'conv', 'token_assign'] else None,
                downsample=downsample,
                use_checkpoint=use_checkpoint)
            stages.append(stage_block)
        self.stages = nn.ModuleList(stages)

        self.norm = norm_layer(self.stage_dims[-1])

        # NOTE as per official impl, we could have a pre-logits representation dense layer + tanh here
        # self.repr = nn.Linear(embed_dim, representation_size)
        # self.repr_act = nn.Tanh()

        # Classifier head
        self.head = nn.Linear(self.stage_dims[-1],
                              num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


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
        skip_set = {'pos_embed', 'cls_token'}
        skip_subkey = ['cluster_token', 'pos_embed']
        for name, param in self.named_parameters():
            for subkey in skip_subkey:
                if subkey in name:
                    skip_set.add(name)
        print(f'no weight decay: {skip_set}')
        return skip_set

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'cluster_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim,
                              num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        x = self.patch_embed(x)

        B, C, H, W = x.shape
        x = self.pos_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c', h=H, w=W)
        x = self.patch_norm(x)

        # coord = self.coord_embed.pe.expand(B, -1, -1, -1)
        # coord = rearrange(coord, 'b c h w -> b (h w) c', h=H, w=W)

        for stage_idx, stage_block in enumerate(self.stages):
            # x, coord = stage_block(x, coord)
            x = stage_block(x)

        x = self.norm(x)
        return F.adaptive_avg_pool1d(x.transpose(1, 2), 1).squeeze(2)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
