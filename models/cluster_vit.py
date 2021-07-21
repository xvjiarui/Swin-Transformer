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
from einops.layers.torch import Rearrange
from .vit import PEG


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

    # update C
    C = tensor.shape[1]

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
        assert mode in ['depth-conv', 'conv', 'max', 'avg', 'unfold']
        norm_dim = dim
        if mode == 'conv' or mode == 'depth-conv':
            self.pool = nn.Conv2d(
                dim,
                dim,
                (3, 3),
                stride=(2, 2),
                padding=(1, 1),
                groups=dim if mode == 'depth-conv' else 1,
                bias=False)
        elif mode == 'max':
            self.pool = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        elif mode == 'avg':
            self.pool = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        elif mode == 'unfold':
            self.pool = nn.Sequential(
                nn.Unfold(kernel_size=(2, 2), stride=(2, 2)),
                Rearrange('b c (h w) -> b c h w', h=hw_shape[0]//2, w=hw_shape[1]//2))
            norm_dim = 4 * dim
        else:
            raise NotImplementedError

        self.norm = norm_layer(norm_dim)
        self.reduction = nn.Linear(norm_dim, out_dim, bias=False)

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

def hard_softmax_sample(attn, dim):
    """
    Args:
        attn (torch.Tensor): the attention map, shape [B, L, S]
        dim (int): dimension to perform softmax
    """
    y_soft = F.log_softmax(attn, dim=dim)
    if dim != -1 or dim != attn.ndim -1:
        swap_dim = -1
        y_soft = y_soft.transpose(swap_dim, dim)
    else:
        swap_dim = dim
    index = torch.distributions.categorical.Categorical(logits=y_soft).sample().unsqueeze(-1)
    y_hard = torch.zeros_like(y_soft).scatter_(-1, index, 1.0)
    ret = y_hard - y_soft.detach() + y_soft

    if swap_dim != dim:
        ret = ret.transpose(swap_dim, dim)

    return ret

def gumbel_softmax(logits: torch.Tensor, tau: float = 1, hard: bool = False, dim: int = -1) -> torch.Tensor:
    # _gumbels = (
    #     -torch.empty_like(logits, memory_format=torch.legacy_contiguous_format).exponential_().log()
    # )  # ~Gumbel(0,1)
    # more stable https://github.com/pytorch/pytorch/issues/41663
    gumbel_dist = torch.distributions.gumbel.Gumbel(
        torch.tensor(0., device=logits.device, dtype=logits.dtype),
        torch.tensor(1., device=logits.device, dtype=logits.dtype))
    gumbels = gumbel_dist.sample(logits.shape)

    gumbels = (logits + gumbels) / tau  # ~Gumbel(logits,tau)
    y_soft = gumbels.softmax(dim)

    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret

class AssignAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., hard=True, inv_attn=True,
                 gumbel=False, categorical=False, gumbel_tau=1.):
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
        self.gumbel = gumbel
        self.categorical = categorical
        if categorical:
            assert hard
        self.gumbel_tau = gumbel_tau

    def get_attn(self, attn):
        if self.inv_attn:
            attn_dim = -2
        else:
            attn_dim = -1
        if self.gumbel and self.training:
            # attn = F.gumbel_softmax(attn, dim=attn_dim, hard=self.hard, tau=1)
            attn = gumbel_softmax(attn, dim=attn_dim, hard=self.hard,
                                  tau=self.gumbel_tau)
        elif self.categorical and self.training:
            attn = hard_softmax_sample(attn, dim=attn_dim)
        else:
            if self.hard:
                attn = hard_softmax(attn, dim=attn_dim)
            else:
                attn = F.softmax(attn, dim=attn_dim)

        if self.inv_attn:
            attn = attn / (attn.sum(dim=-1, keepdim=True) + 1)
        #
        return attn

    def forward(self, query, key=None, *, value=None, attn_weight=None):
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
               f'inv_attn: {self.inv_attn}, \n' \
               f'gumbel: {self.gumbel}, \n' \
               f'categorical={self.categorical}, \n' \
               f'gumbel_tau: {self.gumbel_tau}'

class TokenAssign(nn.Module):

    def __init__(self, dim, out_dim, num_heads, num_cluster, out_seq_len,
                 with_cls_token, norm_layer,
                 mlp_ratio=(0.5, 4.0), hard=True, inv_attn=True, gumbel=False,
                 categorical=False, inter_mode='attn',
                 assign_skip=True, gumbel_tau=1.,
                 inter_hard=False, inter_gumbel=False):
        super(TokenAssign, self).__init__()
        self.hard = hard
        self.inv_attn = inv_attn
        self.gumbel = gumbel
        self.categorical = categorical
        assert inter_mode in ['attn', 'linear', 'copy', 'mixer']
        self.inter_mode = inter_mode
        self.inter_hard = inter_hard
        self.inter_gumbel = inter_gumbel
        self.with_cls_token = with_cls_token
        self.out_seq_len = out_seq_len
        self.assign_skip = assign_skip
        # norm on cluster_tokens
        self.norm_tokens = norm_layer(dim)
        tokens_dim, channels_dim = [int(x * dim) for x in to_2tuple(mlp_ratio)]
        if inter_mode == 'attn':
            self.inter_attn = Attention(dim=dim, out_dim=out_seq_len, num_heads=num_heads, qkv_bias=True)
        elif inter_mode == 'linear':
            self.inter_proj = nn.Linear(dim, out_seq_len)
        elif inter_mode == 'copy':
            assert num_cluster == out_seq_len, f'{num_cluster} vs {out_seq_len}'
        elif inter_mode == 'mixer':
            self.mlp_inter = Mlp(num_cluster, tokens_dim, out_seq_len)
            self.norm_post_tokens = norm_layer(dim)
        else:
            raise ValueError
        # norm on x
        self.norm_x = norm_layer(dim)
        self.assign = AssignAttention(dim=dim, num_heads=num_heads,
                                      qkv_bias=True, hard=hard,
                                      inv_attn=inv_attn, gumbel=gumbel,
                                      categorical=categorical,
                                      gumbel_tau=gumbel_tau)
        self.norm_new_x = norm_layer(dim)
        self.mlp_channels = Mlp(dim, channels_dim, out_dim)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(
                norm_layer(dim),
                nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()

    def extra_repr(self) -> str:
        return f'inter_mode={self.inter_mode}, \n' \
               f'hard={self.hard}, \n' \
               f'inv_attn={self.inv_attn}, \n' \
               f'gumbel={self.gumbel}, \n' \
               f'categorical={self.categorical}, \n' \
               f'out_seq_len={self.out_seq_len}, \n ' \
               f'assign_skip={self.assign_skip}'

    def interpolate_token(self, x, cluster_tokens):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            cluster_tokens (torch.Tensor): cluster tokens, [B, S_1, C]

        inter_weight (torch.Tensor): [B, S_2, S_1], S_2 is the new number of cluster tokens,
            it's already softmaxed along dim=-1

        Returns:
            inter_cluster_tokens (torch.Tensor): [B, S_2, C]
        """
        if self.inter_mode == 'mixer':
            # [B, S_2, C] <- [B, S_1, C]
            inter_cluster_tokens = self.mlp_inter(cluster_tokens.transpose(1, 2)).transpose(1, 2)
            inter_cluster_tokens = self.norm_post_tokens(inter_cluster_tokens)
            return inter_cluster_tokens
        elif self.inter_mode == 'copy':
            return cluster_tokens
        elif self.inter_mode == 'attn':
            # [N, S_1, S_2]
            inter_weight = self.inter_attn(cluster_tokens, key=x)
        elif self.inter_mode == 'linear':
            # [N, S_1, S_2]
            inter_weight = self.inter_proj(cluster_tokens)
        else:
            raise ValueError
        # [N, S_2, S_1]
        inter_weight = inter_weight.transpose(1, 2)
        if self.inter_gumbel and self.training:
            inter_weight = gumbel_softmax(inter_weight, dim=-1, hard=self.inter_hard, tau=1.)
        else:
            if self.inter_hard:
                inter_weight = hard_softmax(inter_weight, dim=-1)
            else:
                inter_weight = F.softmax(inter_weight, dim=-1)
        # [B, S_2, C]
        inter_cluster_tokens = inter_weight @ cluster_tokens
        return inter_cluster_tokens

    def forward(self, x, cluster_tokens):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            cluster_tokens (torch.Tensor): cluster tokens, [B, S_1, C]

        Returns:
            new_x (torch.Tensor): [B, S_2, C], S_2 is the new number of cluster tokens
        """
        cluster_tokens = self.norm_tokens(cluster_tokens)
        x = self.norm_x(x)
        # [B, S_2, C]
        inter_cluster_tokens = self.interpolate_token(x, cluster_tokens)
        if self.with_cls_token:
            new_x = self.assign(inter_cluster_tokens, x[:, 1:])
        else:
            new_x = self.assign(inter_cluster_tokens, x)
        if self.assign_skip:
            new_x += inter_cluster_tokens

        if self.with_cls_token:
            new_x = torch.cat((x[:, :1], new_x), dim=1)

        new_x = self.reduction(new_x) + self.mlp_channels(self.norm_new_x(new_x))

        return new_x, None

class TokenLearner(nn.Module):

    def __init__(self, dim, out_dim, out_seq_len, hw_shape,
                 norm_layer,
                 num_convs=3,
                 with_cls_token=False, act_layer=nn.GELU):
        super(TokenLearner, self).__init__()
        self.hw_shape = hw_shape
        self.norm = norm_layer(dim)
        convs = []
        for i in range(num_convs - 1):
            convs.append(
                nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=(3, 3),
                          padding=(1, 1)))
            convs.append(act_layer())
        convs.append(nn.Conv2d(in_channels=dim, out_channels=out_seq_len,
                               kernel_size=(3, 3), padding=(1, 1)))
        self.convs = nn.Sequential(*convs)
        if out_dim is not None and dim != out_dim:
            self.reduction = nn.Sequential(
                norm_layer(dim),
                nn.Linear(dim, out_dim, bias=False))
        else:
            self.reduction = nn.Identity()
        self.sigmoid = nn.Sigmoid()
        assert not with_cls_token

    def forward(self, x):
        # [B, L, S1]
        attn, hw_shape = attention_pool(self.norm(x), self.convs, self.hw_shape, False)
        attn = self.sigmoid(attn)
        # [B, S1, C]
        x = attn.transpose(1, 2) @ x
        x = self.reduction(x)

        return x, None



class Attention(nn.Module):
    def __init__(self, dim, num_heads, out_dim=None, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., qkv_fuse=False):
        super().__init__()
        if out_dim is None:
            out_dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv_fuse = qkv_fuse

        if qkv_fuse:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        else:
            self.q_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.k_proj = nn.Linear(dim, dim, bias=qkv_bias)
            self.v_proj = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, out_dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def extra_repr(self) -> str:
        return f'num_heads={self.num_heads}, \n' \
               f'qkv_bias={self.scale}, \n' \
               f'qkv_fuse={self.qkv_fuse}'

    def forward(self, query, key=None, *, value=None, mask=None):
        if self.qkv_fuse:
            assert key is None
            assert value is None
            x = query
            B, N, C = x.shape
            S = N
            # [3, B, nh, N, C//nh]
            qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                      C // self.num_heads).permute(2, 0, 3, 1, 4)
            # [B, nh, N, C//nh]
            q, k, v = qkv[0], qkv[1], qkv[
                2]  # make torchscript happy (cannot use tensor as tuple)
        else:
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
        if mask is not None:
            attn = attn + mask.unsqueeze(dim=1)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, self.num_heads, N, S)

        # [B, nh, N, C//nh] -> [B, N, C]
        # out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = rearrange(attn @ v, 'b h n c -> b n (h c)', h=self.num_heads, b=B,
                        n=N, c=C // self.num_heads)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class AttentionAvg(nn.Module):
    def __init__(self, dim, qk_bias=False, qk_scale=None,
                 attn_drop=0.):
        super().__init__()
        self.scale = qk_scale or dim ** -0.5

        self.q_proj = nn.Linear(dim, dim, bias=qk_bias)
        self.k_proj = nn.Linear(dim, dim, bias=qk_bias)
        self.attn_drop = nn.Dropout(attn_drop)

    def forward(self, query, *, key=None, value=None):
        B, N, C = query.shape
        if key is None:
            key = query
        if value is None:
            value = key
        S = key.size(1)
        # [B, N, C]
        q = self.q_proj(query)
        # [B, S, C]
        k = self.k_proj(key)

        # [B, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        assert attn.shape == (B, N, S)

        # [B, N, C]
        out = attn @ value
        return out

class CrossAttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm_q = norm_layer(dim)
        self.norm_k = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path_prob = drop_path
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim,
                       act_layer=act_layer, drop=drop)
    def extra_repr(self) -> str:
        return f"drop_path_prob={self.drop_path_prob}"

    def drop_path(self, x, drop_prob):
        if drop_prob is None:
            drop_prob = self.drop_path_prob
        return drop_path(x, drop_prob, self.training)

    def forward(self, query, key, *, mask=None, drop_prob=None):
        x = query
        x = x + self.drop_path(
            self.attn(self.norm_q(query), self.norm_k(key), mask=mask),
            drop_prob=drop_prob)
        x = x + self.drop_path(self.mlp(self.norm2(x)), drop_prob=drop_prob)
        return x

class AttnBlock(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, qkv_fuse=True)
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


    def forward(self, x, mask=None, drop_prob=None):
        x = x + self.drop_path(self.attn(self.norm1(x), mask=mask), drop_prob=drop_prob)
        x = x + self.drop_path(self.mlp(self.norm2(x)), drop_prob=drop_prob)
        return x

class BasicLayer(nn.Module):
    """ A basic Transformer layer for one stage.

    Args:
        dim (int): Number of input channels.
        input_seq_len (int): Input resolution.
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

    def __init__(self, dim, input_seq_len, depth, num_heads, num_cluster,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0.,
                 attn_drop=0., drop_path=0.,
                 norm_layer=nn.LayerNorm, downsample=None,
                 use_checkpoint=False,
                 with_cls_token=True,
                 cluster_weight_proj=None,
                 zero_init_cluster_token=False,
                 cluster_attn_avg=False,
                 attn_mask_style=['c2c'],
                 cluster_as_key=True):

        super().__init__()
        self.dim = dim
        self.input_length = input_seq_len
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.num_cluster = num_cluster
        if num_cluster > 0:
            self.cluster_token = nn.Parameter(torch.zeros(1, num_cluster, dim))
            if not zero_init_cluster_token:
                trunc_normal_(self.cluster_token, std=.02)
        else:
            self.cluster_token = None
        self.with_cls_token = with_cls_token
        self.cluster_attn_avg = cluster_attn_avg
        assert len(set(attn_mask_style) - {'c2c'}) == 0
        self.cluster_as_key = cluster_as_key

        if len(attn_mask_style) and num_cluster > 0:
            attn_mask = torch.zeros((1, input_seq_len + num_cluster,
                                     input_seq_len + num_cluster))
            if 'c2c' in attn_mask_style:
                attn_mask[:, -num_cluster:, -num_cluster:] = 1
                attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(
                    attn_mask == 0, float(0.0))
                assert cluster_as_key
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

        if self.with_cluster_token and cluster_attn_avg:
            self.attn_avg = AttentionAvg(dim=dim, qk_bias=qkv_bias,
                                         qk_scale=qk_scale, attn_drop=attn_drop)

        # build blocks
        self.depth = depth
        if self.cluster_as_key or not self.with_cluster_token:
            self.blocks = nn.ModuleList([
                AttnBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                          attn_drop=attn_drop, drop_path=drop_path[i],
                          norm_layer=norm_layer)
                for i in range(depth)])
        else:
            self.blocks = nn.ModuleList([
                CrossAttnBlock(dim=dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                          qkv_bias=qkv_bias, qk_scale=qk_scale, drop=drop,
                          attn_drop=attn_drop, drop_path=drop_path[i],
                          norm_layer=norm_layer)
                for i in range(depth)])

        self.downsample = downsample
        self.input_resolution = input_seq_len
        self.use_checkpoint = use_checkpoint

        self.cluster_weight_proj = cluster_weight_proj
        if isinstance(cluster_weight_proj, nn.Linear):
            if cluster_weight_proj.in_features != dim:
                self.cluster_proj = nn.Sequential(
                    norm_layer(cluster_weight_proj.in_features),
                    nn.Linear(cluster_weight_proj.in_features, dim))
            else:
                self.cluster_proj = nn.Identity()

    @property
    def with_cluster_token(self):
        return self.cluster_token is not None

    def extra_repr(self) -> str:
        return f"dim={self.dim}, \n" \
               f"input_resolution={self.input_resolution}, \n" \
               f"depth={self.depth}, \n" \
               f"num_cluster={self.num_cluster}, \n" \
               f"cluster_as_key={self.cluster_as_key}" \

    def split_x(self, x):
        if self.with_cluster_token:
            return x[:, :-self.num_cluster], x[:, -self.num_cluster:]
        else:
            return x, None

    def concat_x(self, x, cluster_token=None):
        if cluster_token is None:
            return x
        return torch.cat([x, cluster_token], dim=1)

    def forward(self, x, prev_cluster_token=None, return_all_x=False):
        """
        Args:
            x (torch.Tensor): image tokens, [B, L, C]
            prev_cluster_token (torch.Tensor): cluster tokens, [B, S_1, C]
            return_all_x (bool): whether return all intermediate feature
        """
        outs = []
        if self.with_cluster_token:
            cluster_token = self.cluster_token.expand(x.size(0), -1, -1)
            if self.cluster_weight_proj is not None:
                # [B, S_2, S_1]
                inter_weight = self.cluster_weight_proj(prev_cluster_token).transpose(1, 2).softmax(dim=-1)
                cluster_token = cluster_token + inter_weight @ self.cluster_proj(prev_cluster_token)

            if self.cluster_attn_avg:
                cluster_token = self.attn_avg(cluster_token, key=x)
        else:
            cluster_token = None

        B, L, C = x.shape
        cat_x = self.concat_x(x, cluster_token)
        for blk_idx, blk in enumerate(self.blocks):
            if self.cluster_as_key or not self.with_cluster_token:
                if self.use_checkpoint:
                    cat_x = checkpoint.checkpoint(blk, cat_x, mask=self.attn_mask)
                else:
                    cat_x = blk(cat_x, mask=self.attn_mask)
            else:
                if self.use_checkpoint:
                    cat_x = checkpoint.checkpoint(blk, cat_x, x, mask=self.attn_mask)
                else:
                    cat_x = blk(cat_x, x, mask=self.attn_mask)
                x = self.split_x(cat_x)[0]
            if return_all_x:
                outs.append(self.split_x(cat_x)[0])
        x, cluster_token = self.split_x(cat_x)

        if self.downsample is not None:
            if isinstance(self.downsample, TokenAssign):
                x, hw_shape = self.downsample(x, cluster_token)
            else:
                x, hw_shape = self.downsample(x)

        if return_all_x:
            return x, cluster_token, tuple(outs)

        return x, cluster_token

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

class PositionalEncodingFourier(nn.Module):
    """
    Positional encoding relying on a fourier kernel matching the one used in the
    "Attention is all of Need" paper. The implementation builds on DeTR code
    https://github.com/facebookresearch/detr/blob/master/models/position_encoding.py
    """

    def __init__(self, hidden_dim=32, dim=768, temperature=10000):
        super().__init__()
        self.token_projection = nn.Conv2d(hidden_dim * 2, dim, kernel_size=1)
        self.scale = 2 * np.pi
        self.temperature = temperature
        self.hidden_dim = hidden_dim
        self.dim = dim

    def forward(self, B, H, W):
        mask = torch.zeros(B, H, W).bool().to(self.token_projection.weight.device)
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
        x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.hidden_dim, dtype=torch.float32, device=mask.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.hidden_dim)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(),
                             pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(),
                             pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        pos = self.token_projection(pos)
        pos = rearrange(pos, 'b c h w -> b (h w) c')
        return pos


class ClusterViT(nn.Module):
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

    def __init__(self,
                 img_size=224,
                 patch_size=4,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=96,
                 embed_factors=[1, 2, 4, 8],
                 depths=[1, 1, 10, 1],
                 dim_per_head=96,
                 num_clusters=(64, 32, 16, 8),
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.1,
                 attn_mask_style=['c2c'],
                 norm_layer=nn.LayerNorm,
                 patch_norm=True,
                 use_checkpoint=False,
                 downsample_types=['conv', 'conv', 'conv'],
                 num_assign=[-1, -1, -1],
                 assign_type=('gumbel', 'hard', 'inv'),
                 assign_skip=True,
                 inter_mode='attn',
                 inter_type=(),
                 with_gap=False,
                 pos_embed_type='simple',
                 cluster_token_wd=False,
                 patch_embed_type='simple',
                 with_cluster_proj=False,
                 zero_init_cluster_token=False,
                 gumbel_tau=1.,
                 with_cluster_attn_avg=False,
                 pred_src=['image'],
                 deep_sup=[],
                 cluster_as_key=True,
                 freeze_patch_embed=False):
        super().__init__()
        assert patch_size in [4, 16]
        self.num_classes = num_classes
        assert len(embed_factors) == len(depths) == len(num_clusters)
        assert len(depths)-1 == len(downsample_types) == len(num_assign)
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.dim_per_head = dim_per_head
        self.patch_norm = patch_norm
        self.num_features = int(embed_dim * embed_factors[len(depths)-1])
        self.mlp_ratio = mlp_ratio
        self.qkv_bias = qkv_bias
        self.qk_scale = qk_scale
        self.drop_rate = drop_rate
        self.attn_drop_rate = attn_drop_rate
        self.drop_path_rate = drop_path_rate
        self.with_gap = with_gap
        assert len(set(downsample_types) - {'conv', 'unfold', 'assign', 'none', 'learner'}) == 0
        self.num_clusters = num_clusters
        self.pos_embed_type = pos_embed_type
        assert inter_mode in ['attn', 'linear', 'copy', 'mixer']
        assert len(set(assign_type) - {'categorical', 'hard', 'inv', 'gumbel'}) == 0
        assert len(set(inter_type) - {'hard', 'gumbel'}) == 0
        assert pos_embed_type in ['simple', 'fourier']
        self.cluster_token_wd = cluster_token_wd
        assert patch_embed_type in ['simple', 'stem-BN', 'stem-LN', 'stem-depth-BN', 'stem-depth-LN']
        assert len(set(pred_src) - {'image', 'cluster'}) == 0
        assert len(pred_src) > 0
        self.pred_src = pred_src

        if patch_embed_type == 'simple':
            if patch_size == 16:
                # split image into non-overlapping patches
                self.patch_embed = PatchEmbed(
                    img_size=img_size,
                    kernel_size=16,
                    stride=16,
                    padding=0,
                    in_chans=in_chans, embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None)
            else:
                self.patch_embed = PatchEmbed(
                    img_size=img_size, in_chans=in_chans, embed_dim=embed_dim,
                    norm_layer=norm_layer if self.patch_norm else None)
        else:
            self.patch_embed = DeepPatchEmbed(
                img_size=img_size, total_stride=patch_size, embed_dim=embed_dim,
                depthwise='depth' in patch_embed_type,
                norm_type=patch_embed_type[-2:])
        num_patches = self.patch_embed.num_patches
        patches_resolution = self.patch_embed.patches_resolution
        self.patches_resolution = patches_resolution
        if freeze_patch_embed:
            for param in self.patch_embed.parameters():
                param.requires_grad = False

        if self.with_gap:
            self.avgpool = nn.AdaptiveAvgPool1d(1)
            if pos_embed_type == 'simple':
                self.pos_embed = nn.Parameter(
                    torch.zeros(1, num_patches, embed_dim))
                trunc_normal_(self.pos_embed, std=.02)
            elif pos_embed_type == 'fourier':
                self.pos_embed = PositionalEncodingFourier(dim=embed_dim)
            else:
                raise ValueError
        else:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            assert pos_embed_type == 'simple'
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
            trunc_normal_(self.pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        deep_sup_layers = []

        input_seq_len = num_patches
        hw_shape = self.patches_resolution
        next_input_seq_len = input_seq_len
        next_hw_shape = hw_shape
        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            dim = int(embed_dim * embed_factors[i_layer])
            downsample = None
            if i_layer < self.num_layers -1 :
                out_dim = embed_dim * embed_factors[i_layer + 1]
                if downsample_types[i_layer] == 'conv':
                    downsample = SpatialPool(mode=downsample_types[i_layer],
                                             dim=dim,
                                             out_dim=out_dim,
                                             hw_shape=hw_shape,
                                             with_cls_token=self.with_cls_token,
                                             norm_layer=norm_layer)

                    next_hw_shape = (hw_shape[0] // 2, hw_shape[1] // 2)
                    next_input_seq_len = next_hw_shape[0] * next_hw_shape[1]
                elif downsample_types[i_layer] == 'assign':
                    downsample = TokenAssign(dim=dim,
                                             out_dim=out_dim,
                                             num_heads=dim // dim_per_head,
                                             num_cluster=num_clusters[i_layer],
                                             out_seq_len=num_assign[i_layer],
                                             norm_layer=norm_layer,
                                             hard='hard' in assign_type,
                                             inv_attn='inv' in assign_type,
                                             gumbel='gumbel' in assign_type,
                                             categorical='categorical' in assign_type,
                                             inter_mode=inter_mode,
                                             assign_skip=assign_skip,
                                             with_cls_token=self.with_cls_token,
                                             gumbel_tau=gumbel_tau,
                                             inter_hard='hard' in inter_type,
                                             inter_gumbel='gumbel' in inter_type)
                    next_hw_shape = [-1, -1]
                    next_input_seq_len = num_assign[i_layer]
                elif downsample_types[i_layer] == 'learner':
                    downsample = TokenLearner(dim=dim, out_dim=out_dim,
                                              out_seq_len=num_assign[i_layer],
                                              hw_shape=hw_shape,
                                              with_cls_token=self.with_cls_token,
                                              norm_layer=norm_layer)
                    next_hw_shape = [-1, -1]
                    next_input_seq_len = num_assign[i_layer]
                elif downsample_types[i_layer] == 'none':
                    downsample = None
                else:
                    raise ValueError

                if len(deep_sup) > i_layer:
                    deep_sup_layers.append(
                        nn.ModuleList(
                            nn.ModuleDict(
                                {'group': TokenAssign(
                                    dim=dim,
                                    out_dim=out_dim,
                                    num_heads=dim // dim_per_head,
                                    num_cluster=num_clusters[
                                        i_layer],
                                    out_seq_len=num_assign[
                                        i_layer],
                                    norm_layer=norm_layer,
                                    hard='hard' in assign_type,
                                    inv_attn='inv' in assign_type,
                                    gumbel='gumbel' in assign_type,
                                    categorical='categorical' in assign_type,
                                    inter_mode=inter_mode,
                                    assign_skip=assign_skip,
                                    with_cls_token=self.with_cls_token,
                                    gumbel_tau=gumbel_tau,
                                    inter_hard='hard' in inter_type,
                                    inter_gumbel='gumbel' in inter_type),
                                    'norm': norm_layer(out_dim),
                                    'head': nn.Linear(out_dim, num_classes)
                                }
                            ) for _ in range(deep_sup[i_layer])))

            if i_layer > 0 and with_cluster_proj:
                cluster_weight_proj = nn.Linear(int(embed_dim * embed_factors[i_layer-1]), num_clusters[i_layer])
            else:
                cluster_weight_proj = None
            layer = BasicLayer(dim=dim,
                               input_seq_len=input_seq_len,
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
                               with_cls_token=self.with_cls_token,
                               cluster_weight_proj=cluster_weight_proj,
                               zero_init_cluster_token=zero_init_cluster_token,
                               cluster_attn_avg=with_cluster_attn_avg,
                               attn_mask_style=attn_mask_style,
                               cluster_as_key=cluster_as_key)
            self.layers.append(layer)
            if i_layer < self.num_layers -1 :
                input_seq_len = next_input_seq_len
                hw_shape = next_hw_shape

        if 'image' in pred_src:
            self.norm = norm_layer(self.num_features)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        if len(deep_sup):
            self.deep_sup_layers = nn.ModuleList(deep_sup_layers)
        else:
            self.deep_sup_layers = None

        if 'cluster' in pred_src:
            self.norm_cluster = norm_layer(self.num_features)

        if self.with_cls_token:
            trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    @property
    def with_deep_sup(self):
        return self.deep_sup_layers is not None

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
        keywords = {'anchor_token'}
        if not self.cluster_token_wd:
            keywords.add('cluster_token')
        return keywords

    def get_pos_embed(self, B, H, W):
        if self.pos_embed_type == 'simple':
            return self.pos_embed
        else:
            return self.pos_embed(B, H, W)

    def forward_features(self, x):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1,
                                               -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.get_pos_embed(B, *hw_shape)
        x = self.pos_drop(x)

        cluster_token = None
        for layer in self.layers:
            x, cluster_token = layer(x, cluster_token)

        out = None
        if 'image' in self.pred_src:
            # [B, L, C]
            x = self.norm(x)
            if self.with_gap:
                x = self.avgpool(x.transpose(1, 2))  # B C 1
                x = torch.flatten(x, 1)
            else:
                x = x[:, 0]
            out = x

        if 'cluster' in self.pred_src:
            cluster_token = self.norm_cluster(cluster_token)
            cluster_token = self.avgpool(cluster_token.transpose(1, 2))
            cluster_token = torch.flatten(cluster_token, 1)
            if out is not None:
                out = (out + cluster_token)/2
            else:
                out = cluster_token

        return out

    def forward_deep_sup(self, x):
        B = x.shape[0]
        x, hw_shape = self.patch_embed(x)

        if self.with_cls_token:
            cls_tokens = self.cls_token.expand(B, -1,
                                               -1)  # stole cls_tokens impl from Phil Wang, thanks
            x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.get_pos_embed(B, *hw_shape)
        x = self.pos_drop(x)

        outs = []
        cluster_token = None
        for i, layer in enumerate(self.layers):
            if len(self.deep_sup_layers) > i and self.training:
                x, cluster_token, layer_outs = layer(x, cluster_token, return_all_x=True)
                assert len(self.deep_sup_layers[i]) == len(layer_outs)
                for j, blk_out in enumerate(layer_outs):
                    group = self.deep_sup_layers[i][j]['group']
                    norm = self.deep_sup_layers[i][j]['norm']
                    head = self.deep_sup_layers[i][j]['head']
                    logit = head(self.avgpool(norm(group(blk_out, cluster_token)[0]).transpose(1, 2)).flatten(1))
                    outs.append(logit)
            else:
                x, cluster_token = layer(x, cluster_token)
        x = self.norm(x)
        if self.with_gap:
            x = self.avgpool(x.transpose(1, 2))  # B C 1
            x = torch.flatten(x, 1)
        else:
            x = x[:, 0]
        x = self.head(x)

        if self.training:
            return [x] + outs
        else:
            return x



    def forward(self, x):
        if self.with_deep_sup:
            return self.forward_deep_sup(x)
        x = self.forward_features(x)
        x = self.head(x)
        return x
