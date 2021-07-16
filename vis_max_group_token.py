# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import os
import sys
import argparse
import cv2
import random
import colorsys
from io import BytesIO
import os.path as osp
import math

import skimage.io
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms as pth_transforms
import numpy as np
from PIL import Image
from torchvision import datasets
import torch.nn.functional as F
from einops import rearrange, repeat

from models import build_model
from models.cluster_vit import Attention, AssignAttention
from config import _C, _update_config_from_file

def apply_mask(image, mask, color, alpha=0.5):
    for c in range(3):
        image[:, :, c] = image[:, :, c] * (1 - alpha * mask) + alpha * mask * color[c] * 255
    return image


def random_colors(N, bright=True):
    """
    Generate random colors.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    random.shuffle(colors)
    return colors


def display_instances(image, mask, fname="test", figsize=(5, 5), blur=False, contour=True, alpha=0.5):
    fig = plt.figure(figsize=figsize, frameon=False)
    ax = plt.Axes(fig, [0., 0., 1., 1.])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax = plt.gca()

    N = 1
    mask = mask[None, :, :]
    # Generate random colors
    colors = random_colors(N)

    # Show area outside image boundaries.
    height, width = image.shape[:2]
    margin = 0
    ax.set_ylim(height + margin, -margin)
    ax.set_xlim(-margin, width + margin)
    ax.axis('off')
    masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        color = colors[i]
        _mask = mask[i]
        if blur:
            _mask = cv2.blur(_mask,(10,10))
        # Mask
        masked_image = apply_mask(masked_image, _mask, color, alpha)
        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        if contour:
            padded_mask = np.zeros((_mask.shape[0] + 2, _mask.shape[1] + 2))
            padded_mask[1:-1, 1:-1] = _mask
            contours = find_contours(padded_mask, 0.5)
            for verts in contours:
                # Subtract the padding and flip (y, x) to (x, y)
                verts = np.fliplr(verts) - 1
                p = Polygon(verts, facecolor="none", edgecolor=color)
                ax.add_patch(p)
    ax.imshow(masked_image.astype(np.uint8), aspect='auto')
    fig.savefig(fname)
    print(f"{fname} saved.")
    return

hidden_outputs = {}


def attn_hook(name):
    def hook(self, input, output):
        query = input[0]
        x = query
        B, N, C = x.shape
        S = N
        # [3, B, nh, N, C//nh]
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        # [B, nh, N, C//nh]
        q, k, v = qkv[0], qkv[1], qkv[
            2]  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        hidden_outputs[name] = attn

    return hook

def assign_attn_hook(name):
    def hook(self, input, output):
        query = input[0]
        key = input[1]
        B, N, C = query.shape
        if key is None:
            key = query
        value = key
        S = key.size(1)
        # [B, nh, N, C//nh]
        q = rearrange(self.q_proj(query), 'b n (h c)-> b h n c',
                      h=self.num_heads, b=B, n=N, c=C // self.num_heads)
        # [B, nh, S, C//nh]
        k = rearrange(self.k_proj(key), 'b n (h c)-> b h n c', h=self.num_heads,
                      b=B, c=C // self.num_heads)

        # [B, nh, N, S]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = self.get_attn(attn)
        hidden_outputs[name] = attn

    return hook

def interpolate_pos_encoding(pos_embed, H, W):
    npatch = H *W

    N = pos_embed.shape[1]
    if npatch == N and w == h:
        return pos_embed
    patch_pos_embed = pos_embed
    dim = pos_embed.shape[-1]
    patch_pos_embed = nn.functional.interpolate(
        patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)),
                                dim).permute(0, 3, 1, 2),
        size=(H, W), mode='bicubic')
    patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
    return patch_pos_embed


# patch pos_embed function
def get_pos_embed_closure(self):
    def get_pos_embed(B, H, W=None):
        if self.pos_embed_type == 'simple':
            if self.with_cls_token:
                pos_embed = self.pos_embed[:, 1:]
            else:
                pos_embed = self.pos_embed
            pos_embed = interpolate_pos_encoding(pos_embed, H, W)
            if self.with_cls_token:
                pos_embed = torch.cat((self.pos_embed[:, :1], pos_embed), dim=1)
            return pos_embed
        else:
            return self.pos_embed(B, H, W)
    return get_pos_embed


def register_attn_hook(model):
    for module_name, module in model.named_modules():
        if isinstance(module, Attention):
            module.register_forward_hook(attn_hook(module_name))
            print(f'{module_name} is registered')
        if isinstance(module, AssignAttention):
            module.register_forward_hook(assign_attn_hook(module_name))
            print(f'{module_name} is registered')



if __name__ == '__main__':
    parser = argparse.ArgumentParser('Visualize Self-Attention maps')
    parser.add_argument('cfg', type=str)
    parser.add_argument('checkpoint', type=str,
                        help="Path to pretrained weights to load.")
    parser.add_argument("--data_path", default='data/imagenet/2012/val',
                        type=str, help="Path of the image to load.")
    parser.add_argument('--output_dir', default='vis_attn_seg',
                        help='Path where to save visualizations.')
    parser.add_argument("--threshold", type=float, default=0.6, help="""We visualize masks
        obtained by thresholding the self-attention maps to keep xx% of the mass.""")
    parser.add_argument('--num_imgs', default=20, type=int)
    args = parser.parse_args()
    cfg = _C.clone()
    _update_config_from_file(cfg, args.cfg)
    model = build_model(cfg)

    # build model
    print(model)
    for p in model.parameters():
        p.requires_grad = False
    model.eval()
    model.cuda()
    model.get_pos_embed = get_pos_embed_closure(model)
    state_dict = torch.load(args.checkpoint, map_location="cpu")
    if 'model' in state_dict:
        state_dict = state_dict['model']
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print('Pretrained weights found at {} and loaded with msg: {}'.format(
        args.checkpoint, msg))

    # open image
    transform = pth_transforms.Compose([
        pth_transforms.Resize(512),
        pth_transforms.CenterCrop(448),
        pth_transforms.ToTensor(),
        pth_transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
    inv_normalize = pth_transforms.Normalize(
        mean=[-0.485 / 0.229, -0.456 / 0.224, -0.406 / 0.255],
        std=[1 / 0.229, 1 / 0.224, 1 / 0.255]
    )
    dataset = datasets.ImageFolder(args.data_path, transform=transform)
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    register_attn_hook(model)
    for img_idx in sorted(
            np.random.choice(list(range(len(dataset))), size=args.num_imgs,
                             replace=False)):
        hidden_outputs.clear()
        torch.cuda.empty_cache()
        img, _ = dataset[img_idx]
        # img = transform(img)

        # make the image divisible by the patch size
        h, w = img.shape[1:]
        ori_image = inv_normalize(img).permute(1, 2, 0).cpu().numpy() * 255
        img = img.unsqueeze(0)

        model.forward(img.cuda())

        for k, v in hidden_outputs.items():

            print(k)
            assert isinstance(k, str)
            i_layer = int(k[k.find('layers')+7])
            num_cluster_token = cfg.MODEL.CVIT.NUM_CLUSTERS[i_layer]
            if num_cluster_token == 0:
                continue

            layer_name = '.'.join(k.split('.')[:4])
            output_dir = osp.join(args.output_dir, f'{img_idx:05d}', layer_name)

            if 'downsample' not in k:
                # [B, nH, N+L, N+L]
                full_attention = v
                # [B, nH, L, N]
                attentions = v[:, :, -num_cluster_token:, :-num_cluster_token]
                # [B, nH, N, L]
                attentions = rearrange(attentions, 'b h l n -> b h n l')
                # [B, nH, N, L]
                # attentions = v[:, :, :-num_cluster_token, -num_cluster_token:]
            else:
                # [B, nH, L, N]
                attentions = v
                # [B, nH, N, L]
                attentions = rearrange(attentions, 'b h l n -> b h n l')


            scale = (h * w // attentions.shape[2]) ** 0.5
            if h > w:
                w_featmap = w // int(np.round(scale))
                h_featmap = attentions.shape[2] // w_featmap
            else:
                h_featmap = h // int(np.round(scale))
                w_featmap = attentions.shape[2] // h_featmap
            if not attentions.shape[2] == h_featmap * w_featmap:
                import ipdb

                ipdb.set_trace()
            assert attentions.shape[
                       2] == h_featmap * w_featmap, f'{attentions.shape[2]} = {h_featmap} x {w_featmap}, h={h}, w={w}'

            nh = attentions.shape[1]  # number of head
            # make head as batch
            attentions = attentions[0].reshape(nh, h_featmap, w_featmap, -1)

            # interpolate
            attentions = rearrange(attentions, 'b h w c -> b c h w')
            attentions = F.interpolate(attentions, size=ori_image.shape[:2],
                                       mode='bilinear', align_corners=False)
            attentions = rearrange(attentions, 'b c h w -> b h w c')
            # create valid mask by confidence sum
            # [B]
            max_idx = attentions.sum(dim=(1, 2)).sort(dim=-1, descending=True)[
                         1][..., 0]
            # [B, H, W]
            heat = torch.zeros(attentions.shape[:3], device=attentions.device, dtype=attentions.dtype)
            for j in range(nh):
                heat[j] = attentions[j, :, :, max_idx[j]]
            # we keep only a certain percentage of the mass
            val, idx = torch.sort(heat.reshape(nh, -1))
            val /= torch.sum(val, dim=1, keepdim=True)
            cumval = torch.cumsum(val, dim=1)
            threshold = 0.5
            th_attn = cumval > (1 - threshold)
            idx2 = torch.argsort(idx)
            for j in range(nh):
                th_attn[j] = th_attn[j][idx2[j]]
            th_attn = th_attn.reshape_as(heat).cpu().numpy()
            heat = heat.cpu().numpy()

            # save attentions heatmaps
            os.makedirs(output_dir, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir, "img.jpg"),
                        ori_image[..., ::-1])
            for j in range(nh):
                fname = os.path.join(output_dir, "attn-head_act" + str(j) + ".jpg")
                plt.imsave(fname=fname, arr=heat[j], format='png')
                print(f"{fname} saved.")
                display_instances(ori_image, th_attn[j],
                                  fname=os.path.join(output_dir, "attn-head_area" + str(j) + ".jpg"),
                                  blur=False)

