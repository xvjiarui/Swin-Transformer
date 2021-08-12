# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------
import dis
import os
import torch
import numpy as np
import torch.distributed as dist
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import Mixup
from timm.data import create_transform
from timm.data.transforms import _pil_interp
import webdataset as wds

from .cached_image_folder import CachedImageFolder
from .samplers import SubsetRandomSampler

TRAIN_LEN = 1281167
# ImageNet-21k
# TRAIN_LEN = 14197122
VAL_LEN = 50000

def build_loader(config):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config)
    config.freeze()
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build train dataset")
    dataset_val, _ = build_dataset(is_train=False, config=config)
    print(f"local rank {config.LOCAL_RANK} / global rank {dist.get_rank()} successfully build val dataset")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    if config.DATA.WEB_MODE:
        data_loader_train = wds.WebLoader(
            dataset_train.batched(config.DATA.BATCH_SIZE, partial=False),
            batch_size=None, shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            persistent_workers=True)

        train_nbatches = max(1, TRAIN_LEN // (config.DATA.BATCH_SIZE * dist.get_world_size()))
        data_loader_train = (
            data_loader_train.with_epoch(train_nbatches)
                .with_length(train_nbatches))

        data_loader_val = wds.WebLoader(
            dataset_val.batched(config.DATA.BATCH_SIZE),
            batch_size=None, shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            persistent_workers=True)

        val_nbatches = max(1, VAL_LEN // (config.DATA.BATCH_SIZE * dist.get_world_size()))
        data_loader_val = (
            data_loader_val.with_epoch(val_nbatches)
                .with_length(val_nbatches))

    else:
        if config.DATA.ZIP_MODE and config.DATA.CACHE_MODE == 'part':
            indices = np.arange(dist.get_rank(), len(dataset_train), dist.get_world_size())
            sampler_train = SubsetRandomSampler(indices)
        else:
            sampler_train = torch.utils.data.DistributedSampler(
                dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
            )

        indices = np.arange(dist.get_rank(), len(dataset_val), dist.get_world_size())
        sampler_val = SubsetRandomSampler(indices)

        data_loader_train = torch.utils.data.DataLoader(
            dataset_train, sampler=sampler_train,
            batch_size=config.DATA.BATCH_SIZE,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=True,
        )

        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=config.DATA.BATCH_SIZE,
            shuffle=False,
            num_workers=config.DATA.NUM_WORKERS,
            pin_memory=config.DATA.PIN_MEMORY,
            drop_last=False
        )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config):
    transform = build_transform(is_train, config)
    if config.DATA.DATASET == 'imagenet':
        prefix = 'train' if is_train else 'val'
        if config.DATA.ZIP_MODE:
            ann_file = prefix + "_map.txt"
            prefix = prefix + ".zip@/"
            dataset = CachedImageFolder(config.DATA.DATA_PATH, ann_file, prefix, transform,
                                        cache_mode=config.DATA.CACHE_MODE if is_train else 'part')
        elif config.DATA.WEB_MODE:
            # repeat dataset infinitely for training mode
            repeat = is_train
            prefix = 'imagenet-train-{000000..001281}.tar' if is_train else 'imagenet-val-{000000..000049}.tar'
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            length = TRAIN_LEN if is_train else VAL_LEN

            # shuffle in training mode but not for validation
            shuffle = config.DATA.SHUFFLE_BUFFER if is_train else 0


            dataset = (
                wds.WebDataset(root, repeat=repeat)
                    .shuffle(shuffle)
                    .decode("pil")
                    .to_tuple("jpg;png;jpeg cls")
                    .map_tuple(transform)
            )
            # sampler for validation set
            if not is_train:
                dataset = dataset.slice(dist.get_rank(), VAL_LEN, dist.get_world_size())

            dataset = dataset.with_length(length)

        else:
            root = os.path.join(config.DATA.DATA_PATH, prefix)
            dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
        # ImageNet-21k
        # nb_classes = 21841
    else:
        raise NotImplementedError("We only support ImageNet Now.")

    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)
