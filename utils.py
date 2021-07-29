# --------------------------------------------------------
# Swin Transformer
# Copyright (c) 2021 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ze Liu
# --------------------------------------------------------

import os
import subprocess
import torch
import torch.distributed as dist

try:
    # noinspection PyUnresolvedReferences
    from apex import amp
except ImportError:
    amp = None


def load_pretrained(config, model, ckpt_path, logger):
    state_dict = torch.load(ckpt_path, map_location='cpu')['model']

    if config.MODEL.REMOVE_PRETRAINED_HEAD:
        popping_prefix = ['head', 'norm']
        popping_keys = []
        for k in state_dict.keys():
            for prefix in popping_prefix:
                if k.startswith(prefix):
                    popping_keys.append(k)
        for k in popping_keys:
            state_dict.pop(k)

        logger.info(f'Popping {popping_keys} from state dict')

    missing_keys, unexpected_keys = model.load_state_dict(state_dict,
                                                          strict=False)
    logger.info(f'loaded pretrained checkpoint from: {ckpt_path}')
    if len(missing_keys) > 0 or len(unexpected_keys) > 0:
        logger.warning(
            f'Missing keys: {missing_keys}\nUnexpected keys: {unexpected_keys}')


def load_checkpoint(config, model, optimizer, lr_scheduler, loss_scaler, logger):
    logger.info(f"==============> Resuming form {config.MODEL.RESUME}....................")
    if config.MODEL.RESUME.startswith('https'):
        checkpoint = torch.hub.load_state_dict_from_url(
            config.MODEL.RESUME, map_location='cpu', check_hash=True)
    else:
        checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        if loss_scaler is not None:
            loss_scaler.load_state_dict(checkpoint['loss_scaler'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def save_checkpoint(config, epoch, model, max_accuracy, optimizer, lr_scheduler, loss_scaler, logger, suffix=''):
    save_state = {'model': model.state_dict(),
                  'optimizer': optimizer.state_dict(),
                  'lr_scheduler': lr_scheduler.state_dict(),
                  'max_accuracy': max_accuracy,
                  'epoch': epoch,
                  'config': config}
    if loss_scaler is not None:
        save_state['loss_scaler'] = loss_scaler.state_dict()
    if config.AMP_OPT_LEVEL != "O0":
        save_state['amp'] = amp.state_dict()

    if len(suffix) > 0 and not suffix.startswith('_'):
        suffix = '_' + suffix
    filename = f'ckpt_epoch_{epoch}{suffix}.pth'

    save_path = os.path.join(config.OUTPUT, filename)
    logger.info(f"{save_path} saving......")
    torch.save(save_state, save_path)
    logger.info(f"{save_path} saved !!!")
    if config.MAX_KEPT_CKPT > 0:
        if epoch >= config.MAX_KEPT_CKPT:
            logger.info(f"Epoch: {epoch}, greater than config.MAX_KEPT_CKPT: {config.MAX_KEPT_CKPT}")
            end_clean_epoch = epoch - config.MAX_KEPT_CKPT
            old_path_list = []
            for cur_clean_epoch in range(end_clean_epoch+1):
                old_path = os.path.join(config.OUTPUT, f'ckpt_epoch_{cur_clean_epoch}{suffix}.pth')
                if os.path.exists(old_path) :
                    logger.info(f"old checkpoint path {old_path} exits")
                    old_path_list.append(old_path)
            for old_path in old_path_list[:-config.MAX_KEPT_CKPT]:
                os.remove(old_path)
                logger.info(f"old checkpoint path {old_path} removed!!!")


def get_grad_norm(parameters, norm_type=2):
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    norm_type = float(norm_type)
    total_norm = 0
    for p in parameters:
        param_norm = p.grad.data.norm(norm_type)
        total_norm += param_norm.item() ** norm_type
    total_norm = total_norm ** (1. / norm_type)
    return total_norm


def auto_resume_helper(output_dir):
    checkpoints = os.listdir(output_dir)
    checkpoints = [ckpt for ckpt in checkpoints if ckpt.endswith('pth')]
    print(f"All checkpoints founded in {output_dir}: {checkpoints}")
    if len(checkpoints) > 0:
        latest_checkpoint = max([os.path.join(output_dir, d) for d in checkpoints], key=os.path.getmtime)
        print(f"The latest checkpoint founded: {latest_checkpoint}")
        resume_file = latest_checkpoint
    else:
        resume_file = None
    return resume_file


def reduce_tensor(tensor):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt
