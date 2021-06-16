import argparse
import os
import os.path as osp
import re
import tempfile
from pathlib import Path
from datetime import date

WANDB_KEY = '18a953cf069a567c46b1e613f940e6eb8f878c3d'
AISTORE = 'http://10.150.172.62:51080/v1/objects/imagenet'

def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith(".") and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def parse_args():
    parser = argparse.ArgumentParser(description="Submit to nautilus via kubectl")
    parser.add_argument("config", help="train config file path")
    parser.add_argument("--branch", "-b", type=str, default="dev", help="git clone branch")
    parser.add_argument("--ln-exp", "-l", action="store_true", help="link experiment directory")
    parser.add_argument("--gpus", type=int, default=8, help="number of gpus to use ")
    parser.add_argument("--mem", type=int, default=64, help="amount of memory to use")
    parser.add_argument("--file", "-f", type=str, help="config txt file")
    parser.add_argument("--limit", type=str, default='7d')
    parser.add_argument(
        "--work-space",
        type=str,
        default="lpr-jiaruix",
        choices=["lpr-jiaruix"],
    )
    args, rest = parser.parse_known_args()

    return args, rest


def submit(config, args, rest):
    py_args = " ".join(rest)
    script = "tools/dist_launch.sh"
    base_config = osp.splitext(osp.basename(config))[0]
    job_name = f'{base_config}x{args.gpus}-{date.today().strftime("%m/%d/%y")}'
    git_clone_cmd = f'git clone -b {args.branch} https://github.com/xvjiarui/Swin-Transformer.git && cd Swin-Transformer'
    link_dirs_cmd = f'mkdir -p /work_dirs/swin && ln -s /work_dirs/swin output'
    launch_cmd = f'{script} {config} {args.gpus} --data-path {AISTORE} --web {py_args}'
    instance_name = f'dgx1v.{args.mem}g.{args.gpus}.norm.beta'
    image_name = "nvcr.io/nvidian/lpr/swin:latest"
    ngc_cmd_list = [git_clone_cmd, link_dirs_cmd, launch_cmd]
    ngc_cmd = ' && '.join(ngc_cmd_list)
    ngc_submit_cmd = f'ngc batch run ' \
                     f'--instance {instance_name} ' \
                     f'--name {job_name}' \
                     f'--image {image_name} ' \
                     f'--workspace {args.work_space}:/work_dirs:RW ' \
                     f'--result /result ' \
                     f'--total-runtime {args.limit} ' \
                     f'--commandline {ngc_cmd}'
    print(ngc_submit_cmd)
    os.system(ngc_submit_cmd)


def main():
    args, rest = parse_args()
    if osp.isdir(args.config):
        if args.file is not None:
            with open(args.file) as f:
                submit_cfg_names = [line.strip() for line in f.readlines()]
            for cfg in scandir(args.config, recursive=True):
                if osp.basename(cfg) in submit_cfg_names:
                    submit(osp.join(args.config, cfg), args, rest)
        else:
            for cfg in scandir(args.config, suffix=".py"):
                if "playground" in cfg:
                    continue
                submit(osp.join(args.config, cfg), args, rest)
    else:
        submit(args.config, args, rest)


if __name__ == "__main__":
    main()