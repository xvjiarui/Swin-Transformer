import argparse
import os
import os.path as osp
import re
import tempfile
from pathlib import Path
from datetime import date
from collections import OrderedDict

WANDB_KEY = '18a953cf069a567c46b1e613f940e6eb8f878c3d'
AISTORE = 'http://10.150.172.62:51080/v1/objects/imagenet'
DATASET_ID = '80922'

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
    parser.add_argument("--mem", type=int, default=16, choices=[16, 32], help="amount of memory to use")
    parser.add_argument("--file", "-f", type=str, help="config txt file")
    parser.add_argument("--limit", type=str, default='7d')
    parser.add_argument('--wandb', '-w', action='store_true', help='use wandb')
    parser.add_argument('--batch-size', type=int, help="batch size for single GPU")
    parser.add_argument(
        "--work-space",
        type=str,
        default="lpr-jiarui",
        choices=["lpr-jiarui"],
    )
    parser.add_argument(
        "--data-type",
        type=str,
        default="aistore",
        choices=["aistore", "ngc"],
    )
    parser.add_argument("--network", type=str, default='ethernet',
                        choices=['ethernet', 'infiniband'])
    parser.add_argument("--ace-type", type=str, default='norm', choices=['norm', 'norm.beta'])
    args, rest = parser.parse_known_args()

    return args, rest

class Command(object):

    def __init__(self, main_cmd, arg_dict=None):
        self.main_cmd = main_cmd
        self.arg_dict = OrderedDict()
        if arg_dict is not None:
            self.arg_dict.update(arg_dict)

    def __setitem__(self, key, value):
        self.arg_dict[key] = value

    def __getitem__(self, item):
        return self.arg_dict[item]

    @property
    def text(self):
        arg_str = ' '.join([f'--{k} {v}' for k, v in self.arg_dict.items()])
        return f'{self.main_cmd} {arg_str}'

    def compose(self, cmd_list):
        if not isinstance(cmd_list, (list, tuple)):
            cmd_list = [cmd_list]
        cmd_str = self.join([self] + cmd_list)
        return cmd_str

    @staticmethod
    def to_str(cmd):
        if isinstance(cmd, Command):
            return cmd.text
        else:
            return cmd

    @staticmethod
    def join(cmd_list, sep=' && '):
        cmd_str_list = [Command.to_str(cmd) for cmd in cmd_list if len(Command.to_str(cmd))]
        cmd_str = sep.join(cmd_str_list)
        return cmd_str


def submit(config, args, rest):
    num_node = args.gpus//8
    ngc_arg_dict = OrderedDict()
    py_args = " ".join(rest)
    if args.batch_size is not None:
        batch_size = args.batch_size
    else:
        batch_size = 128
    py_args += f" --batch-size {batch_size}"
    if args.wandb:
        py_args += " --wandb "
    base_config = osp.splitext(osp.basename(config))[0]
    ngc_cmd_list = []
    git_clone_cmd = f'git clone -b {args.branch} https://github.com/xvjiarui/Swin-Transformer.git && cd Swin-Transformer'
    ngc_cmd_list.append(git_clone_cmd)
    link_dirs_cmd = f'mkdir -p /work_dirs/swin && ln -s /work_dirs/swin output'
    ngc_cmd_list.append(link_dirs_cmd)
    if args.wandb:
        ngc_cmd_list.append(f'pip install wandb && wandb login {WANDB_KEY}')

    ngc_arg_dict['name'] = f'{base_config}_bs{batch_size}x{args.gpus}'
    ngc_arg_dict['image'] = "nvcr.io/nvidian/lpr/swin:latest"
    ngc_arg_dict['workspace'] = f'{args.work_space}:/work_dirs:RW'
    ngc_arg_dict['result'] = '/result'
    ngc_arg_dict['network'] = f'{args.network.upper()}'
    if args.data_type == 'ngc':
        ngc_arg_dict['datasetid'] = f'{DATASET_ID}:/job_data'
        data_path = '/job_data'
    else:
        data_path = AISTORE
    if num_node > 1:
        gpus = 8
        mem = 32
        ace_type = 'norm'
        script = "tools/dist_mn_launch.sh"
        launch_cmd = f'{script} {config} {num_node} {gpus} --data-path {data_path} --web {py_args}'
        # ngc_arg_dict['total-runtime'] = f'{128//num_node}h'
        ngc_arg_dict['replicas'] = num_node
        ngc_arg_dict['array-type'] = 'MPI'
    else:
        gpus = args.gpus
        mem = args.mem
        ace_type = args.ace_type
        script = "tools/dist_launch.sh"
        launch_cmd = f'{script} {config} {gpus} --data-path {data_path} --web {py_args}'
        # ngc_arg_dict['total-runtime'] = f'{args.limit} '
        ngc_arg_dict['preempt'] = 'RESUMABLE'
    ngc_cmd_list.append(launch_cmd)
    if num_node > 1:
        mpi_cmd = f'mpirun --allow-run-as-root -x IBV_DRIVERS=/usr/lib/libibverbs/libmlx5 -np {num_node} -npernode 1 '
        ngc_arg_dict['commandline'] = f'"{mpi_cmd} bash -c \'{Command.join(ngc_cmd_list)}\'"'
    else:
        ngc_arg_dict['commandline'] = f'"{Command.join(ngc_cmd_list)}"'
    ngc_arg_dict['instance'] = f'dgx1v.{mem}g.{gpus}.{ace_type}'
    ngc_submit_cmd = Command('ngc batch run', ngc_arg_dict).text
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
            for cfg in scandir(args.config, suffix=".yaml"):
                if "playground" in cfg:
                    continue
                submit(osp.join(args.config, cfg), args, rest)
    else:
        submit(args.config, args, rest)


if __name__ == "__main__":
    main()
