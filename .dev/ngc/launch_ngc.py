import argparse
import itertools
import os
import os.path as osp
from collections import OrderedDict
# pip install pyyaml addict
import yaml
from addict import Dict

DATASET_IDS = {
    'gcc3m': '83154',
    'gcc12m': '84486',
    'yfcc14m': '85609',
    'imagenet': '80922',
}


def parse_args():
    parser = argparse.ArgumentParser(description="Submit jobs to NGC")
    parser.add_argument("launch_cfg", help="train config file path")
    args = parser.parse_args()

    return args

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



def sweep_arg_gen(sweep_args):
    sweep_args = OrderedDict(sweep_args)
    return itertools.product(*list(list(values) for values in sweep_args.values()))

def compile_cfg(job_cfg):
    """

    Args:
        job_cfg:

    Returns:
        list[str]

    """
    all_cmds = []
    ngc_arg_dict = Dict()
    ngc_arg_dict.image = job_cfg.NGC_BATCH.IMAGE
    ngc_arg_dict.workspace = job_cfg.NGC_BATCH.WORKSPACE
    ngc_arg_dict.result = job_cfg.NGC_BATCH.RESULT
    ngc_arg_dict.network = job_cfg.NGC_BATCH.NETWORK
    ngc_arg_dict['total-runtime'] = job_cfg.NGC_BATCH.LIMIT
    ngc_arg_dict.instance = job_cfg.NGC_BATCH.INSTANCE
    if job_cfg.NGC_BATCH.get('REPLICAS', 1) > 1:
        ngc_arg_dict.replicas = job_cfg.NGC_BATCH.REPLICAS
        ngc_arg_dict['array-type'] = 'MPI'
    else:
        ngc_arg_dict.replicas = 1
        ngc_arg_dict.preempt = 'RESUMABLE'
    datasets_list = []
    for dataset in job_cfg.NGC_BATCH.DATASETS:
        for ds in DATASET_IDS:
            if dataset.startswith(ds):
                datasets_list.append(dataset.replace(ds, DATASET_IDS[ds], 1))
    ngc_arg_dict.datasetid = ' --datasetid '.join(datasets_list)
    for sweep_args in sweep_arg_gen(job_cfg.LAUNCH.SWEEP_ARGS):
        cur_ngc_arg_dict = ngc_arg_dict.copy()
        name_args_dict = dict(zip(job_cfg.LAUNCH.SWEEP_ARGS.keys(), sweep_args))
        if 'config' in name_args_dict:
            name_args_dict['config'] = osp.splitext(osp.basename(name_args_dict['config']))[0]
        cur_ngc_arg_dict.name = job_cfg.NGC_BATCH.NAME.format_map(name_args_dict)
        cmd_args_dict = dict(zip(job_cfg.LAUNCH.SWEEP_ARGS.keys(), sweep_args), **job_cfg.LAUNCH.ARGS)
        cmd_list = []
        cmd_list.extend(job_cfg.PRE_LAUNCH)
        for template in job_cfg.LAUNCH.TEMPLATES:
            cmd_list.append(template.format_map(cmd_args_dict))
        cmd_str = Command.join([Command.join(cmd_list), *job_cfg.POST_LAUNCH], sep=' ; ')
        if cur_ngc_arg_dict.replicas > 1:
            mpi_cmd = f'mpirun --allow-run-as-root -x IBV_DRIVERS=/usr/lib/libibverbs/libmlx5 -np {ngc_arg_dict.replicas} -npernode 1 '
            cmd_str = f'{mpi_cmd} bash -c \'{cmd_str}\''
        cur_ngc_arg_dict.commandline = f'"{cmd_str}"'
        ngc_submit_cmd = Command('ngc batch run', cur_ngc_arg_dict).text
        all_cmds.append(ngc_submit_cmd)

    return all_cmds


def main():
    args = parse_args()
    with open(args.launch_cfg) as f:
        launch_cfg = Dict(yaml.safe_load(f))
    for job_set in launch_cfg.JOB_CFGS:
        cmds = compile_cfg(job_set)
        for cmd in cmds:
            print(cmd)
            os.system(cmd)



if __name__ == "__main__":
    main()
