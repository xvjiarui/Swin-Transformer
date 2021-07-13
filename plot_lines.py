import argparse
import os.path as osp

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pgf import PdfPages
from collections import OrderedDict

# plt.rcParams.update({
#     "text.usetex": True,
#     "font.family": "sans-serif",
#     "font.sans-serif": ["Helvetica"]})
## for Palatino and other serif fonts use:
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'Times New Roman',
    # "font.serif": ["Palatino"],
    'font.size': 15,
    'xtick.labelsize': 18,
    'ytick.labelsize': 18,
})


def parse_args():
    parser = argparse.ArgumentParser(description='plot ImageNet acc')
    parser.add_argument('data', help='input data')
    parser.add_argument('out', help='output file')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    data = pd.read_csv(args.data)
    np.random.seed(19680801)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w']
    colors = ['#ef476f', '#ffd166', '#06d6a0', '#118ab2', '#073b4c']

    # with grid
    ax = None
    # model_name_mapping = OrderedDict(**{'cait': 'CaiT', 'deit': 'DeiT'})
    model_name_mapping = OrderedDict()
    model_name_mapping['tf_efficientnet'] = 'EfficientNet'
    model_name_mapping['regnety'] = 'RegNetY'
    model_name_mapping['resnet'] = 'ResNet'
    model_name_mapping['swin'] = 'Swin ViT'
    model_name_mapping['vit'] = 'ViT'
    # model_name_mapping['deit'] = 'DeiT'
    # metric = 'param_count'
    # data.sort_values(by=[metric], inplace=True)
    for i, model in enumerate(model_name_mapping):
        data_m = data.loc[lambda df: df['model'].str.match(model), :]
        # ax = data_m.plot(x=metric, y='top1', ax=ax)
        ax = data_m.plot.scatter(x='infer_samples_per_sec', y='top1',
                                 s=(data_m['param_count']*3)**1.3,
                                 # c=np.repeat(colors[i], data_m.shape[0]),
                                 c=colors[i],
                                 label=model_name_mapping[model],
                                 alpha=0.5,
                                 ax=ax)
    # ax.legend(model_name_mapping.values(), scatterpoints=1)
    # ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    lgnd = ax.legend(markerscale=1, scatterpoints=1)
    for handle in lgnd.legendHandles:
        handle.set_sizes([96.0])
    ax.set_xscale('log', base=2)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(8))
    # ax.yaxis.set_major_locator(ticker.FixedLocator([58, 62, 66]))
    # ax.yaxis.set_minor_locator(ticker.MaxNLocator(5))
    # ax.xaxis.set_major_locator(ticker.FixedLocator([0, 100]))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    # ax.xaxis.set_minor_locator(ticker.MaxNLocator(10))
    # ax.xaxis.set_major_locator(ticker.LogLocator(base=10, numticks=5))
    # ax.xaxis.get_major_formatter().labelOnlyBase = False
    ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
    ax.xaxis.set_major_locator(ticker.FixedLocator([25, 50, 100, 200, 500, 1000, 2000]))
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    # ax.set_ylim([data.min().min(), data.max().max()])
    ax.set_ylim([76, 90])
    ax.set_xlim([20, 3000])
    # ax.set_xlabel(xlabel='Number of Parameter (M)', labelpad=-5, fontsize=18)
    ax.set_xlabel(xlabel='Throughput (images/s)', labelpad=0, fontsize=18)
    ax.set_ylabel('ImageNet Acc@1 (\%)', labelpad=5, fontsize=18)
    # plt.grid(True, which='both', color='0.95')
    # plt.show()
    # plt.savefig(args.out, backend='pgf', bbox_inches = 'tight', pad_inches = 0)
    plt.savefig(args.out)


if __name__ == '__main__':
    main()
