#!/usr/bin/env bash

CONFIG=$1
NODES=$2
GPUS_PER_NODE=$3
PORT=${PORT:-29500}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH OMP_NUM_THREADS=4 \
python -m torch.distributed.launch --nproc_per_node=$GPUS_PER_NODE \
  --nnodes=${NGC_ARRAY_SIZE} --node_rank=${NGC_ARRAY_INDEX} \
  --master_addr=${NGC_MASTER_ADDR}  \
    main.py --cfg $CONFIG ${@:4}
