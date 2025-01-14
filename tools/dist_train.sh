#!/usr/bin/env bash

CONFIG=$1
GPUS=$2
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29508}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

#! python -m torch.distributed.launch表示需要进行分布式训练
#！--nnodes参数指明有几台机器（一台机器可能有多个gpu，即多卡）--nproc_per_node指明每台机器上有多少进程（一般一个进程负责一个GPU）
#! --master_addr和--master_port分别指第一块GPU所在主机的ip地址和端口，这两个参数用于分布式训练期间进行同步.--node_rank 当期机器的编号，在多机多卡中，这个参数的值会不一样
#！把前面这些参数指定之后，再指定要跑的py脚本，以及那个脚本的命令行参数
#！nnodes这些参数会被自动写到环境变量中
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    $(dirname "$0")/train.py \
    $CONFIG \
    --seed 0 \
    --launcher pytorch ${@:3}

#！上述命令是多机多卡的命令。如果是单机多卡，仅需要保留--nproc_per_node参数即可
#！上述更改后的单机多卡命令只适用于pytorch 1.x版本，在2.0以上版本，torch.distributed.launch被弃用，把torch.distributed.launch改为torch.distributed.run即可