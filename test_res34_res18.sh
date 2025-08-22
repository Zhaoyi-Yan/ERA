#!/bin/bash
eval "$(/userhome/YOUR_ANACONDA/anaconda/bin/conda shell.bash hook)"

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x
DATA_PATH='IMAGENET_PATH/imagenet'

CONFIG=configs/strategies/distill/resnet.yaml
MODEL=tv_resnet18
TEC_MODEL=tv_resnet34
EXP_NAME=imagenet_res34_res18
RESUME_CKPT=checkpoint-94.pth

gap_n_blocks=2
gap_n_module=4
is_multi_fc=1

gap_type='enc'
act_type='relu'


EVAL_ALPHAS_V="0.0, 0.5, 1.0"


torchrun --nproc_per_node=8 --nnodes 1  --master_port=6666 test_ERA.py  --world_size 1 \
            -c ${CONFIG} \
            --model ${MODEL} \
            --teacher-model ${TEC_MODEL} \
            --data-path ${DATA_PATH} \
            --experiment ${EXP_NAME} \
            --gap_n_blocks ${gap_n_blocks} \
            --gap_n_module ${gap_n_module} \
            --is_multi_fc ${is_multi_fc} \
            --act_type ${act_type} \
            --gap_type ${gap_type} \
            --eval_alphas "${EVAL_ALPHAS_V}" \
            --resume_ckpt ${RESUME_CKPT} \



