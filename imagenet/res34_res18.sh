#!/bin/bash
eval "$(/userhome/gushzh/yzy/miconda/bin/conda shell.bash hook)"
export TORCH_HOME=/code/userhome/gushzh/yzy/CC_conformer/torch_home/

MASTER_IP=$(bash get_master_ip.sh)
echo ${MASTER_IP}
RANK=$1
echo $1

if [ ${RANK} -eq 0 ]
then
 export NCCL_IB_HCA=mlx5_0
 echo "export NCCL_IB_HCA=mlx5_0"
else
 echo "not master!"
fi

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x
DATA_PATH='/userhome/gushzh/data/imagenet'

CONFIG=configs/strategies/distill/resnet_kdt1_1-2_oriKL.yaml
MODEL=tv_resnet18
TEC_MODEL=tv_resnet34
EXP_NAME=res34_res14


# It mainly set scales to all 1

gap_n_blocks=2
gap_n_module=4
is_multi_fc=1


scales="1.0, 0.5, 0.25, 0.125"

final_out_type='acc' # acc

gap_type='enc'
act_type='relu'
start_eval_epoch=0

gap_feat_weight=1

# It is mentioned, if you do not use kl loss, you should modify the yaml, instead of the args.py
feat_distill_weight=5
up_dynamic=1
isAccuX=0
featKL=0
featcls=1
gap_model_detach=0


EVAL_ALPHAS_V="0.0, 0.1, 0.3, 0.5, 1.0"
NORM_OUTPUT=0


torchrun --nproc_per_node=8 --nnodes 1  --master_port=6666 imagenet.py  --world_size 1 \
            -c ${CONFIG} \
            --model ${MODEL} \
            --teacher-model ${TEC_MODEL} \
            --data-path ${DATA_PATH} \
            --experiment ${EXP_NAME} \
            --output_normed ${NORM_OUTPUT} \
            --gap_n_blocks ${gap_n_blocks} \
            --gap_n_module ${gap_n_module} \
            --gap_model_detach ${gap_model_detach} \
            --is_multi_fc ${is_multi_fc} \
            --start_eval_epoch ${start_eval_epoch} \
            --final_out_type ${final_out_type} \
            --feat_distill_weight ${feat_distill_weight} \
            --act_type ${act_type} \
            --gap_type ${gap_type} \
            --eval_alphas "${EVAL_ALPHAS_V}" \
            --gap_feat_weight ${gap_feat_weight} \
            --scales "${scales}" \
            --up_dynamic ${up_dynamic} \
            --isAccuX ${isAccuX} \
            --featKL ${featKL} \
            --featcls ${featcls} \


