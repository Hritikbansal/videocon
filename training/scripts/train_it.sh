#!/bin/bash
DIR=`pwd`
DATETIME=`date +'date_%y-%m-%d_time_%H-%M-%S'`

if [ $MASTER_ADDR ];then
	echo $MASTER_ADDR
    echo $MASTER_PORT
    echo $WORLD_SIZE
    echo $RANK
else
	MASTER_ADDR=127.0.0.1
    MASTER_PORT=2$(($RANDOM % 10))$(($RANDOM % 10))15
    WORLD_SIZE=1
    RANK=0
fi

DISTRIBUTED_ARGS="--nproc_per_node 4 \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

EXP_NAME=videocon_model_lora
SAVE_NAME=videocon_model_lora

SAVE_PATH="${SAVE_NAME}"

max_length=256
micro_batch_size=16
gradient_accumulation_steps=1

train_epochs=2
lr_warmup_iters=200
eval_iter=100

mkdir -p ${SAVE_PATH}

options=" \
	--pretrained-ckpt  <path to mplugowl 7b video> \
	--seq-length ${max_length} \
	--micro-batch-size ${micro_batch_size} \
    --train-epochs ${train_epochs} \
	--num-warmup-steps ${lr_warmup_iters} \
	--gradient-accumulation-steps ${gradient_accumulation_steps} \
	--lr 1e-4 \
	--min-lr 1e-7 \
	--eval-iters ${eval_iter} \
	--save-path ${SAVE_PATH} \
    --clip-grad 1.0 \
	--weight-decay 0.0001 \
	--adam-beta1 0.9 \
	--adam-beta2 0.999 \
	--num-workers 32 \
	--use-lora \
	--all-params \
	--lora-r 32 \
	--gradient-checkpointing \
	--wandb_run_name ${EXP_NAME} \
	--bf16 \
	--loss_objective sequential" 

multimodal_options=" \
	--mm-config configs/video.yaml 
    "

CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch $DISTRIBUTED_ARGS ./pipeline_video/train.py $@ ${options} ${multimodal_options} 2>&1 | tee ${SAVE_PATH}/train.log 