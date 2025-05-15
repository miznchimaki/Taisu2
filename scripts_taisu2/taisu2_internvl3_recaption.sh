#!/usr/bin/bash


# turn off the execution trace mode
set +x

HOST_FILE=${1:-"$HOME/projects/Taisu2/scripts_taisu2/multinode_hostfile"}
MASTER_ADDR=${2:-$(cat $HOST_FILE | head -n 1 | cut -d" " -f 1)}
NNODES=$(cat $HOST_FILE | wc -l)
NODE_RANK=${3:-0}
MIN_PORT=${3:-23333}
MAX_PORT=${4:-45678}
PORT_RANGE=$((MAX_PORT - MIN_PORT + 1))
MASTER_PORT=$((RANDOM % PORT_RANGE + MIN_PORT))

echo "multi-node deepspeed host file: ${HOST_FILE}"
echo "total multi-node number: ${NNODES}"
echo "current node rank index: ${NODE_RANK}"
echo "master node address: ${MASTER_ADDR}"
echo "communication port of master node: ${MASTER_PORT}"

start_time_stamp=$(date +%Y-%m-%d-%H:%M:%S)
echo "Begin Taisu2 image-alttext pairs recaptioning at ${start_time_stamp}"
# torchrun --nnodes=${NNODES} --nproc_per_node=${GPUS} --master_port=${PORT} \
#     llava/train/train_mem.py \
#     --deepspeed ./scripts/zero3.json \
#     --model_name_or_path ./pretrained/vicuna-7b-v1.5 \
#     --version v1 \
#     --data_path ./playground/llava_v1_5_mix665k.json \
#     --image_folder ./playground/data \
#     --vision_tower ./pretrained/InternViT-6B-224px \
#     --pretrain_mm_mlp_adapter ./work_dirs/pretrain_internvit6b_224to336_vicuna7b/mm_projector.bin \
#     --mm_projector_type mlp2x_gelu \
#     --mm_vision_select_layer -4 \
#     --mm_use_im_start_end False \
#     --mm_use_im_patch_token False \
#     --image_aspect_ratio pad \
#     --group_by_modality_length True \
#     --bf16 True \
#     --output_dir ${OUTPUT_DIR} \
#     --num_train_epochs 1 \
#     --per_device_train_batch_size 16 \
#     --per_device_eval_batch_size 4 \
#     --gradient_accumulation_steps 1 \
#     --evaluation_strategy "no" \
#     --save_strategy "steps" \
#     --save_steps 1000 \
#     --save_total_limit 3 \
#     --learning_rate 2e-5 \
#     --weight_decay 0. \
#     --warmup_ratio 0.03 \
#     --lr_scheduler_type "cosine" \
#     --logging_steps 1 \
#     --tf32 True \
#     --model_max_length 2048 \
#     --gradient_checkpointing True \
#     --dataloader_num_workers 4 \
#     --lazy_preprocess True \
#     --report_to "tensorboard" \
#     | tee ${OUTPUT_DIR}/train.log

end_time_stamp=`date +"%Y-%m-%d-%H:%M:%S"`
echo "End Taisu2 image-alttext pairs recaptioning at ${end_time_stamp}"
