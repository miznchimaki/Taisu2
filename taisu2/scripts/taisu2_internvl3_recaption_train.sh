#!/usr/bin/bash


# turn off the execution trace mode
set +x

HOST_FILE=${1:-"$HOME/projects/Taisu2/taisu2/scripts/ds_hostfile"}
MASTER_ADDR=${2:-$(cat $HOST_FILE | head -n 1 | cut -d" " -f 1)}
NNODES=$(cat $HOST_FILE | wc -l)
NODE_RANK=${3:-0}
RANDOM=${4:-42}
MIN_PORT=${5:-23333}
MAX_PORT=${6:-45678}
PORT_RANGE=$((MAX_PORT - MIN_PORT + 1))
MASTER_PORT=$((RANDOM % PORT_RANGE + MIN_PORT))
OUTPUT_DIR=$HOME/outputs/Taisu2/1th_recaption_1e-1M_train_lr_1e-6_epochs_1_max_subimg_num_12
# OUTPUT_DIR=$HOME/outputs/Taisu2/probe_2nodes
if [ ! -d $OUTPUT_DIR ]; then
    mkdir -p $OUTPUT_DIR
fi

OUTPUT_FILE=${OUTPUT_DIR}/output.log
OUTPUT_NAME=`echo ${OUTPUT_DIR} | rev | cut -d"/" -f1-1 | rev`

echo "multi-node deepspeed host file: ${HOST_FILE}" 2>&1 | tee $OUTPUT_FILE
echo "total multi-node number: ${NNODES}" 2>&1 | tee --append $OUTPUT_FILE
echo "current node rank index: ${NODE_RANK}" 2>&1 | tee --append $OUTPUT_FILE
echo "master node address: ${MASTER_ADDR}" 2>&1 | tee --append $OUTPUT_FILE
echo "communication port of master node: ${MASTER_PORT}" 2>&1 | tee --append $OUTPUT_FILE
printf "\n"


source $HOME/.bashrc  # for bash 4.2 (CentOS 7)
source $HOME/depends/anaconda3/etc/profile.d/conda.sh  # for bash 5.1 (Ubuntu 22.04)
conda activate
cd $HOME/projects/Taisu2/taisu2/
# export WANDB_MODE=offline
start_time_stamp=$(date +%Y-%m-%d-%H:%M:%S)
echo "Begin Taisu2 image-alttext pairs recaption (model train) at ${start_time_stamp}" 2>&1 | tee --append ${OUTPUT_FILE}

deepspeed --hostfile ${HOST_FILE} --no_ssh --node_rank=${NODE_RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
          ./llava/train/train_mem.py --deepspeed ./scripts/zero3.json \
          --accelerator_config ./scripts/accelerator_cfg.json \
          --model_name_or_path $HOME/ckpts/InternVL3-2B \
          --version internvl3 \
          --freeze_backbone false \
          --tune_mm_mlp_adapter false \
          --tune_vit_pos_embedding true \
          --tune_vision_tower true \
          --vision_tower "" \
          --mm_vision_select_layer -1 \
          --pretrain_mm_mlp_adapter "" \
          --mm_projector_type linear \
          --mm_use_im_start_end false \
          --mm_use_im_patch_token false \
          --mm_vision_select_feature patch \
          --data_path "" \
          --lazy_preprocess true \
          --is_multimodal true \
          --image_folder "" \
          --image_aspect_ratio pad \
          --image_grid_pinpoints "" \
          --dynamic_resolution true \
          --base_img_size 448 \
          --min_subimg_num 1 \
          --max_subimg_num 12 \
          --use_thumbnail true \
          --padding do_not_pad \
          --padding_side right \
          --return_tensors "" \
          --return_attention_mask false \
          --wds_shards_folder image-alttext-total-0.10M-at-2025-04-16-18:34:43 \
          --wds_shards_subfolder rename_and_rearchive 1th_recaption \
          --wds_nsamples_per_epoch 112021 \
          --wds_last_batch true \
          --wds_shuffle_seed 42 \
          --wds_worker_drop_last false \
          --txts_separator "\n" \
          --per_device_train_batch_size 2 \
          --gradient_accumulation_steps 2 \
          --num_train_epochs 1.0 \
          --max_steps -1 \
          --lr_scheduler_type cosine \
          --learning_rate 1e-6 \
          --weight_decay 0.1 \
          --warmup_ratio 0.1 \
          --output_dir ${OUTPUT_DIR} \
          --cache_dir "" \
          --wandb_project "Taisu2" \
          --run_name ${OUTPUT_NAME} \
          --bf16 true \
          --tf32 true \
          --save_total_limit 1 \
          --save_steps 3000 \
          --report_to "wandb" \
          --logging_steps 1 \
          --optim adamw_torch \
          --adam_beta1 0.9 \
          --adam_beta2 0.999 \
          --adam_epsilon 1e-8 \
          --max_grad_norm 1 \
          --dataloader_pin_memory true \
          --dataloader_drop_last false \
          --dataloader_num_workers 8 \
          --dataloader_persistent_worker true \
          --remove_unused_columns false \
          --freeze_mm_mlp_adapter false \
          --freeze_llm false \
          --mpt_attn_impl trition \
          --model_max_length 12288 \
          --double_quant true \
          --quant_type nf4 \
          --bits 16 \
          --lora_enable false \
          --lora_r 64 \
          --lora_alpha 16 \
          --lora_dropout 0.05 \
          --lora_weight_path "" \
          --lora_bias none \
          --group_by_modality_length false 2>&1 | tee --append ${OUTPUT_FILE}

end_time_stamp=`date +%Y-%m-%d-%H:%M:%S`
echo "End Taisu2 image-alttext pairs recaption (model train) at ${end_time_stamp}" 2>&1 | tee --append ${OUTPUT_FILE}
