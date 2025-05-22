#!/usr/bin/bash


# turn off the execution trace mode
set +x

HOST_FILE=${1:-"$HOME/projects/Taisu2/scripts/multinode_hostfile"}
MASTER_ADDR=${2:-$(cat $HOST_FILE | head -n 1 | cut -d" " -f 1)}
NNODES=$(cat $HOST_FILE | wc -l)
NODE_RANK=${3:-0}
MIN_PORT=${4:-23333}
MAX_PORT=${5:-45678}
PORT_RANGE=$((MAX_PORT - MIN_PORT + 1))
MASTER_PORT=$((RANDOM % PORT_RANGE + MIN_PORT))

echo "multi-node deepspeed host file: ${HOST_FILE}"
echo "total multi-node number: ${NNODES}"
echo "current node rank index: ${NODE_RANK}"
echo "master node address: ${MASTER_ADDR}"
echo "communication port of master node: ${MASTER_PORT}"
printf "\n"


source $HOME/.bashrc
conda activate xiaobao12
cd $HOME/projects/Taisu2/
start_time_stamp=$(date +%Y-%m-%d-%H:%M:%S)
echo "Begin Taisu2 image-alttext pairs recaption train at ${start_time_stamp}"

deepspeed --hostfile=${HOST_FILE} --no_ssh --node_rank=${NODE_RANK} \
          --master_addr=${MASTER_ADDR} --master_port=${MASTER_PORT} \
          ./taisu2/llava/train/train_mem.py --deepspeed ./scripts/zero3.json \
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
          --wds_shards_folder image-alttext-total-8.00M-at-2025-04-11-19:42:01 \
          --wds_shards_subfolder rename_and_rearchive \
          --wds_nsamples_per_epoch 5856804 \
          --wds_last_batch true \
          --wds_shuffle_seed 42 \
          --txts_separator "\n" \
          --per_device_train_batch_size 4 \
          --gradient_accumulation_steps 1 \
          --num_train_epochs 1.0 \
          --max_steps -1 \
          --lr_scheduler_type linear \
          --learning_rate 5e-5 \
          --weight_decay 0 \
          --warmup_ratio 0.0 \
          --output_dir "$HOME/outputs/Taisu2/debug" \
          --cache_dir "" \
          --wandb_project "Taisu2" \
          --run_name "debug" \
          --bf16 true \
          --tf32 true \
          --save_total_limit 1 \
          --save_steps 500 \
          --report_to "wandb" \
          --logging_steps 1 \
          --optim adamw_torch \
          --adam_beta1 0.9 \
          --adam_beta2 0.999 \
          --adam_epsilon 1e-8 \
          --max_grad_norm 1 \
          --dataloader_pin_memory true \
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
          --group_by_modality_length false \

end_time_stamp=`date +%Y-%m-%d-%H:%M:%S`
echo "End Taisu2 image-alttext pairs recaption train at ${end_time_stamp}"
