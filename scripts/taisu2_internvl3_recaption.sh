#!/usr/bin/bash


# turn off the execution trace mode
set +x

HOST_FILE=${1:-"$HOME/projects/Taisu2/scripts_taisu2/multinode_hostfile"}
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
          --model_name_or_path $HOME/ckpts/InternVL3-2B \
          --version internvl3 \
          --freeze_backbone False \
          --tune_mm_mlp_adapter False \
          --tune_vit_pos_embedding True \
          --vision_tower None \
          --mm_vision_select_layer -1 \
          --pretrain_mm_mlp_adapter None \
          --mm_projector_type linear \
          --mm_use_im_start_end False \
          --mm_use_im_patch_token False \
          --mm_vision_select_feature patch \
          --data_path None \
          --lazy_preprocess True \
          --is_multimodal True \
          --image_folder None \
          --image_aspect_ratio pad \
          --image_grid_pinpoints None \
          --dynamic_resolution True \
          --base_img_size 448 \
          --min_subimg_num 1 \
          --max_subimg_num 12 \
          --use_thumbnail True \
          --padding False \
          --padding_side right \
          --return_tensors None \
          --return_attention_mask False \
          --wds_shards_folder image-alttext-total-8.00M-at-2025-04-11-19:42:01 \
          --wds_shards_subfolder rename_and_rearchive \
          --wds_nsamples_per_epoch None \
          --wds_last_batch True \
          --wds_shuffle_seed 42 \
          --txts_separator \\n \
          --cache_dir None \
          --optim adamw_torch \
          --remove_unused_columns False \
          --freeze_mm_mlp_adapter False \
          --freeze_llm False \
          --mpt_attn_impl trition \
          --model_max_length 12288 \
          --double_quant True \
          --quant_type nf4 \
          --bits 16 \
          --lora_enable False \
          --lora_r 64 \
          --lora_alpha 16 \
          --lora_dropout 0.05 \
          --lora_weight_path None \
          --lora_bias none \
          --group_by_modality_length False \

end_time_stamp=`date +"%Y-%m-%d-%H:%M:%S"`
echo "End Taisu2 image-alttext pairs recaption train at ${end_time_stamp}"
