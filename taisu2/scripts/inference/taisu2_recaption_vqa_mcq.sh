#!/usr/bin/bash


# turn off the execution trace mode
set +x

GPU_DEVICES=${1:-"0,1,2,3,4,5,6,7"}
MIN_PORT=${2:-23333}
MAX_PORT=${3:-45678}
RANDOM=${4:-42}
PORT_RANGE=$((MAX_PORT - MIN_PORT + 1))
MASTER_PORT=$((RANDOM % PORT_RANGE + MIN_PORT))
MODEL_NAME_OR_PATH=${5:-"${HOME}/ckpts/InternVL3-2B"}
DATA_TYPE_WEIGHT=${6:-"3 3 4"}  # recaption VQA MCQ
read -r -a DATA_TYPE_WEIGHT_ARR <<< "${DATA_TYPE_WEIGHT}"
CAPTION_DATA_WEIGHT=${DATA_TYPE_WEIGHT_ARR[0]}
VQA_DATA_WEIGHT=${DATA_TYPE_WEIGHT_ARR[1]}
MCQ_DATA_WEIGHT=${DATA_TYPE_WEIGHT_ARR[2]}
DATA_WEIGHT_RANDOM_SEED=${7:-"42"}
OUTPUT_FOLDER="recaption_${CAPTION_DATA_WEIGHT}_vqa_${VQA_DATA_WEIGHT}_mcq_${MCQ_DATA_WEIGHT}_random_seed_${DATA_WEIGHT_RANDOM_SEED}"

DATALOADER_WORKERS=${8:-"8"}
BATCH_SIZE=${9:-"128"}
BASE_IMG_SIZE=${10:-"448"}
MIN_SUBIMG_NUM=${11:-"1"}
MAX_SUBIMG_NUM=${12:-"9"}
TARS_FOLDER=${13:-"image-alttext-total-0.10M-at-2025-04-16-18:34:43"}
TARS_SUBFOLDER=${14:-"rename_and_rearchive"}
OUTPUT_DIR=${HOME}/datasets/Taisu2_datasets/${TARS_FOLDER}/${TARS_SUBFOLDER}/${OUTPUT_FOLDER}
if [ -d ${OUTPUT_DIR} ]; then
    rm --recursive --force ${OUTPUT_DIR}
fi
mkdir -p ${OUTPUT_DIR}

LOG_PATH=${OUTPUT_DIR}/"data_synthesis.log"
log_wrap(){
    "$@" 2>&1 | tee --append ${LOG_PATH}
}
TOTAL_SAMPLES=${15:-"112021"}

echo "start recaption, vqa, and mcq data synthesis at $(date '+%Y-%m-%d-%H:%M:%S')" 2>&1 | tee ${LOG_PATH}
log_wrap echo "communication port of master node: ${MASTER_PORT}"
log_wrap echo "data synthesis model: ${MODEL_NAME_OR_PATH}"
log_wrap echo "data weight for detailed caption: ${CAPTION_DATA_WEIGHT}; vqa: ${VQA_DATA_WEIGHT}; mcq: ${MCQ_DATA_WEIGHT}"
log_wrap echo "random seed for data synthesis: ${DATA_WEIGHT_RANDOM_SEED}"
log_wrap echo "batch size  per device: ${BATCH_SIZE}"
log_wrap echo "base image size: ${BASE_IMG_SIZE}"
log_wrap echo "minimum sub-image number of dynamic resolutaion strategy: ${MIN_SUBIMG_NUM}"
log_wrap echo "maximum sub-image number of dynamic resolutaion strategy: ${MAX_SUBIMG_NUM}"
log_wrap echo "image-text data folder: ${TARS_FOLDER}; sub-folder: ${TARS_SUBFOLDER}"
log_wrap printf "\n"

source $HOME/.bashrc
conda activate xiaobao12
cd $HOME/projects/Taisu2/taisu2/

data_synthesis_cmd=(
deepspeed --include localhost:${GPU_DEVICES} --master_port ${MASTER_PORT}
          ./llava/inference/recaption_vqa_mcq.py --data-type-weight ${DATA_TYPE_WEIGHT}
                                                 --random-seed ${DATA_WEIGHT_RANDOM_SEED}
                                                 --conv-template-name internvl2_5
                                                 --num-workers ${DATALOADER_WORKERS}
                                                 --batch-size ${BATCH_SIZE}
                                                 --pin-memory True
                                                 --drop-last False
                                                 --use-fast False
                                                 --trust-remote-code False
                                                 --cache-dir None
                                                 --model-max-length 12288
                                                 --padding do_not_pad
                                                 --padding-side left
                                                 --return-tensors None
                                                 --return-attention-mask None
                                                 --base-img-size ${BASE_IMG_SIZE}
                                                 --min-subimg-num ${MIN_SUBIMG_NUM}
                                                 --max-subimg-num ${MAX_SUBIMG_NUM}
                                                 --use-thumbnail True
                                                 --tars-folder ${TARS_FOLDER}
                                                 --tars-subfolder ${TARS_SUBFOLDER}
                                                 --total-samples ${TOTAL_SAMPLES}
                                                 --wds-shuffle-seed None
                                                 --model-name-or-path ${MODEL_NAME_OR_PATH}
                                                 --data-type bfloat16
                                                 --mpt-attn-impl triton
                                                 --use-flash-attn True
                                                 --max-length None
                                                 --max-new-tokens 1000
                                                 --min-length 0
                                                 --do-sample False
                                                 --num-beams 1
                                                 --temperature 1.0
                                                 --top-k 50
                                                 --top-p 1.0
                                                 --repetition-penalty 1.0
                                                 --length-penalty 1.0
                                                 --num-return-sequences 1
                                                 --return-dict-in-generate False
                                                 --output-attentions False
                                                 --output-hidden-states False
                                                 --output-scores False
                                                 --output-logits False
)
log_wrap "${data_synthesis_cmd[@]}"
log_wrap echo "end recaption, vqa, and mcq data synthesis at `date '+%Y-%m-%d-%H:%M:%S'`"
