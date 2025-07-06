#!/usr/bin/bash


source $HOME/.bashrc
source $HOME/depends/anaconda3/etc/profile.d/conda.sh
conda activate
cd $HOME/projects/Taisu2/vlmevalkit/

DATA=${1:-"MME-RealWorld-CN"}
MODEL_PATH=${2:-"InternVL3-2B"}
CONFIG_FILE=${3:-"${HOME}/projects/Taisu2/taisu2/scripts/vlmevalkit_cfg.json"}
if [ -n ${CONFIG_FILE} ] && [ ! -e ${CONFIG_FILE} ]; then
    echo "specified evaluation config file for VLMEvalKit does not exist: ${CONFIG_FILE}, hence exit abnormally"
    exit 1
fi

VLMEVAL_WORK_DIR=${HOME}/outputs/Taisu2/evaluation
if [ ! -e ${VLMEVAL_WORK_DIR} ]; then
    mkdir -p ${VLMEVAL_WORK_DIR}
fi

VLMEVAL_MODE=${4:-"all"}
VLMEVAL_VERBOSE=${5:-"true"}

NNODES=${6:-"1"}
MASTER_ADDR=${7:-"127.0.0.1"}
MASTER_PORT=${8:-"32233"}
NPROC_PER_NODE=${9:-"1"}
NODE_RANK=${10:-"0"}
DEVICES=${11:-"7"}

export CUDA_VISIBLE_DEVICES=${DEVICES}
EVAL_CMD=(
    torchrun --nnodes ${NNODES}
    --node-rank ${NODE_RANK}
    --master-addr ${MASTER_ADDR}
    --master-port ${MASTER_PORT}
    --nproc-per-node ${NPROC_PER_NODE} run.py
    --work-dir ${VLMEVAL_WORK_DIR}
    --mode ${VLMEVAL_MODE}
)

if [ -z "${CONFIG_FILE}" ]; then
    EVAL_CMD+=("--data" "${DATA}" "--model" "${MODEL_PATH}")
else
    EVAL_CMD+=("--config" "${CONFIG_FILE}")
fi

if [ "${VLMEVAL_VERBOSE}" = "true" ]; then
    EVAL_CMD+=("--verbose")
fi

echo "start evaluation using VLMEvalKit at $(date '+%Y-%m-%d-%H:%M:%S')" | tee ${VLMEVAL_WORK_DIR}/output.log
"${EVAL_CMD[@]}" 2>&1 | tee --append ${VLMEVAL_WORK_DIR}/output.log
echo "end evaluation using VLMEvalKit at $(date '+%Y-%m-%d-%H:%M:%S')" | tee --append ${VLMEVAL_WORK_DIR}/output.log

