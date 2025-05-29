#!/usr/bin/bash


# turn off the execution trace mode
set +x

MIN_PORT=${1:-23333}
MAX_PORT=${2:-45678}
PORT_RANGE=$((MAX_PORT - MIN_PORT + 1))
MASTER_PORT=$((RANDOM % PORT_RANGE + MIN_PORT))
MODEL_NAME_OR_PATH=${3:-$HOME/ckpts/InternVL3-2B}

echo "communication port of master node: ${MASTER_PORT}"
printf "\n"

source $HOME/.bashrc
conda activate xiaobao12
cd $HOME/projects/Taisu2/taisu2/
start_time_stamp=$(date +%Y-%m-%d-%H:%M:%S)
echo "Begin Taisu2 image-alttext pairs recaption (inference) at ${start_time_stamp}"

deepspeed --include localhost:0,1,2,3 --master_port ${MASTER_PORT} \
          ./llava/inference/recaption.py --recaption-idx 1 \
                                         --conv-template-name internvl2_5 \
                                         --num-workers 8 \
                                         --batch-size 64 \
                                         --pin-memory True \
                                         --drop-last False \
                                         --use-fast False \
                                         --trust-remote-code False \
                                         --cache-dir None \
                                         --model-max-length 12288 \
                                         --padding do_not_pad \
                                         --padding-side left \
                                         --return-tensors None \
                                         --return-attention-mask None \
                                         --base-img-size 448 \
                                         --min-subimg-num 1 \
                                         --max-subimg-num 12 \
                                         --use-thumbnail True \
                                         --tars-folder image-alttext-total-8.00M-at-2025-04-11-19:42:01 \
                                         --tars-subfolder rename_and_rearchive \
                                         --total-samples 5856804 \
                                         --wds-shuffle-seed None \
                                         --model-name-or-path ${MODEL_NAME_OR_PATH} \
                                         --data-type bfloat16 \
                                         --mpt-attn-impl triton \
                                         --use-flash-attn True \
                                         --max-length None \
                                         --max-new-tokens 5000 \
                                         --min-length 0 \
                                         --do-sample False \
                                         --num-beams 3 \
                                         --temperature 1.0 \
                                         --top-k 50 \
                                         --top-p 1.0 \
                                         --repetition-penalty 1.0 \
                                         --length-penalty 1.0 \
                                         --num-return-sequences 1 \
                                         --output-attentions False \
                                         --output-hidden-states False \
                                         --output-scores False \
                                         --output-logits False \

end_time_stamp=`date +%Y-%m-%d-%H:%M:%S`
echo "End Taisu2 image-alttext pairs recaption (inference) at ${end_time_stamp}"
