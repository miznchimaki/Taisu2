#!/usr/bin/bash

source $HOME/.bashrc
cd $HOME/projects/Taisu2_self/data_preprocess/image-alttext
conda activate lxb39

taisu2_base_folder=${1:-"image-alttext-total-8.00M-at-2025-04-11-19:42:01"}
taisu2_specific_folder=${2:-"images_pixel_filter_336x336_split_1_12 image-text-pairs"}
taisu2_output_folder=${3:-"rename_and_rearchive"}
max_workers_for_data_num=${4:-"90"}
data_num_per_tar=${5:-"10000"}
max_workers=${6:-"90"}
logging_level=${7:-"10"}

python rename_and_rearchive.py --taisu2-base-folder ${taisu2_base_folder} \
                               --taisu2-specific-folder ${taisu2_specific_folder} \
                               --taisu2-output-folder ${taisu2_output_folder} \
                               --max-workers-for-data-num ${max_workers_for_data_num} \
                               --data-num-per-tar ${data_num_per_tar} \
                               --max-workers ${max_workers} \
                               --logging-level ${logging_level}
