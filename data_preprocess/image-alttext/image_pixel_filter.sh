#!/usr/bin/bash


source $HOME/.bashrc
cd $HOME/projects/Taisu2/data_preprocess/image-alttext/
conda activate xiaobao12

raw_data_folder=${1:-"image-alttext-total-8.00M-at-2025-04-11-19:42:01"}
specific_data_folder=${2:-"images_group_dedup image-text-pairs"}
filter_res_folder=${3:-"images_pixel_filter"}
visual_input_size=${4:-"336"}
min_img_split=${5:-"1"}
max_img_split=${6:-"12"}
max_workers=${7:-"90"}
logging_level=${8:-"10"}

python image_pixel_filter.py --raw-data-folder ${raw_data_folder} \
                             --specific-data-folder ${specific_data_folder} \
                             --filter-res-folder ${filter_res_folder} \
                             --visual-input-size ${visual_input_size} \
                             --min-img-split ${min_img_split} \
                             --max-img-split ${max_img_split} \
                             --max-workers ${max_workers} \
                             --logging-level ${logging_level}
