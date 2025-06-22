#!/usr/bin/bash


source $HOME/.bashrc
cd $HOME/projects/Taisu2/data_preprocess/image-alttext/
conda activate xiaobao12

raw_data_folder=${1:-"image-alttext-total-8.00M-at-2025-04-11-19:42:01"}
hash_method=${2:-"PHash"}
num_hash_workers=${3:-"90"}
verbose=${4:-"False"}
group_dupsfind=${5:-"True"}
outer_loop_iter=${6:-"30"}
shuffle_interval=${7:-"5"}
num_grp_inner_iter=${8:-"23"}
num_imgs_per_grp=${9:-"50000"}
max_distance_threshold=${10:-"5"}
scores=${11:-"False"}
search_method=${12:-"brute_force_cython"}
num_tarread_workers=${13:-"90"}
num_dist_workers=${14:-"90"}
group_dist_workers=${15:-"4"}
num_dedup_workers=${16:-"90"}

python images_hash_and_dedup.py --raw-data-folder ${raw_data_folder} \
                                --hash-method ${hash_method} \
                                --num-hash-workers ${num_hash_workers} \
                                --verbose ${verbose} \
                                --group-dupsfind ${group_dupsfind} \
                                --outer-loop-iter ${outer_loop_iter} \
                                --shuffle-interval ${shuffle_interval} \
                                --num-grp-inner-iter ${num_grp_inner_iter} \
                                --num-imgs-per-grp ${num_imgs_per_grp} \
                                --max-distance-threshold ${max_distance_threshold} \
                                --scores ${scores} \
                                --search-method ${search_method} \
                                --num-tarread-workers ${num_tarread_workers} \
                                --num-dist-workers ${num_dist_workers} \
                                --group-dist-workers ${group_dist_workers} \
                                --num-dedup-workers ${num_dedup_workers}
