# image-alttext randomly sampling execution script
#!/usr/bin/bash

raw_data_dir=${1:-"/home/yidongyi/ImageDataSets/bdbk_citiao_sougou/output"}
base_out_dir=${2:-"$HOME/datasets/Taisu2_datasets"}
total_sample_data_num=${3:-"5e6"}
tar_patterns=${4:-"{00000..15079}.tar"}
preserved_file_types=${5:-"tar json parquet"}
base_random_seed=${6:-None}
sample_num_per_proc=${7:-None}
sample_max_workers=${8:-8}
logging_level=${9:-DEBUG}

source $HOME/.bashrc
conda activate xiaobao12
cd $HOME/projects/Taisu2/data_preprocess/image-alttext/

start_time_stamp=$(date +%Y-%m-%d-%H:%M:%S)
echo "image-alttext randomly sampling begins at ${start_time_stamp}"

python image_alttext_sampling.py --raw-data-dir ${raw_data_dir} \
                                 --base-out-dir ${base_out_dir} \
                                 --total-sample-data-num ${total_sample_data_num} \
                                 --tar-patterns ${tar_patterns} \
                                 --preserved-file-types ${preserved_file_types} \
                                 --base-random-seed ${base_random_seed} \
                                 --sample-num-per-proc ${sample_num_per_proc} \
                                 --sample-max-workers ${sample_max_workers} \
                                 --logging-level ${logging_level}

end_time_stamp=$(date +%Y-%m-%d-%H:%M:%S)
echo "image-alttext randomly sampling ends at ${end_time_stamp}"
