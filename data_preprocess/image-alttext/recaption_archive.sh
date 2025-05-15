#!/usr/bin/bash

source $HOME/.bashrc
cd $HOME/projects/Taisu2_self/data_preprocess/image-alttext/
conda activate lxb39

# command line parameters
taisu2_base_folder=${1:-"image-alttext-total-8.00M-at-2025-04-11-19:42:01"}
taisu2_specific_folder=${2:-"rename_and_rearchive"}
taisu2_recap_folder=${3:-"recaption_iter1"}
tarread_workers=${4:-"80"}
max_workers=${5:-"80"}
logging_level=${6:-"10"}

# execute recaptions archiving python3 script
python recaption_archive.py --taisu2-base-folder ${taisu2_base_folder} \
                            --taisu2-specific-folder ${taisu2_specific_folder} \
                            --taisu2-recap-folder ${taisu2_recap_folder} \
                            --tarread-workers ${tarread_workers} \
                            --max-workers ${max_workers} \
                            --logging-level ${logging_level}
