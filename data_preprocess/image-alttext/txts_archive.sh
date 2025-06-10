#!/usr/bin/bash


source $HOME/.bashrc
cd $HOME/projects/Taisu2/data_preprocess/image-alttext/
conda activate xiaobao12

python txt_archive.py --base-data-folder image-alttext-total-1.00M-at-2025-04-16-21:10:39 \
                      --specific-data-folder rename_and_rearchive \
                      --recaption-idx 1 \
                      --num-cnt-proc 64 \
                      --num-archive-proc 64 \
                      --data-num-per-tar 10000 \
                      --logging-level 10
