#!/bin/bash

. /vol/ep/mm/venv/bin/activate
cd /vol/ep/mm/anomaly_detection/Medical-Image-Anomaly-Detection

train_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-2000-2999-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-3000-3999-clean/lookup_table.csv"
test_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7200-7299-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7300-7399-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-200-299-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-300-399-clean/lookup_table.csv"

python \
./Panda/panda.py \
--dataset=rsna3D \
--model=resnet \
--epochs=0 \
--lr 1e-2 \
--diag_path "/vol/ep/mm/anomaly_detection/data/fisher_diagnol.pth" \
--batch_size 5 \
--n_neighbours 2 \
--n_unfrozen_layers 3 \
--train_norm \
--train_lookup_table=$train_path \
--test_lookup_table=$test_path \
--results_output_dir_name="16_frame_2k_2000_4000_7200_7400_200_400_train_2_n_neighbours" \
