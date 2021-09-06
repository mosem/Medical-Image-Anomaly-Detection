#!/bin/bash

. /vol/ep/mm/venv/bin/activate
cd /vol/ep/mm/anomaly_detection/Medical-Image-Anomaly-Detection

train_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-4000-4999-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-5000-5999-clean/lookup_table.csv"
test_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7400-7499-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7500-7599-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-400-499-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-500-599-clean/lookup_table.csv"

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
--results_output_dir_name="16_frame_2k_4000_6000_7400_7600_400_600_train_2_n_neighbours" \
