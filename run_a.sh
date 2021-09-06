#!/bin/bash

. /vol/ep/mm/venv/bin/activate
cd /vol/ep/mm/anomaly_detection/Medical-Image-Anomaly-Detection

train_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-0-999-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-1000-1999-clean/lookup_table.csv"
test_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7000-7099-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7100-7199-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-0-99-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-100-199-clean/lookup_table.csv"

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
--results_output_dir_name="16_frame_2k_0_200_train_2_n_neighbours" \
