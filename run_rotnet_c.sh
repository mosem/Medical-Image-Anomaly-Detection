#!/bin/bash

. /vol/ep/mm/rotnet-venv/bin/activate
cd /vol/ep/mm/anomaly_detection/Medical-Image-Anomaly-Detection/rotnet/ss-ood-master/

train_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-4000-4999-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-5000-5999-clean/lookup_table.csv"
test_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7400-7499-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7500-7599-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-400-499-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-500-599-clean/lookup_table.csv"
anomal_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-400-499-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-500-599-clean/lookup_table.csv"
normal_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7400-7499-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7500-7599-clean/lookup_table.csv"
experiment_name="2k_5_normalized"

python ./train.py --epochs=5 --train_set=3 --train_lookup_table=$train_path --test_lookup_table=$test_path --experiment_name=$experiment_name
python ./test.py --train_set=3 --anomal_lookup_table=$anomal_path --normal_lookup_table=$normal_path --experiment_name=$experiment_name