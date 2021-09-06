#!/bin/bash

. /vol/ep/mm/rotnet-venv/bin/activate
cd /vol/ep/mm/anomaly_detection/Medical-Image-Anomaly-Detection/rotnet/ss-ood-master/

train_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-0-999-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-1000-1999-clean/lookup_table.csv"
test_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7000-7099-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7100-7199-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-0-99-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-100-199-clean/lookup_table.csv"
anomal_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-0-99-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-100-199-clean/lookup_table.csv"
normal_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7000-7099-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7100-7199-clean/lookup_table.csv"
experiment_name="2k_5_normalized"

python ./train.py --epochs=5 --train_set=1 --train_lookup_table=$train_path --test_lookup_table=$test_path --experiment_name=$experiment_name
python ./test.py --train_set=1 --anomal_lookup_table=$anomal_path --normal_lookup_table=$normal_path --experiment_name=$experiment_name