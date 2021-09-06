#!/bin/bash

. /vol/ep/mm/rotnet-venv/bin/activate
cd /vol/ep/mm/anomaly_detection/Medical-Image-Anomaly-Detection/rotnet/ss-ood-master/

train_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-2000-2999-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-train-3000-3999-clean/lookup_table.csv"
test_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7200-7299-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7300-7399-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-200-299-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-300-399-clean/lookup_table.csv"
anomal_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-200-299-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-anomal-300-399-clean/lookup_table.csv"
normal_path="/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7200-7299-clean/lookup_table.csv,/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/filtered-16-frame-middle-data-test-normal-7300-7399-clean/lookup_table.csv"
experiment_name="2k_5_normalized_2"

python ./train.py --epochs=5 --train_set=2 --train_lookup_table=$train_path --test_lookup_table=$test_path --experiment_name=$experiment_name
python ./test.py --train_set=2 --anomal_lookup_table=$anomal_path --normal_lookup_table=$normal_path --experiment_name=$experiment_name