import argparse
from pathlib import Path
import pandas as pd
from pandas import read_csv
import os
import pickle
import concurrent.futures
import pydicom as dicom
import ast
from math import ceil
import shutil

from PIL import Image
import numpy as np
from itertools import islice

from skimage import measure, filters
from skimage.morphology import binary_dilation, disk, reconstruction

import matplotlib.pyplot as plt

pd.set_option('display.max_columns', None)


LABELS_MAP_FILEPATH = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/stage_2_train.csv'
OUTPUT_ROWDICT_FILEPATH = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/rowDict.pickle'
DATA_ROOT_DIR_PATH = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/stage_2_train'

PATIENT_TABLES_ROOT_DIR = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/patient_tables'
DATASET_ROOT_PATH = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001'
DATASET_PATH = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/16-frame-data.csv'
LOOKUP_TABLES_ROOT_DIR = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/lookup_tables_metadata'

OUTPUT_ROOT_DIR = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001'

METADATA_ATTRIBUTES_NAMES = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'ImagePositionPatient',
                             'ImageOrientationPatient']

#Create Metadata tables

def createDirLookupTable(child_dir_path, labels_map_file_path, rowDict, output_dir_path):
    print(f"starting {child_dir_path}")
    output_filepath = os.path.join(output_dir_path, f"{child_dir_path.name}_lookup_table.csv")
    if Path(output_filepath).is_file():
        print(f"{child_dir_path} lookup file already exists.")
        return
    df = read_csv(labels_map_file_path)
    lookupTable = pd.DataFrame(columns=df.columns)
    lookupTable['filepath'] = []
    addMetadataColumns(lookupTable)
    lookupTable.dropna(inplace=True)
    print(f"{child_dir_path}: starting enumeration")
    for i, file in enumerate(os.scandir(child_dir_path)):
        filename, file_extension = os.path.splitext(file.name)
        if filename not in rowDict:
            print(f"{child_dir_path.name}. KeyError: {filename} not in rowDict")
            continue
        row_indexes = rowDict[filename]
        temp = df.loc[row_indexes].copy(deep=True)
        filepath = os.path.abspath(file)
        addMetadata(temp, filepath)
        lookupTable = lookupTable.append(temp, ignore_index=True)
    lookupTable.to_csv(output_filepath)
    print(f"createDirLookupTable: Done {child_dir_path}.")


def addMetadataColumns(frame):
    for name in METADATA_ATTRIBUTES_NAMES:
        frame[name] = []

def addMetadata(frame, filepath):
    dicom_file = dicom.dcmread(filepath)
    frame['filepath'] = filepath
    for name in METADATA_ATTRIBUTES_NAMES:
        if name not in frame.columns:
            frame[name] = ""
        for i, row in frame.iterrows():
            frame.at[i,name] = dicom_file[name].value



def extractId(fullIdStr):
    return "_".join(fullIdStr.split("_", 2)[:2])


def extractType(fullIdStr):
    return fullIdStr.split("_",2)[-1]

def createRowDict(labels_map_file):
    df = read_csv(labels_map_file)
    rowDict = {}
    for index, row in df.iterrows():
        sample_type = extractType(row['ID'])
        if sample_type != 'any':
            continue
        id = extractId(row['ID'])
        if id not in rowDict:
            rowDict[id] = []
        rowDict[id].append(index)
    print('Done creating rowDict.')
    return rowDict


def createAndSaveRowDict(labels_map_file, output_file_path):
    rowDict = createRowDict(labels_map_file)
    filename = 'rowDict.pickle'
    with open(output_file_path, 'wb') as handle:
        pickle.dump(rowDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved row dict as {filename}")


def createLookupTables(root_dir_path, labels_map_file_path, rowDict_file_path, output_dir_path):
    with open(rowDict_file_path, 'rb') as handle:
        rowDict = pickle.load(handle)
    args = [(child_dir_path, labels_map_file_path, rowDict, output_dir_path) for child_dir_path in os.scandir(root_dir_path)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda p: createDirLookupTable(*p), args)
    print("createLookupTable: Done.")


def runCreateLookupTables(data_root_dir_path =DATA_ROOT_DIR_PATH,labels_map_filepath = LABELS_MAP_FILEPATH, output_rowdict_filepath = OUTPUT_ROWDICT_FILEPATH, lookup_tables_root_dir = LOOKUP_TABLES_ROOT_DIR):

    if not Path(output_rowdict_filepath).is_file():
        createAndSaveRowDict(labels_map_filepath, output_rowdict_filepath)

    if not Path(lookup_tables_root_dir).is_dir():
        os.mkdir(lookup_tables_root_dir)

    createLookupTables(data_root_dir_path, labels_map_filepath, output_rowdict_filepath, lookup_tables_root_dir)


# Save patient directories

def getPatientFrames(dir_path):
    metadata_paths = [path for path in os.scandir(dir_path) if path.name.endswith('.csv')]
    frames = [read_csv(metadata_table_path) for metadata_table_path in metadata_paths]
    full_metadata_frame = pd.concat(frames, sort=False)

    patients_frames = [x.reset_index(drop=True).drop("Unnamed: 0", axis=1) for _, x in
                       full_metadata_frame.groupby(['PatientID'])]
    return patients_frames

def savePatientDirectory(patient_frame, output_dir):
    patient_id = patient_frame['PatientID'][0]
    print(f"savePatientDirectory: Starting: {patient_id}")

    sorted = patient_frame.sort_values(by="ImagePositionPatient",
                                       key=lambda positions: [ast.literal_eval(l)[2] for l in positions]).reset_index(drop=True)
    patient_dir_path = os.path.join(output_dir, patient_id)
    if Path(patient_dir_path).is_dir():
        print(f"{patient_dir_path} exists.")
        return
    os.mkdir(patient_dir_path)

    series_frames = [x.reset_index(drop=True) for _, x in sorted.groupby(['SeriesInstanceUID'])]
    for series_frame in series_frames:
        frame_filename = series_frame['SeriesInstanceUID'][0] + '.csv'
        frame_path = os.path.join(patient_dir_path, frame_filename)
        if Path(frame_path).is_file():
            print(f"{frame_path} already exists.")
            continue
        series_frame.reset_index(drop=True).to_csv(frame_path)
    print(f"savePatientDirectories: done {patient_id}")

def savePatientDirectories(patients_frames, output_dir):
    if not Path(output_dir).is_dir():
        os.mkdir(output_dir)
    else:
        print(f"{output_dir} already exists.")

    args = [(patient_frame, output_dir) for patient_frame in patients_frames]
    # for arg in args:
    #     savePatientDirectory(*arg)
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda p: savePatientDirectory(*p), args)


def runSavePatientFrames(lookup_tables_root_dir=LOOKUP_TABLES_ROOT_DIR, patient_tables_root_dir=PATIENT_TABLES_ROOT_DIR):
    patient_frames = getPatientFrames(lookup_tables_root_dir)
    savePatientDirectories(patient_frames, patient_tables_root_dir)


# Create table for n-frame dataset

def createRow(table_frame_entry, n_frames, mode='middle'):
    print(f"Reading table frame: {table_frame_entry.path}")
    table_frame = read_csv(table_frame_entry.path)
    n_rows = len(table_frame.index)
    if n_rows < n_frames:
        print(f"{table_frame_entry.name}: Not enough rows. n_rows: {n_rows}, n_frames: {n_frames}./nExiting.")
        return None
    middle_row = n_rows // 2
    if mode == 'middle':
        start_row = middle_row - n_frames // 2
        end_row = middle_row + ceil(n_frames / 2) - 1
    elif mode == 'bottom':
        start_row = middle_row - n_frames
        end_row = middle_row - 1
    elif mode == 'top':
        start_row = middle_row
        end_row = middle_row + n_frames - 1
    normal = (table_frame.loc[start_row:end_row, ['Label']] == 0).all().bool()
    label = 0 if normal else 1
    indices = [i for i in range(start_row, end_row + 1)]
    print(f"Done creating row: {table_frame_entry.path}")
    return pd.Series({'path': table_frame_entry.path, 'label': label, 'indices': indices})


def createPatientData(patient_dir_entry, n_frames, mode = 'middle'):
    print(f"creating data for patient: {patient_dir_entry}")
    table_entries = [entry for entry in os.scandir(patient_dir_entry.path) if entry.name.endswith('.csv')]
    for i, table_entry in enumerate(table_entries):
        row = createRow(table_entry, n_frames, mode)
        if row is not None:
            return row  # create a single row for each patient
        elif i == len(table_entries) - 1:
            print(f"no rows were created for {table_entry.name}")
    print(f"no data created for {patient_dir_entry}")
    return None


def createDataset(root_dir, n_frames, n_samples=10000, mode = 'middle'):
    print('Creating dataset')
    patient_dirs_entries = [path for path in os.scandir(root_dir) if path.is_dir()]
    n_samples = min(n_samples, len(patient_dirs_entries))
    args = [(patient_dir_entry, n_frames, mode) for patient_dir_entry in patient_dirs_entries[:n_samples]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        rows = executor.map(lambda p: createPatientData(*p), args)

    print('Done creating dataset')
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows)


def runCreateDataset(dataset_path, n_frames = 8, mode = 'middle', patient_tables_root_dir = PATIENT_TABLES_ROOT_DIR):
    dataset_frame = createDataset(patient_tables_root_dir, n_frames, n_samples=10000, mode = mode)
    dataset_frame.to_csv(dataset_path)


# Create frames for datasets

def createOneClassFrame(dataset_frame, n_samples, start=0):
    normal_frame, _ = [x.reset_index(drop=True).drop("Unnamed: 0", axis=1) for _, x in dataset_frame.groupby(['label'])]
    return normal_frame[start:start + n_samples]


def createBalancedClassFrame(dataset_frame, n_samples, start_normal=0, start_anomal=0):
    normal_frame, anomal_frame = [x.reset_index(drop=True).drop("Unnamed: 0", axis=1) for _, x in dataset_frame.groupby(['label'])]

    normal_frame = normal_frame[start_normal:start_normal+n_samples//2]
    anomal_frame = anomal_frame[start_anomal:start_anomal+n_samples//2]
    frame = pd.concat([normal_frame, anomal_frame], ignore_index=True)
    return frame


def createFramesForDatasets(dataset_path, n_datasets=5, n_train_samples=1000, n_test_samples=200, output_name='data',output_root_dir=OUTPUT_ROOT_DIR):
    dataset_frame = read_csv(dataset_path)
    # train_frame_dummy = createOneClassFrame(dataset_frame, 10, 1000)
    # test_frame_dummy = createBalancedClassFrame(dataset_frame, 10, 0, 0)
    # train_frame_dummy.to_csv(os.path.join(OUTPUT_ROOT_DIR, '16-frame-data-train-dummy.csv'), index=False)
    # test_frame_dummy.to_csv(os.path.join(OUTPUT_ROOT_DIR, '16-frame-data-test-dummy.csv'), index=False)
    paths = []
    for i in range(n_datasets):
        train_start = n_test_samples * n_datasets + n_train_samples * i
        train_end = train_start + n_train_samples - 1
        train_frame = createOneClassFrame(dataset_frame, n_train_samples, train_start)
        train_name = '-'.join([output_name, 'train', str(train_start), str(train_end)]) + '.csv'
        train_path = os.path.join(output_root_dir, train_name)
        train_frame.to_csv(train_path, index=False)
        print(f'done {train_name}')

        test_start = i*n_test_samples//2
        test_end = test_start + n_test_samples//2
        test_frame = createBalancedClassFrame(dataset_frame, n_test_samples, test_start, test_start)
        test_name = '-'.join([output_name, 'test', str(test_start), str(test_end)]) + '.csv'
        test_path = os.path.join(output_root_dir, test_name)
        test_frame.to_csv(test_path, index=False)
        print(f'done {test_name}')

        val_start = n_test_samples//2*n_datasets + i*n_test_samples//2
        val_end = val_start + n_test_samples//2
        val_frame = createBalancedClassFrame(dataset_frame, n_test_samples, val_start, val_start)
        val_name = '-'.join([output_name, 'val', str(val_start), str(val_end)]) + '.csv'
        val_path = os.path.join(output_root_dir, val_name)
        val_frame.to_csv(val_path, index=False)

        paths.append(train_path)
        paths.append(test_path)
        paths.append(val_path)
        print(f'done {val_name}')





    # train_frame_a = createOneClassFrame(dataset_frame, 1000, 5000)
    # train_frame_b = createOneClassFrame(dataset_frame, 1000, 1000)
    # train_frame_c = createOneClassFrame(dataset_frame, 1000, 2000)
    # train_frame_d = createOneClassFrame(dataset_frame, 1000, 3000)
    # train_frame_e = createOneClassFrame(dataset_frame, 1000, 4000)
    # test_frame_a = createBalancedClassFrame(dataset_frame, 200,0,0)
    # test_frame_b = createBalancedClassFrame(dataset_frame, 200,100,100)
    # test_frame_c = createBalancedClassFrame(dataset_frame, 200,200,200)
    # test_frame_d = createBalancedClassFrame(dataset_frame, 200,300,300)
    # test_frame_e = createBalancedClassFrame(dataset_frame, 200,400,400)
    # val_frame_a = createBalancedClassFrame(dataset_frame, 200,500,500)
    # val_frame_b = createBalancedClassFrame(dataset_frame, 200,600,600)
    # val_frame_c = createBalancedClassFrame(dataset_frame, 200,700,700)
    # val_frame_d = createBalancedClassFrame(dataset_frame, 200,800,800)
    # val_frame_e = createBalancedClassFrame(dataset_frame, 200,900,900)
    #
    # train_frame_a.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-1.csv'), index=False)
    # train_frame_b.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-2.csv'), index=False)
    # train_frame_c.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-3.csv'), index=False)
    # train_frame_d.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-4.csv'), index=False)
    # train_frame_e.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-5.csv'), index=False)
    #
    # test_frame_a.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-1.csv'), index=False)
    # test_frame_b.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-2.csv'), index=False)
    # test_frame_c.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-3.csv'), index=False)
    # test_frame_d.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-4.csv'), index=False)
    # test_frame_e.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-5.csv'), index=False)
    #
    # val_frame_a.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-1.csv'), index=False)
    # val_frame_b.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-2.csv'), index=False)
    # val_frame_c.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-3.csv'), index=False)
    # val_frame_d.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-4.csv'), index=False)
    # val_frame_e.to_csv(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-5.csv'), index=False)
    print('done.')
    return paths



# Save Dataset Images to image/dicom files


def window_image(pixel_array, window_center, window_width, is_normalize=True):
    image_min = window_center - window_width // 2
    image_max = window_center + window_width // 2
    image = np.clip(pixel_array, image_min, image_max)

    if is_normalize:
        image = (image - image_min) / (image_max - image_min)
    return image


def fill_mask(mask):
    seed = np.ones_like(mask)
    h,w  = seed.shape
    seed[0,0] = 0 if not mask[0,0] else 1
    seed[h-1,0] = 0 if not mask[h-1,0] else 1
    seed[h-1,w-1] = 0 if not mask[h-1,w-1] else 1
    seed[0,w-1] = 0 if not mask[0,w-1] else 1

    filled  = reconstruction(seed, mask.copy(), method='erosion')

    return filled


def get_largest_connected_components(image, n_components=1):
    labels, num_of_cc = measure.label(image, connectivity=2, return_num=True)

    background_label = labels[0, 0]

    unique, counts = np.unique(labels, return_counts=True)
    mask = unique != background_label
    counts = counts[mask]
    unique = unique[mask]
    sorted_indices = np.argsort(counts)
    largest_component_values = unique[sorted_indices[-n_components:]]

    single_component_data_mask = np.isin(labels, largest_component_values)
    single_component_data = np.zeros_like(image)
    single_component_data[single_component_data_mask] = 1

    return single_component_data


def get_mask(pixel_array):
    DICOM_THRESHOLD = -10
    MASK_THRESHOLD = 0.2

    image = np.array(pixel_array > DICOM_THRESHOLD, dtype = np.float64)

    image = (image - np.min(image)) / (np.max(image) - np.min(image))

    image = filters.gaussian(image, sigma=0.2)

    image = image > MASK_THRESHOLD

    largest_component_mask = get_largest_connected_components(image)

    mask = binary_dilation(largest_component_mask, disk(10))

    return fill_mask(mask)


def normalize_dicom(dicom_image, mask_flag):
    if (dicom_image.BitsStored == 12) and (dicom_image.PixelRepresentation == 0) and (int(dicom_image.RescaleIntercept) > -100):
        # see: https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        p = dicom_image.pixel_array + 1000
        p[p >= 4096] = p[p >= 4096] - 4096
        dicom_image.PixelData = p.tobytes()
        dicom_image.RescaleIntercept = -1000

    pixel_array = dicom_image.pixel_array * dicom_image.RescaleSlope + dicom_image.RescaleIntercept
    brain = window_image(pixel_array, 40, 80)
    subdural = window_image(pixel_array, 80, 200)
    soft_tissue = window_image(pixel_array, 40, 380)

    image_array = np.dstack([soft_tissue, subdural, brain])
    image_array = (image_array * 255).astype(np.uint8)

    if mask_flag:
        mask = get_mask(pixel_array)
        mask = np.dstack([mask, mask, mask])
        image_array = np.where(mask > 0, image_array, 0)

    image = Image.fromarray(image_array)
    return image


def saveImages(item_table_path, indices, output_dir_row, convert_to_png, mask_flag):
    if not Path(output_dir_row).is_dir():
        print(f"making dir {output_dir_row}")
        os.mkdir(output_dir_row)
    table = read_csv(item_table_path)
    samples_output_paths = []
    for idx in indices:
        filename = f"slice_{str(idx)}"
        if convert_to_png:
            output_path = os.path.join(output_dir_row, filename + '.png')
        else:
            output_path = os.path.join(output_dir_row, filename + '.dcm')
        if Path(output_path).is_file():
            print(f'{output_path}: already exists.')
            samples_output_paths.append(output_path)
            continue
        else:
            print(f'{output_path}: processing.')

        img_path = table.loc[idx, 'filepath']
        dicom_image = dicom.dcmread(img_path)

        if convert_to_png:
            normalized_image = normalize_dicom(dicom_image, mask_flag)
            normalized_image.save(output_path)
        else:
            dicom_image.save_as(output_path)

        samples_output_paths.append(output_path)
    return samples_output_paths


def saveRow(input_frame, row_idx, output_dir_path, convert_to_png, mask_flag):
    output_frame = pd.DataFrame(columns=['ID', 'label', 'filepaths'])
    csv_filepath = input_frame.at[row_idx, 'path']
    label = input_frame.at[row_idx, 'label']
    row_path, row_basename = os.path.split(csv_filepath)
    output_frame.at[row_idx, 'ID'] = os.path.splitext(row_basename)[0]
    output_frame.at[row_idx, 'label'] = label
    output_dir_row = os.path.join(output_dir_path, os.path.splitext(row_basename)[0])
    indices = ast.literal_eval(input_frame.loc[row_idx, 'indices'])
    output_frame.at[row_idx, 'filepaths'] = saveImages(csv_filepath, indices, output_dir_row, convert_to_png, mask_flag)
    return output_frame


def saveDataset(table_path, output_dir_name=None, convert_to_png=True, mask_flag = True, start=0, n_samples=1000):
    dataset_frame = read_csv(table_path)
    root_dir, table_basename = os.path.split(table_path)
    print(f"saving from table {table_basename}, samples: {start}-{start + n_samples}")
    if output_dir_name is None:
        output_dir_name = os.path.splitext(table_basename)[0]
        if not convert_to_png:
            output_dir_name += '-dcm'
    output_dir_path = os.path.join(root_dir, output_dir_name)
    # if Path(output_dir_path).is_dir():
    #     print(f'{output_dir_path} already exists. Removing directory.')
    #     # return
    #     shutil.rmtree(output_dir_path)
    if not Path(output_dir_path).is_dir():
        print(f"making dir {output_dir_path}")
        os.mkdir(output_dir_path)

    end = min(start + n_samples, len(dataset_frame.index))

    # each row in dataset_frame is a new sample

    # args = [(dataset_frame,i,output_dir_path, convert_to_png, mask_flag) for i, _ in islice(dataset_frame.iterrows(), start, end)]
    # with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    #       results = executor.map(lambda p: saveRow(*p), args)
    #       samples_frame = pd.concat([frame for frame in results], ignore_index=True)
    # samples_frame.to_csv(os.path.join(output_dir_path, 'lookup_table.csv'))

    samples_frame = pd.DataFrame(columns=['ID', 'label', 'filepaths'])

    for i, _ in islice(dataset_frame.iterrows(), start, end):
        csv_filepath = dataset_frame.at[i, 'path']
        label = dataset_frame.at[i, 'label']
        row_path, row_basename = os.path.split(csv_filepath)
        samples_frame.at[i, 'ID'] = os.path.splitext(row_basename)[0]
        samples_frame.at[i, 'label'] = label
        output_dir_row = os.path.join(output_dir_path, os.path.splitext(row_basename)[0])
        indices = ast.literal_eval(dataset_frame.loc[i, 'indices'])
        samples_frame.at[i, 'filepaths'] = saveImages(csv_filepath, indices, output_dir_row, convert_to_png, mask_flag)

    samples_frame.to_csv(os.path.join(output_dir_path, 'lookup_table.csv'))

    print(f'done saving to {output_dir_name}')


def saveDatasets(table_paths):
    for path in table_paths:
        head, tail = os.path.split(path)
        table_name = tail.split('.')[0]
        saveDataset(path, '-'.join([table_name, 'dcm']), False)  # saving as dicom files
        saveDataset(path, '-'.join([table_name, 'clean']), True)  # saving as png files

    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '16-frame-data-train-dummy.csv'),
    #             '16-frame-data-train-dummy-dcm', False) # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '16-frame-data-train-dummy.csv'),
    #             '16-frame-data-train-dummy-clean', True)  # saving as png files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '16-frame-data-test-dummy.csv'),
    #             '16-frame-data-test-dummy-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '16-frame-data-test-dummy.csv'),
    #             '16-frame-data-test-dummy-clean', True)  # saving as png files

    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-1.csv'),
    #             '8-frame-data-train-1-dcm', False) # saving as dicom files

    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-1.csv'),
    #             '8-frame-data-train-1-clean', True)  # saving as png files

    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-1.csv'),
    #             '8-frame-data-test-1-dcm', False)  # saving as dicom files

    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-1.csv'),
    #             '8-frame-data-test-1-clean', True)  # saving as png files

    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-1.csv'),
    #             '8-frame-data-val-1-dcm', False)  # saving as dicom files

    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-1.csv'),
    #             '8-frame-data-val-1-clean', True)  # saving as png files

    #######

    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-2.csv'),
    #             '8-frame-data-train-2-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-2.csv'),
    #             '8-frame-data-train-2-clean', True)  # saving as png files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-2.csv'),
    #             '8-frame-data-test-2-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-2.csv'),
    #             '8-frame-data-test-2-clean', True)  # saving as png files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-2.csv'),
    #             '8-frame-data-val-2-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-2.csv'),
    #             '8-frame-data-val-2-clean', True)  # saving as png files

    #######

    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-3.csv'),
    #             '8-frame-data-train-3-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-3.csv'),
    #             '8-frame-data-train-3-clean', True)  # saving as png files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-3.csv'),
    #             '8-frame-data-test-3-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-3.csv'),
    #             '8-frame-data-test-3-clean', True)  # saving as png files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-3.csv'),
    #             '8-frame-data-val-3-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-3.csv'),
    #             '8-frame-data-val-3-clean', True)  # saving as png files
    #
    # ######
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-4.csv'),
    #             '8-frame-data-train-4-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-4.csv'),
    #             '8-frame-data-train-4-clean', True)  # saving as png files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-4.csv'),
    #             '8-frame-data-test-4-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-4.csv'),
    #             '8-frame-data-test-4-clean', True)  # saving as png files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-4.csv'),
    #             '8-frame-data-val-4-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-4.csv'),
    #             '8-frame-data-val-4-clean', True)  # saving as png files
    #
    # #####
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-5.csv'),
    #             '8-frame-data-train-5-dcm', False)  # saving as dicom files
    # #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-train-5.csv'),
    #             '8-frame-data-train-5-clean', True)  # saving as png files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-5.csv'),
    #             '8-frame-data-test-5-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-test-5.csv'),
    #             '8-frame-data-test-5-clean', True)  # saving as png files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-5.csv'),
    #             '8-frame-data-val-5-dcm', False)  # saving as dicom files
    #
    # saveDataset(os.path.join(OUTPUT_ROOT_DIR, '8-frame-data-val-5.csv'),
    #             '8-frame-data-val-5-clean', True)  # saving as png files


def test_mask(filepath):
    dicom_image = dicom.dcmread(filepath)
    if (dicom_image.BitsStored == 12) and (dicom_image.PixelRepresentation == 0) and (
            int(dicom_image.RescaleIntercept) > -100):
        # see: https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        p = dicom_image.pixel_array + 1000
        p[p >= 4096] = p[p >= 4096] - 4096
        dicom_image.PixelData = p.tobytes()
        dicom_image.RescaleIntercept = -1000

    pixel_array = dicom_image.pixel_array * dicom_image.RescaleSlope + dicom_image.RescaleIntercept

    soft_tissue = window_image(pixel_array, 40, 380)

    image_array = (soft_tissue * 255).astype(np.uint8)
    image = Image.fromarray(image_array)

    mask = get_mask(pixel_array)

    masked_image = np.where(mask>0, image, 0)


    fig = plt.figure(figsize=(8, 4))
    fig.add_subplot(1, 3, 1)
    plt.imshow(image, cmap=plt.cm.gray)

    fig.add_subplot(1, 3, 2)
    plt.imshow(mask, cmap=plt.cm.gray)

    fig.add_subplot(1, 3, 3)
    plt.imshow(masked_image, cmap=plt.cm.gray)

    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_root_dir_path', default=DATA_ROOT_DIR_PATH)
    parser.add_argument('--labels_map_filepath', default=LABELS_MAP_FILEPATH)
    parser.add_argument('--output_rowdict_filepath', default=OUTPUT_ROWDICT_FILEPATH)
    parser.add_argument('--lookup_tables_root_dir', default=LOOKUP_TABLES_ROOT_DIR)
    parser.add_argument('--patient_tables_root_dir', default=PATIENT_TABLES_ROOT_DIR)
    parser.add_argument('--dataset_root_path', default=DATASET_ROOT_PATH)
    parser.add_argument('--output_root_dir', default=OUTPUT_ROOT_DIR)
    parser.add_argument('--n_frames', default=16, type=int)
    parser.add_argument('--mode', default='middle')

    args = parser.parse_args()

    runCreateLookupTables(data_root_dir_path=args.data_root_dir_path, labels_map_filepath=args.labels_map_filepath,
                          output_rowdict_filepath=args.output_rowdict_filepath, lookup_tables_root_dir=args.lookup_tables_root_dir)
    runSavePatientFrames(lookup_tables_root_dir=args.lookup_tables_root_dir, patient_tables_root_dir=args.patient_tables_root_dir)

    dataset_name = f'{args.n_frames}-frame-{args.mode}-data'
    dataset_path = os.path.join(DATASET_ROOT_PATH, dataset_name+'.csv')
    runCreateDataset(dataset_path, n_frames=args.n_frames, mode=args.mode,
                     patient_tables_root_dir=args.patient_tables_root_dir)
    paths = createFramesForDatasets(dataset_path, n_datasets=5, n_train_samples=1000, n_test_samples=200,
                                    output_name=dataset_name, output_root_dir=args.output_root_dir)
    saveDatasets(paths)
