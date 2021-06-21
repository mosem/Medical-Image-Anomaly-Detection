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

#TODO: change paths

LABELS_MAP_FILEPATH = '/content/drive/MyDrive/anomaly_detection/data/rsna/stage_2_train.csv'
OUTPUT_ROWDICT_FILEPATH = '/content/drive/MyDrive/anomaly_detection/data/rsna/rowDict.pickle'
DATA_ROOT_DIR_PATH = '/content/drive/MyDrive/anomaly_detection/data/rsna/stage_2_train'

PATIENT_TABLES_ROOT_DIR = '/content/drive/MyDrive/anomaly_detection/data/rsna/patient_tables'
DATASET_PATH = '/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data.csv'
LOOKUP_TABLES_ROOT_DIR = '/content/drive/MyDrive/anomaly_detection/data/rsna/lookup_tables_metadata'

OUTPUT_ROOT_DIR = '/content/drive/MyDrive/anomaly_detection/data/rsna/'

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
    lookupTable.dropna(inplace=True)
    print(f"{child_dir_path}: starting enumeration")
    for i, file in enumerate(os.scandir(child_dir_path)):
        filename, file_extension = os.path.splitext(file.name)
        if filename not in rowDict:
            print(f"{child_dir_path.name}. KeyError: {filename} not in rowDict")
            continue
        row_indexes = rowDict[filename]
        temp = df.loc[row_indexes].copy()
        filepath = os.path.abspath(file)
        addMetadata(temp, filepath)
        lookupTable = lookupTable.append(temp, ignore_index=True)
    lookupTable.to_csv(output_filepath)
    print(f"createDirLookupTable: Done {child_dir_path}.")


def addMetadata(frame, filepath):
    dicom_file = dicom.dcmread(filepath)
    frame['filepath'] = filepath
    metadata_attributes_names = ['PatientID', 'StudyInstanceUID', 'SeriesInstanceUID', 'ImagePositionPatient', 'ImageOrientationPatient']
    for name in metadata_attributes_names:
        frame[name] = dicom_file[name].value


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


def runCreateLookupTables():

    if not Path(OUTPUT_ROWDICT_FILEPATH).is_file():
        createAndSaveRowDict(LABELS_MAP_FILEPATH, OUTPUT_ROWDICT_FILEPATH)

    if not Path(LOOKUP_TABLES_ROOT_DIR).is_dir():
        os.mkdir(LOOKUP_TABLES_ROOT_DIR)

    createLookupTables(DATA_ROOT_DIR_PATH, LABELS_MAP_FILEPATH, OUTPUT_ROWDICT_FILEPATH, LOOKUP_TABLES_ROOT_DIR)


# Save patient directories

def getPatientFrames(dir_path):
    metadata_paths = [path for path in os.scandir(dir_path)]
    frames = [read_csv(metadata_table_path) for metadata_table_path in metadata_paths]
    full_metadata_frame = pd.concat(frames)

    patients_frames = [x.reset_index(drop=True).drop("Unnamed: 0", axis=1) for _, x in
                       full_metadata_frame.groupby(['PatientID'])]
    return patients_frames

def savePatientDirectory(patient_frame, output_dir):
    patient_id = patient_frame['PatientID'][0]
    sorted = patient_frame.sort_values(by="ImagePositionPatient",
                                       key=lambda positions: [ast.literal_eval(l)[2] for l in positions]).reset_index(
        drop=True)
    patient_dir_path = os.path.join(output_dir, patient_id)
    if Path(patient_dir_path).is_dir():
        print(f"{patient_dir_path} exists.")
        return
    os.mkdir(patient_dir_path)

    series_frames = [x.reset_index(drop=True).drop("Unnamed: 0", axis=1) for _, x in
                     sorted.groupby(['SeriesInstanceUID'])]
    for series_frame in series_frames:
        frame_filename = series_frame['SeriesInstanceUID'][0] + '.csv'
        frame_path = os.path.join(patient_dir_path, frame_filename)
        if Path(frame_path).is_file():
            print(f"{frame_path} already exists.")
            continue
        series_frame.reset_index(drop=True).to_csv(frame_path)
    print(f"savePatientDirectories: done {patient_id}")

def savePatientDirectories(patients_frames, output_dir):
    args = [(patient_frame, output_dir) for patient_frame in patients_frames]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        executor.map(lambda p: savePatientDirectory(*p), args)


def runSavePatientFrames():
    patient_frames = getPatientFrames(LOOKUP_TABLES_ROOT_DIR)
    savePatientDirectories(patient_frames, PATIENT_TABLES_ROOT_DIR)


# Create table for n-frame dataset

def createRow(table_frame_entry, n_frames):
    print(f"Reading table frame: {table_frame_entry.path}")
    table_frame = read_csv(table_frame_entry.path)
    n_rows = len(table_frame.index)
    if n_rows < n_frames:
        print(f"{table_frame_entry.name}: Not enough rows. n_rows: {n_rows}, n_frames: {n_frames}./nExiting.")
        return None
    middle_row = n_rows // 2
    start_row = middle_row - n_frames // 2
    end_row = middle_row + ceil(n_frames / 2) - 1
    normal = (table_frame.loc[start_row:end_row, ['Label']] == 0).all().bool()
    label = 0 if normal else 1
    indices = [i for i in range(start_row, end_row + 1)]
    print(f"Done creating row: {table_frame_entry.path}")
    return pd.Series({'path': table_frame_entry.path, 'label': label, 'indices': indices})


def createPatientData(patient_dir_entry, n_frames):
    print(f"creating data for patient: {patient_dir_entry}")
    table_entries = [entry for entry in os.scandir(patient_dir_entry.path) if entry.name.endswith('.csv')]
    for i, table_entry in enumerate(table_entries):
        row = createRow(table_entry, n_frames)
        if row is not None:
            return row  # create a single row for each patient
        elif i == len(table_entries) - 1:
            print(f"no rows were created for {table_entry.name}")
    print(f"no data created for {patient_dir_entry}")
    return None


def createDataset(root_dir, n_frames, n_samples=10000):
    print('Creating dataset')
    patient_dirs_entries = [path for path in os.scandir(root_dir) if path.is_dir()]
    n_samples = min(n_samples, len(patient_dirs_entries))
    args = [(patient_dir_entry, n_frames) for patient_dir_entry in patient_dirs_entries[:n_samples]]
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        rows = executor.map(lambda p: createPatientData(*p), args)

    print('Done creating dataset')
    rows = [row for row in rows if row is not None]
    return pd.DataFrame(rows)


def runCreateDataset():
    n_frames = 8
    dataset_frame = createDataset(PATIENT_TABLES_ROOT_DIR, n_frames, n_samples=10000)
    dataset_frame.to_csv(DATASET_PATH)


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


def createFramesForDatasets(dataset_path, n_datasets, output_name):
    dataset_frame = read_csv(dataset_path)
    for i in range(n_datasets):
        train_frame_filename = '-'.join(output_name, 'train', str(i))
        val_frame_filename = '-'.join(output_name, 'val', str(i))
        test_frame_filename = '-'.join(output_name, 'test', str(i))

    train_frame_a = createOneClassFrame(dataset_frame, 1000, 5000)
    # train_frame_b = createOneClassFrame(dataset_frame, 1000, 1000)
    # train_frame_c = createOneClassFrame(dataset_frame, 1000, 2000)
    # train_frame_d = createOneClassFrame(dataset_frame, 1000, 3000)
    # train_frame_e = createOneClassFrame(dataset_frame, 1000, 4000)
    # test_frame_a = createBalancedClassFrame(dataset_frame, 200,0,0)
    # test_frame_b = createBalancedClassFrame(dataset_frame, 200,100,100)
    # test_frame_c = createBalancedClassFrame(dataset_frame, 200,200,200)
    # test_frame_d = createBalancedClassFrame(dataset_frame, 200,300,300)
    # test_frame_e = createBalancedClassFrame(dataset_frame, 200,400,400)

    train_frame_a.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-train-a.csv', index=False)
    # train_frame_b.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-train-b.csv', index=False)
    # train_frame_c.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-train-c.csv', index=False)
    # train_frame_d.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-train-d.csv', index=False)
    # train_frame_e.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-train-e.csv', index=False)

    # test_frame_a.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-test-a.csv', index=False)
    # test_frame_b.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-test-b.csv', index=False)
    # test_frame_c.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-test-c.csv', index=False)
    # test_frame_d.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-test-d.csv', index=False)
    # test_frame_e.to_csv('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-test-e.csv', index=False)
    print('done.')



# Save Dataset Images to image/dicom files


def window_image(pixel_array, window_center, window_width, is_normalize=True):
    image_min = window_center - window_width // 2
    image_max = window_center + window_width // 2
    image = np.clip(pixel_array, image_min, image_max)

    if is_normalize:
        image = (image - image_min) / (image_max - image_min)
    return image


def normalize_dicom(dicom):
    if (dicom.BitsStored == 12) and (dicom.PixelRepresentation == 0) and (int(dicom.RescaleIntercept) > -100):
        # see: https://www.kaggle.com/jhoward/cleaning-the-data-for-rapid-prototyping-fastai
        p = dicom.pixel_array + 1000
        p[p >= 4096] = p[p >= 4096] - 4096
        dicom.PixelData = p.tobytes()
        dicom.RescaleIntercept = -1000

    pixel_array = dicom.pixel_array * dicom.RescaleSlope + dicom.RescaleIntercept
    brain = window_image(pixel_array, 40, 80)
    subdural = window_image(pixel_array, 80, 200)
    soft_tissue = window_image(pixel_array, 40, 380)

    image_array = np.dstack([soft_tissue, subdural, brain])
    image_array = (image_array * 255).astype(np.uint8)
    image = Image.fromarray(image_array)
    return image


def saveImages(item_table_path, indices, output_dir_row, convert_to_png):
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
            continue
        else:
            print(f'{output_path}: processing.')

        img_path = table.loc[idx, 'filepath']
        dicom_image = dicom.dcmread(img_path)

        if convert_to_png:
            normalized_image = normalize_dicom(dicom_image)
            normalized_image.save(output_path)
        else:
            dicom_image.save_as(output_path)
        samples_output_paths.append(output_path)
    return samples_output_paths


def saveRow(input_frame, row_idx, output_dir_path, convert_to_png):
    output_frame = pd.DataFrame(columns=['ID', 'label', 'filepaths'])
    csv_filepath = input_frame.at[row_idx, 'path']
    label = input_frame.at[row_idx, 'label']
    row_path, row_basename = os.path.split(csv_filepath)
    output_frame.at[row_idx, 'ID'] = os.path.splitext(row_basename)[0]
    output_frame.at[row_idx, 'label'] = label
    output_dir_row = os.path.join(output_dir_path, os.path.splitext(row_basename)[0])
    indices = ast.literal_eval(input_frame.loc[row_idx, 'indices'])
    output_frame.at[row_idx, 'filepaths'] = saveImages(csv_filepath, indices, output_dir_row, convert_to_png)
    return output_frame


def saveDataset(table_path, output_dir_name=None, convert_to_png=True, start=0, n_samples=1000):
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

    # args = [(dataset_frame,i,output_dir_path, convert_to_png) for i, _ in islice(dataset_frame.iterrows(), start, end)]
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
        samples_frame.at[i, 'filepaths'] = saveImages(csv_filepath, indices, output_dir_row, convert_to_png)

    samples_frame.to_csv(os.path.join(output_dir_path, 'lookup_table.csv'))

    print(f'done saving to {output_dir_name}')


def saveDatasets():
    saveDataset('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-train-a.csv',
                '8-frame-data-train-a-dcm', False) # saving as dicom files

    saveDataset('/content/drive/MyDrive/anomaly_detection/data/rsna/8-frame-data-train-a.csv',
                '8-frame-data-train-a', True)  # saving as png files