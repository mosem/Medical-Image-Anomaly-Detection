import pandas as pd
from pandas import read_csv
import os
import concurrent.futures
import logging
import pickle

root_dir_path = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/stage_2_train'
labels_map_file = '/vol/ep/mm/anomaly_detection/data/rsna/local/dir_001/stage_2_train.csv'

pd.set_option('display.max_columns', None)

def createDirLookupTable(child_dir_path, rowDict):
    logging.info(f"starting {child_dir_path}")
    output_filepath = os.path.join(root_dir_path, f"{child_dir_path.name}_lookup_table.csv")
    df = read_csv(labels_map_file)
    lookupTable = pd.DataFrame(columns=df.columns)
    lookupTable['filepath'] = []
    lookupTable.dropna(inplace=True)
    for i, file in enumerate(os.scandir(child_dir_path)):
        filename, file_extension = os.path.splitext(file.name)
        # logging.info(f"{i}: {filename}")
        row_indexes = rowDict[filename]
        temp = df.loc[row_indexes].copy()
        temp['filepath'] = os.path.abspath(file)
        lookupTable = lookupTable.append(temp, ignore_index=True)
    lookupTable.to_csv(output_filepath)
    logging.info(f"createDirLookupTable: Done {child_dir_path}.")


def extractId(fullIdStr):
    return "_".join(fullIdStr.split("_", 2)[:2])


def createRowDict():
    df = read_csv(labels_map_file)
    rowDict = {}
    for index, row in df.iterrows():
        id = extractId(row['ID'])
        if id not in rowDict:
            rowDict[id] = []
        rowDict[id].append(index)
    print('Done creating rowDict.')
    return rowDict


def createAndSaveRowDict():
    rowDict = createRowDict()
    filename = 'rowDict.pickle'
    with open(filename, 'wb') as handle:
        pickle.dump(rowDict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"saved row dict as {filename}")

def createLookupTable(dir_path):
    with open('rowDict.pickle', 'rb') as handle:
        rowDict = pickle.load(handle)
    args = [(child_dir_path, rowDict) for child_dir_path in os.scandir(dir_path)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        executor.map(lambda p: createDirLookupTable(*p), args)
    print("createLookupTable: Done.")


if __name__ == "__main__":
    logging.basicConfig(format='%(levelname)s:%(message)s', level=logging.DEBUG)
    # createAndSaveRowDict()
    createLookupTable(root_dir_path)
