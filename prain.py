import pickle
import h5py

import numpy as np

from pathlib import Path
from natsort import natsorted

from datetime import datetime


CORNERS = [
    (52.899699999999996, 0.1285),
    (49.3965, 6.838899999999999)
]
DH = CORNERS[1][0] - CORNERS[0][0]
DW = CORNERS[1][1] - CORNERS[0][1]


def get_all_files(folder_path, ext=None):
    """
    Get all files from the specified folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        list: A list of file paths.
    """
    folder = Path(folder_path)

    if ext is None:
        return [str(f) for f in folder.rglob("*") if f.is_file()]
    else:
        return [str(f) for f in folder.rglob(f"*.{ext}") if f.is_file()]


def to_pixel_loc(lat, lon, height, width):
    y = int((lat - CORNERS[0][0]) / DH * height)
    x = int((lon - CORNERS[0][1]) / DW * width)
    return y, x


def get_data_from_file(f, data_dict, rainfall_dict):
    dt_str_splitted = f.split('/')[-1].split('.')[0].split('-')
    year = int(dt_str_splitted[0])
    month = int(dt_str_splitted[1])
    day = int(dt_str_splitted[2])
    hour = int(dt_str_splitted[3])
    dt = datetime(year, month, day, hour)
    dt_str = dt.strftime('%Y-%m-%dT%H:%M:%S.') + '000000000'

    try:
        with h5py.File(f, "r") as hdf:   
            # Access a specific dataset
            dataset = hdf["dataset1"]  # Replace with the actual dataset name

            if len(list(dataset.keys())) != 3:
                print(f, dataset.keys())

            rainmap = np.array(dataset['data1']['data'])
            height, width = rainmap.shape
            for k in data_dict.keys():
                y, x = to_pixel_loc(k[0], k[1], height, width)
                r1x1 = rainmap[y, x]
                r3x3 = np.sum(rainmap[y-1:y+1, x-1:x+1])
                r5x5 = np.sum(rainmap[y-2:y+2, x-2:x+2])
                r7x7 = np.sum(rainmap[y-3:y+3, x-3:x+3])
                r9x9 = np.sum(rainmap[y-4:y+4, x-4:x+4])

                if k not in rainfall_dict.keys():
                    rainfall_dict[k] = {}
                
                rainfall_dict[k][dt_str] = {
                    'r1x1': r1x1,
                    'r3x3': r3x3,
                    'r5x5': r5x5,
                    'r7x7': r7x7,
                    'r9x9': r9x9,
                }
    except (OSError, IOError) as e:
        print(f"[Warning] Failed to open HDF5 file {f}: {e}")
        return None  # or an empty dict, depending on your use case

    return rainfall_dict


# Path to the HDF5 file
file_paths = get_all_files('data/rainfall', 'hdf5')
file_paths = natsorted(file_paths)

with open('data/selected_stats.pkl', 'rb') as f:
    data = pickle.load(f)

rainfall_dict = {}
for f in file_paths:
    print(f)
    get_data_from_file(f, data, rainfall_dict)

with open('data/rainfall2.pkl', 'wb') as f:
    pickle.dump(rainfall_dict, f)