import csv
from pathlib import Path

import random
import math
import pickle
import numpy as np


"""
Lower-left corner (longitude, latitude): (0.1285, 49.3965)
Lower-right corner (longitude, latitude): (6.838899999999999, 49.3965)
Upper-left corner (longitude, latitude): (0.1285, 52.899699999999996)
Upper-right corner (longitude, latitude): (6.838899999999999, 52.899699999999996)
"""

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


def create_neighboring_dict(loc_dict, k=10):
    """
    Create a neighboring dictionary where the key is the keys of loc_dict
    and the values are the 10 closest keys in loc_dict.

    Args:
        loc_dict: Dictionary with keys as (latitude, longitude) tuples.
        k: Number of closest neighbors to find for each key.

    Returns:
        A dictionary where the keys are the same as loc_dict, and the values are
        lists of 10 closest keys.
    """
    keys = list(loc_dict.keys())
    coords = np.array(keys)  # Convert keys to a numpy array for easier distance computation
    neighbor_dict = {}

    for i, key in enumerate(keys):
        distances = np.linalg.norm(coords - coords[i], axis=1)  # Compute distances to all other keys
        closest_indices = np.argsort(distances)[1:k+1]  # Sort distances and get indices of 10 closest, excluding itself
        neighbor_dict[key] = [keys[j] for j in closest_indices]
    
    return neighbor_dict


def get_all_sensor_locations():
    file_paths = get_all_files('data/raw', ext='pkl')

    loc_dict = {}
    for k, fp in enumerate(file_paths):
        with open(fp, 'rb') as f:
            data = pickle.load(f)

            date_time, _ = extract_values_from_file(fp.replace('pkl', 'csv'))
            loc_dict[(data['latitude'], data['longitude'])] = {}

    return loc_dict


def extract_values_from_file(file_path):
        """
        Reads a file with a semi-colon delimiter and extracts values from the second column.
        
        Args:
            file_path (str): Path to the file to be read.
        
        Returns:
            list: A list of values from the second column.
        """
        values = []
        date_time = []
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                # Skip lines that are metadata or headers
                if len(row) > 1 and not row[0].startswith('#'):
                    date_time.append(row[0])
                    str_value = row[1].replace(',', '.')
                    str_value = str_value if str_value != '' else '-1'
                    values.append(float(str_value))  # Extract the value in the second column
        return date_time, values


# Function to calculate the distance between two geographic points
def haversine(lat1, lon1, lat2, lon2):
    # Radius of Earth in kilometers
    R = 6371
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# Function to find a set of close locations
def find_close_locations(loc_dict, num_points=5):
    loc_list = list(loc_dict.keys())  # Extract all location coordinates
    selected_points = []
    
    # Start with a random point
    current_point = random.choice(loc_list)
    selected_points.append(current_point)
    loc_list.remove(current_point)
    
    # Find the closest points iteratively
    while len(selected_points) < num_points and loc_list:
        closest_point = None
        min_distance = float('inf')
        
        # Find the closest point to the selected set
        for point in loc_list:
            # Calculate distance to the closest selected point
            distance = min(haversine(lat, lon, point[0], point[1]) 
                           for lat, lon in selected_points)
            if distance < min_distance:
                min_distance = distance
                closest_point = point
        
        # Add the closest point to the set and remove it from the list
        selected_points.append(closest_point)
        loc_list.remove(closest_point)
    
    return selected_points


def main():
    # loc_dict = get_all_sensor_locations()

    # # Path to the HDF5 file
    # file_paths = get_all_files('rainfall')
    # file_paths = natsorted(file_paths)

    # previous = None
    # for f in file_paths:
    #     loc_dict, previous = get_data_from_file(f, loc_dict, previous)

    # with open('rainfall.pkl', 'wb') as f:
    #     pickle.dump(loc_dict, f)
    # exit()

    loc_dict = get_all_sensor_locations()
    exit()

    neighbor_dict = create_neighboring_dict(loc_dict)
    # with open('neighbor.pkl', 'wb') as f:
    #     pickle.dump(neighbor_dict, f)

    selected_points = find_close_locations(loc_dict, num_points=256)

    new_selected_points = selected_points.copy()
    for s in selected_points:
        for nb in neighbor_dict[s]:
            if nb not in new_selected_points:
                new_selected_points.append(nb)
    
    with open('subset.pkl', 'wb') as f:
        pickle.dump(new_selected_points, f)

    subset_loc_dict = {
        p:loc_dict[p] for p in new_selected_points
    }

    neighbor_dict = create_neighboring_dict(subset_loc_dict)
    with open('subset_neighbor.pkl', 'wb') as f:
        pickle.dump(neighbor_dict, f)


main()