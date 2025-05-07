import pytz
import torch

import os
import csv
import math

import random
import pickle
import numpy as np

from geopy.distance import geodesic
from datetime import datetime, timedelta
from dateutil import parser
from torch.utils.data import Dataset
from scipy.signal import medfilt

from torch_geometric.data import Data

import matplotlib.pyplot as plt

import networkx as nx


def plot_predictions_with_time(values, save_path=None):
    # Plot the predictions
    plt.figure(figsize=(12, 6))
    plt.plot(np.arange(len(values)), values, linestyle='-', label='Predicted Values')
    plt.title('Predicted Values Over Time')
    plt.xlabel('Time')
    plt.ylabel('Predicted Value')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    
    # Save the plot if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=600)
    plt.close()


def get_time_difference_in_hours(time1: str, time2: str) -> int:
    """
    Computes the total number of hours (as an integer) between two timestamps.

    Parameters:
    - time1 (str): First timestamp in ISO 8601 format.
    - time2 (str): Second timestamp in ISO 8601 format.

    Returns:
    - int: Total hours between the two timestamps.
    """
    # Parse timestamps into timezone-aware datetime objects
    dt1 = parser.isoparse(time1)
    dt2 = parser.isoparse(time2)

    # Compute time difference
    time_difference = dt1 - dt2

    # Convert to total hours (integer)
    total_hours = int(time_difference.total_seconds() // 3600)

    return total_hours


def compute_horizontal_vertical_distance(lat1, lon1, lat2, lon2):
    """
    Computes the signed horizontal (East-West) and vertical (North-South) distance 
    between two GPS locations.

    Parameters:
    - lat1, lon1: Latitude and longitude of point A.
    - lat2, lon2: Latitude and longitude of point B.

    Returns:
    - vertical_distance_km: Positive = North, Negative = South
    - horizontal_distance_km: Positive = East, Negative = West
    """

    # Compute signed vertical distance (North-South)
    vertical_distance_km = geodesic((lat1, lon1), (lat2, lon1)).km
    vertical_distance_km *= 1 if lat2 > lat1 else -1  # Make negative if moving South

    # Compute signed horizontal distance (East-West)
    horizontal_distance_km = geodesic((lat1, lon1), (lat1, lon2)).km
    horizontal_distance_km *= 1 if lon2 > lon1 else -1  # Make negative if moving West

    return vertical_distance_km, horizontal_distance_km


def detect_early_shift_cutoff(timestamps, values, window_size=100, drop_threshold=4):
    """
    Automatically detect when an anomalous early period ends in a time series.

    Args:
        timestamps (np.ndarray): Time array (assumed sorted).
        values (np.ndarray): Corresponding value array.
        window_size (int): Rolling window size to compute local medians.
        drop_threshold (float): If a window's median is < this fraction of the later global median, it's considered anomalous.
        min_valid_fraction (float): Minimum proportion of the time series that must be kept.
    
    Returns:
        cutoff_index (int): Index after which the series appears stable.
                            Return 0 if no early anomaly is found.
    """
    values = np.array(values)
    timestamps = np.array(timestamps)
    n = len(values)

    # Compute global stable median using later part
    later_values = values[int(n * 0.5):]  # last 50% assumed more reliable
    stable_median = np.median(later_values)

    if stable_median == 0 or np.isnan(stable_median):
        return 0  # Don't attempt to cut if median is invalid

    # Sliding window through early part
    found = False
    for i in range(0, n - window_size):
        window = values[i:i + window_size]
        window_median = np.median(window)

        # If the window is too far below stable levels
        if stable_median - window_median >= drop_threshold:
            found = True
        else:
            if found:
                return i + window_size

    return 0  # If no transition found


class WaterDataset(Dataset):
    def __init__(self, path, train=False, selected_stations=None, input_type='pte', length=12 * 3, n_neighbors=4): # ='tdata'
        self.train = train
        self.input_type = input_type
        self.length = length
        self.n_neighbors = n_neighbors

        with open('data/selected_stats.pkl', 'rb') as f:
            data = pickle.load(f)

        with open('data/loading.pkl', 'rb') as f:
            self.loading = pickle.load(f)
        
        self.selected_areas = [
            'DHMVIIDTMRAS1m_k13', 'DHMVIIDTMRAS1m_k14', 'DHMVIIDTMRAS1m_k15',
            'DHMVIIDTMRAS1m_k21', 'DHMVIIDTMRAS1m_k22', 'DHMVIIDTMRAS1m_k23',
            'DHMVIIDTMRAS1m_k29', 'DHMVIIDTMRAS1m_k30', 'DHMVIIDTMRAS1m_k31',
        ]

        self.invalid_list = [
            # test
            (50.812199, 3.61216),
            (50.89711467, 3.601662218),
            (51.226010931417, 4.4488923983755),
            (50.8260145913424, 3.61160845853722),
            (51.019212, 3.703334),
            (51.020861, 3.697731),
            (51.003619, 3.747527),
            (51.035775, 3.668258),
            (51.006583, 3.652972),
        ]

        self.stations_in_selected_areas = []
        for a in self.loading.keys():
            flag = True
            for kk in a:
                if kk not in self.selected_areas:
                    flag = False
            if flag:
                self.stations_in_selected_areas.extend(
                    [kk[0] for kk in self.loading[a]]
                )

        # if not os.path.exists('data/selected_stats.pkl'): # True:#
        #     new_data = {}
        #     for k in data.keys():
        #         if k in self.stations_in_selected_areas and k not in self.invalid_list:
        #             new_data[k] = data[k]

        #     with open('data/selected_stats.pkl', 'wb') as f:
        #         pickle.dump(new_data, f)

        #     exit()
        
        self.using_keys = []
        self.data = {}
        for k in data.keys():
            if k in selected_stations and k in self.stations_in_selected_areas and k not in self.invalid_list:
                new_nb = {
                    nb_: data[k]['neighbor'][nb_] for nb_ in data[k]['neighbor'] if nb_ in selected_stations and nb_ in data.keys() and len(data[nb_]['values']) > self.length and len(data[k]['values']) > self.length and len(data[nb_]['values']) / len(data[k]['values']) > 0.5
                }

                data[k]['neighbor'] = new_nb

                self.data[k] = data[k]
                
                if len(list(new_nb.keys())) > 0:
                    self.using_keys.append(k)

        self.original_data = self.data

        stage = 'train' if self.train else 'test'
        if not os.path.exists(f'data/{stage}.pkl'):
            for k in self.data.keys():
                t = self.data[k]['time']
                v = self.data[k]['values']
                idx = detect_early_shift_cutoff(t, v)
                self.data[k]['time'] = self.data[k]['time'][idx:]
                self.data[k]['values'] = self.data[k]['values'][idx:]

            print(f'Final {stage} length: ', len(list(self.data.keys())))
            with open(f'data/{stage}.pkl', 'wb') as f:
                pickle.dump(self.data, f) 
        else:
            with open(f'data/{stage}.pkl', 'rb') as f:
                self.data = pickle.load(f)
            print(f'Final {stage} length: ', len(list(self.data.keys())))

        # with open('data/selected_stats.pkl', 'wb') as f:
        #     pickle.dump(self.data, f)
        # exit()

    def __len__(self):
        return len(self.using_keys) if not self.train else len(self.using_keys) * 10

    def get_neighbor_input(self, stats):
        # avg_elevation, std_elevation, max_z, mean_z, min_z, std_z
        elevation = stats['elevation']
        distance = stats['distance']
        displacement = stats['displacement']
        delta_elevation = stats['key_elevation'] - stats['src_elevation']
        
        slope = stats['slope']
        slope_mean = stats['slope_mean']
        slope_std = stats['slope_std']
        slope_min = stats['slope_min']
        slope_max = stats['slope_max']
        slope_median = stats['slope_median']
        
        key_slope = stats['key_slope']
        key_elevation = stats['key_elevation']
        src_elevation = stats['src_elevation']

        xs = [
            # distance features
            distance, displacement[0], displacement[1],
            # elevation features
            *elevation, delta_elevation,
            # slope features
            key_slope, slope,
            slope_mean, slope_std, slope_min, slope_max, slope_median
        ]

        return xs, elevation, key_elevation, src_elevation

    def __getitem__(self, idx):
        loc_key = self.get_random_loc(idx)

        keep_running = True
        while keep_running:
            nb, loc_time, loc_vals = self.get_time_values_from_loc(loc_key)

            loc_vals = medfilt(loc_vals, kernel_size=5)  # try 3, 5, or 7
            print(loc_key)
            x = []
            xs = []
            es = []

            if nb is None:
                print('======>', loc_key)
                continue

            for i in range(len(nb)):
                _nb = nb[i]

                time_to_idx_nb = {t: idx for idx, t in enumerate(self.data[_nb]['time'])}
                idx_matches = [time_to_idx_nb.get(t, -1) for t in loc_time]
                nb_values = [self.original_data[_nb]['values'][idx] if idx != -1 else -1 for idx in idx_matches]
    
                if np.mean(nb_values) < 0:
                    continue

                if os.path.exists(f'data/tmp/stats-{loc_key}-{_nb}.pkl'):
                    with open(f'data/tmp/stats-{loc_key}-{_nb}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, e, _, _ = self.get_neighbor_input(stats)
                        direction = -1
                elif os.path.exists(f'data/tmp/stats-{_nb}-{loc_key}.pkl'):
                    with open(f'data/tmp/stats-{_nb}-{loc_key}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, e, _, _ = self.get_neighbor_input(stats)
                        direction = 1
                else:
                    displacement = np.array(_nb) - np.array(loc_key)
                    if 'elevation' not in self.data[_nb].keys():
                        print(_nb)
                        continue
                    if 'elevation' not in self.data[loc_key].keys():
                        print(loc_key)
                        continue
                    
                    delta_elevation = self.data[_nb]['elevation'] - self.data[loc_key]['elevation']
                    _xs = [
                        # distance features
                        -1, displacement[0], displacement[1],
                        # elevation features
                        -1, -1, -1, -1, -1, -1, delta_elevation,
                        # slope features
                        -1, -1,
                        -1, -1, -1, -1, -1
                    ]

                print(_nb, _xs[0])
                if 'median' not in self.data[_nb].keys():
                    self.data[_nb]['median'] = np.median(self.data[_nb]['values'])
                    if math.isnan(self.data[_nb]['median']):
                        print('nan')
                        continue
                
                xs.append(_xs)
                es.append(self.data[_nb]['median'])
                x.append(nb_values)

            if 'median' not in self.data[loc_key].keys():
                self.data[loc_key]['median'] = np.median(self.data[loc_key]['values'])
            loc_vals = loc_vals - self.data[loc_key]['median']

            if len(x) <= 0:
                print('======>', loc_key)
            else:
                keep_running = False
        # print(loc_key)
        x = np.array(x)
        xs = np.array(xs)
        es = np.array(es)

        x = torch.tensor(x).float()
        xs = torch.tensor(xs).float()
        es = torch.tensor(es).float().unsqueeze(-1)
        valid = (x > 0).float()
        x = x - es
        y = torch.tensor(loc_vals).float()
        return x, xs, y, es, valid

    def get_time_values_from_loc(self, loc_key):
        nb = self.data[loc_key]['neighbor'].keys()
        nb = [nb_ for nb_ in nb if nb_ in self.original_data.keys() and nb_ != loc_key]

        min_t = np.min(self.data[loc_key]['time'])
        max_t = np.max(self.data[loc_key]['time'])
        for _nb in nb:
            n_min_t = np.min(self.original_data[_nb]['time'])
            n_max_t = np.max(self.original_data[_nb]['time'])
            if n_min_t > min_t:
                min_t = n_min_t
            if n_max_t < max_t:
                max_t = n_max_t

        low_idx = np.where(self.data[loc_key]['time'] == min_t)
        high_idx = np.where(self.data[loc_key]['time'] == max_t)

        if len(low_idx[0]) > 0 and len(high_idx[0]) > 0:
            low_idx = low_idx[0][0]
            high_idx = high_idx[0][0]
        else:
            return None, None, None

        if self.train:
            sample_idx = random.randint(low_idx, high_idx - self.length - 1)
            loc_time = self.data[loc_key]['time'][sample_idx:sample_idx+self.length]
            loc_vals = self.data[loc_key]['values'][sample_idx:sample_idx+self.length]
        else:
            loc_time = self.data[loc_key]['time'][low_idx:high_idx]
            loc_vals = self.data[loc_key]['values'][low_idx:high_idx]

        return nb, loc_time, loc_vals

    def get_random_loc(self, idx):
        while True:
            idx = idx if not self.train else idx % 10 
            loc_key = self.using_keys[idx]

            if len(self.data[loc_key]['time']) >= self.length:
                break
            else:
                idx = random.randint(0, len(self.using_keys) - 1)
        return loc_key


class GWaterDataset(WaterDataset):
    def __getitem__(self, idx):
        loc_key = self.get_random_loc(idx)

        keep_running = True
        while keep_running:
            nb, loc_time, loc_vals = self.get_time_values_from_loc(loc_key)

            updated = False

            if nb is None:
                print(loc_key)
                continue

            loc_vals = medfilt(loc_vals, kernel_size=5)  # try 3, 5, or 7
            
            ndict = {loc_key: {'idx': 0, 'f': np.zeros_like(loc_vals).tolist()}}
            edict = {}

            es = []
            src_values = []
            good_nb = []
            valid = [np.zeros_like(loc_vals, dtype=np.int32).tolist()]

            for _nb in nb:
                time_to_idx_nb = {t: idx for idx, t in enumerate(self.data[_nb]['time'])}
                idx_matches = [time_to_idx_nb.get(t, -1) for t in loc_time]
                nb_values = [self.original_data[_nb]['values'][idx] if idx != -1 else -1 for idx in idx_matches]

                nb_values = np.array(nb_values, dtype=np.float32)

                if np.mean(nb_values) < 0:
                    print('negative!')
                    continue

                # Load stats
                if os.path.exists(f'data/tmp/stats-{loc_key}-{_nb}.pkl'):
                    with open(f'data/tmp/stats-{loc_key}-{_nb}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, _, _, _ = self.get_neighbor_input(stats)
                        direction = -1
                elif os.path.exists(f'data/tmp/stats-{_nb}-{loc_key}.pkl'):
                    with open(f'data/tmp/stats-{_nb}-{loc_key}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, _, _, _ = self.get_neighbor_input(stats)
                        direction = 1
                else:
                    displacement = np.array(_nb) - np.array(loc_key)
                    if 'elevation' not in self.data[_nb].keys():
                        print(_nb)
                        continue
                    if 'elevation' not in self.data[loc_key].keys():
                        print(loc_key)
                        continue
                    delta_elevation = self.data[_nb]['elevation'] - self.data[loc_key]['elevation']
                    _xs = [
                        # distance features
                        -1, displacement[0], displacement[1],
                        # elevation features
                        -1, -1, -1, -1, -1, -1, delta_elevation,
                        # slope features
                        -1, -1,
                        -1, -1, -1, -1, -1
                    ]
                    direction = 0

                # Median handling
                if 'median' not in self.data[_nb]:
                    self.data[_nb]['median'] = np.median(self.data[_nb]['values'])

                valid.append((nb_values > 0).astype(np.int32).tolist())

                src_values.append(nb_values.tolist())

                es.append(self.data[_nb]['median'])
                nb_values -= self.data[_nb]['median']

                if not updated:
                    updated = True
                    if 'median' not in self.data[loc_key]:
                        self.data[loc_key]['median'] = np.median(self.data[loc_key]['values'])
                    loc_vals -= self.data[loc_key]['median']

                ndict = self.update_node_dict(ndict, _nb, nb_values.tolist())
                edict[(_nb, loc_key)] = [*_xs, direction]
                good_nb.append(_nb)

            if len(good_nb) <= 0:
                print('======>', loc_key)
            else:
                keep_running = False

        for nb in good_nb:
            found = True
            while found:
                base = nb
                found = False
                for edge in self.data[loc_key]['sim_graph']['edges']:
                    if base != edge[0] or (edge[1] in ndict.keys() and edge[0] in ndict.keys()):
                        continue

                    path = f'data/tmp/stats-{edge[0]}-{edge[1]}.pkl' if edge[-1] == -1 else f'data/tmp/stats-{edge[1]}-{edge[0]}.pkl'
                    if os.path.exists(path):
                        base = edge[1]
                        found = True
                        with open(path, 'rb') as f:
                            stats = pickle.load(f)
                            _xs, _, _, _ = self.get_neighbor_input(stats)
                        
                        if (edge[1], edge[0]) not in edict.keys():
                            edict[(edge[0], edge[1])] = [*_xs, edge[-1]]
                            prev_len = len(ndict.keys())
                            ndict = self.update_node_dict(ndict, edge[0], [0 for _ in range(len(loc_vals))])
                            if prev_len < len(ndict.keys()):
                                valid.append([0 for _ in range(len(loc_vals))])
                                prev_len += 1
                            
                            ndict = self.update_node_dict(ndict, edge[1], [0 for _ in range(len(loc_vals))])
                            if prev_len < len(ndict.keys()):
                                valid.append([0 for _ in range(len(loc_vals))])

        # Convert node dictionary to tensor
        node_list = list(ndict.keys())
        x = torch.tensor(np.array([ndict[n]['f'] for n in node_list]), dtype=torch.float)

        edge_list = []
        # Convert edge dictionary to tensors
        for idx, e in enumerate(edict.keys()):
            edge_list.append(
                (
                    ndict[e[0]]['idx'],
                    ndict[e[1]]['idx']
                )
            )

        edge_attr = torch.tensor(list(edict.values()), dtype=torch.float)
        mask = edge_attr[:, 0] != -10

        # Loop through each column (feature dimension)
        for i in range(edge_attr.shape[1]):
            col = edge_attr[mask, i]

            # Skip normalization if there's nothing to normalize
            if col.numel() == 0:
                continue

            max_val = col.abs().max()

            if max_val > 0:  # Avoid division by zero
                edge_attr[mask, i] = edge_attr[mask, i] / max_val

        # Convert edge list to tensor format
        edge_index = torch.tensor(edge_list, dtype=torch.long).T  # Transpose to shape [2, num_edges]
        # visualize_edge_index(edge_index, directed=True, title="Graph Visualization")

        # Create PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        valid = torch.tensor(valid)

        es = np.array(es)
        es = torch.tensor(es).float().unsqueeze(-1)
        return torch.tensor(loc_vals), torch.tensor(src_values) - es, valid, graph_data

    def update_node_dict(self, ndict, _nb, nb_values):
        if _nb not in ndict.keys():
            ndict[_nb] = {'idx': len(ndict.keys()), 'f': nb_values}
        return ndict
