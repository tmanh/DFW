import pandas as pd
import pytz
import torch

import os
import csv
import math

import random
import pickle
import numpy as np

from geopy.distance import geodesic
from dateutil import parser
from torch.utils.data import Dataset

from torch_geometric.data import Data

import matplotlib.pyplot as plt

# import geohash


def geohash_to_int_list(geohash_code):
    # Define the base-32 character map for geohashing
    base32_map = "0123456789bcdefghjkmnpqrstuvwxyz"
    # Convert each character in the geohash string to its corresponding integer value
    return [base32_map.index(char) / 31.0 for char in geohash_code]


def gps_encode(loc):
    latitude, longitude = loc
    geohash_code = geohash.encode(latitude, longitude, precision=12)
    return geohash_to_int_list(geohash_code[2:])


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


def split_into_weekly_segments(times, values, r1x1, r3x3, r5x5, r7x7, r9x9, global_start):
    """
    Split the continuous hourly time series into 7-day aligned segments starting from global_start.
    """
    """
    Split into aligned 7-day segments starting from global_start.
    Add a 'week_idx' key that indicates the week number.
    """
    # Convert and sort
    times = np.array(times)
    values = np.array(values)
    r1x1 = np.array(r1x1)
    r3x3 = np.array(r3x3)
    r5x5 = np.array(r5x5)
    r7x7 = np.array(r7x7)
    r9x9 = np.array(r9x9)

    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]
    r1x1 = r1x1[sort_idx]
    r3x3 = r3x3[sort_idx]
    r5x5 = r5x5[sort_idx]
    r7x7 = r7x7[sort_idx]
    r9x9 = r9x9[sort_idx]

    times_pd = pd.to_datetime(times)
    deltas = (times_pd[1:] - times_pd[:-1]).total_seconds() / 3600.0
    breaks = np.where(np.abs(deltas - 1.0) > 1e-3)[0]

    segments = []
    start_idx = 0
    for b in breaks:
        end_idx = b + 1
        if end_idx - start_idx >= 7 * 24:
            segments.append((start_idx, end_idx))
        start_idx = end_idx
    if len(times_pd) - start_idx >= 7 * 24:
        segments.append((start_idx, len(times_pd)))

    one_week = pd.Timedelta(days=7)

    # Initialize result as dict of lists
    aligned_segments = {
        'time': [],
        'values': [],
        'r1x1': [],
        'r3x3': [],
        'r5x5': [],
        'r7x7': [],
        'r9x9': [],
        'week_idx': []  # <<-- Add week index for easy slicing later
    }

    for seg_start, seg_end in segments:
        seg_times = times_pd[seg_start:seg_end]
        seg_values = values[seg_start:seg_end]
        seg_r1x1 = r1x1[seg_start:seg_end]
        seg_r3x3 = r3x3[seg_start:seg_end]
        seg_r5x5 = r5x5[seg_start:seg_end]
        seg_r7x7 = r7x7[seg_start:seg_end]
        seg_r9x9 = r9x9[seg_start:seg_end]

        aligned_start = global_start + ((seg_times[0] - global_start) // one_week) * one_week
        aligned_end = aligned_start + one_week

        while aligned_end <= seg_times[-1]:
            mask = (seg_times >= aligned_start) & (seg_times < aligned_end)
            if np.sum(mask) == 7 * 24:
                aligned_segments['time'].append(seg_times[mask].tolist())
                aligned_segments['values'].append(seg_values[mask].tolist())
                aligned_segments['r1x1'].append(seg_r1x1[mask].tolist())
                aligned_segments['r3x3'].append(seg_r3x3[mask].tolist())
                aligned_segments['r5x5'].append(seg_r5x5[mask].tolist())
                aligned_segments['r7x7'].append(seg_r7x7[mask].tolist())
                aligned_segments['r9x9'].append(seg_r9x9[mask].tolist())
                
                # Add the same week index repeated 168 times (one per hour)
                week_number = int((aligned_start - global_start).total_seconds() // (3600 * 24 * 7))
                aligned_segments['week_idx'].append(week_number)

            aligned_start += one_week
            aligned_end += one_week

    return aligned_segments


def split_into_weekly_segments2(times, values, global_start, interval_minutes=15):
    """
    Split a continuous time series (15-min interval) into 7-day aligned segments starting from global_start.
    Adds a 'week_idx' for each segment.
    """
    # Convert to numpy arrays
    times = np.array(times)
    values = np.array(values)
    sort_idx = np.argsort(times)
    times = times[sort_idx]
    values = values[sort_idx]

    # Convert to pandas datetime64, then to int64 seconds for compact storage
    times_pd = pd.to_datetime(times)
    times_int = times_pd.astype('int64') // 10**9  # seconds since epoch

    # Global start as int (UNIX seconds)
    global_start_int = pd.to_datetime(global_start).to_datetime64().astype('int64') // 10**9

    expected_delta = pd.Timedelta(minutes=interval_minutes)
    samples_per_week = 7 * 24 * (60 // interval_minutes)
    one_week = pd.Timedelta(days=7)

    # Find discontinuities (gaps larger than interval)
    deltas = times_pd[1:] - times_pd[:-1]
    breaks = np.where(deltas > expected_delta + pd.Timedelta(seconds=1))[0]

    segments = []
    start_idx = 0
    for b in breaks:
        end_idx = b + 1
        if end_idx - start_idx >= samples_per_week:
            segments.append((start_idx, end_idx))
        start_idx = end_idx
    if len(times_pd) - start_idx >= samples_per_week:
        segments.append((start_idx, len(times_pd)))

    # Store only week indices and week number at first
    week_indices_list = []
    week_idx_list = []

    for seg_start, seg_end in segments:
        seg_times = times_pd[seg_start:seg_end]
        aligned_start = global_start + ((seg_times[0] - global_start) // one_week) * one_week
        aligned_end = aligned_start + one_week

        while aligned_end <= seg_times[-1]:
            mask = (seg_times >= aligned_start) & (seg_times < aligned_end)
            if np.sum(mask) == samples_per_week:
                week_indices = np.flatnonzero(mask) + seg_start  # global indices
                week_indices_list.append(week_indices)
                week_number = int((aligned_start - global_start) / one_week)
                week_idx_list.append(week_number)
            aligned_start += one_week
            aligned_end += one_week

    # Build compact output dict (times as int UNIX seconds)
    aligned_segments = {
        'time': [],
        'values': [],
        'week_idx': []
    }
    for indices, week_number in zip(week_indices_list, week_idx_list):
        aligned_segments['time'].append(times_int[indices].tolist())  # list of int
        aligned_segments['values'].append(values[indices].tolist())
        aligned_segments['week_idx'].append(week_number)

    return aligned_segments


def pearson_corrcoef(x, y, dim=-1, eps=1e-8):
    """
    Compute Pearson correlation between x and y along given dimension.
    Assumes x and y are of the same shape.
    """
    x_centered = x - x.mean(axis=dim, keepdims=True)
    y_centered = y - y.mean(axis=dim, keepdims=True)

    cov = (x_centered * y_centered).mean(axis=dim)
    std_x = x_centered.std(axis=dim)
    std_y = y_centered.std(axis=dim)

    corr = cov / (std_x * std_y + eps)
    return corr


class WaterDataset(Dataset):
    def create_segments(self, data):
        all_times = []
        for k in data.keys():
            all_times.extend(pd.to_datetime(data[k]['time']))
        global_start = min(all_times)

        # 2. Apply the segmentation
        for k in data.keys():
            aligned_data = split_into_weekly_segments(
                data[k]['time'],
                data[k]['values'],
                data[k]['r1x1'],
                data[k]['r3x3'],
                data[k]['r5x5'],
                data[k]['r7x7'],
                data[k]['r9x9'],
                global_start
            )
            for kk in aligned_data.keys():
                data[k][kk] = aligned_data[kk]
            
        with open('data/selected_stats_rainfall_segment.pkl', 'wb') as f:
            pickle.dump(data, f)
        exit()

    def create_segments2(self, data):
        all_times = []
        for k in data.keys():
            all_times.extend(pd.to_datetime(data[k]['time']))
        global_start = min(all_times)

        # 2. Apply the segmentation
        for k in data.keys():
            aligned_data = split_into_weekly_segments2(
                data[k]['time'],
                data[k]['values'],
                global_start
            )
            for kk in aligned_data.keys():
                data[k][kk] = aligned_data[kk]
            
        with open('data/selected_stats_segment.pkl', 'wb') as f:
            pickle.dump(data, f)
        exit()

    def set_elevation(self, data):
        for k in data.keys():
            nb = data[k]['neighbor'].keys()
            for _nb in nb:
                if os.path.exists(f'data/tmp/stats-{k}-{_nb}.pkl'):
                    with open(f'data/tmp/stats-{k}-{_nb}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        data[k]['elevation'] = stats['key_elevation']
                        data[k]['slope'] = stats['key_slope']
                    break

    def __init__(self, path, train=False, selected_stations=None, input_type='pte', length=12 * 3, n_neighbors=4): # ='tdata'
        self.train = train
        self.input_type = input_type
        self.length = length
        self.n_neighbors = n_neighbors

        with open('data/selected_stats_rainfall_segment.pkl', 'rb') as f:
            data = pickle.load(f)

        # self.create_segments(data)
        # print('xxxx')
        # self.create_segments2(data)
        # print('yyyy')

        with open('data/loading.pkl', 'rb') as f:
            self.loading = pickle.load(f)
        
        self.selected_areas = [
            'DHMVIIDTMRAS1m_k13', 'DHMVIIDTMRAS1m_k14', 'DHMVIIDTMRAS1m_k15',
            'DHMVIIDTMRAS1m_k21', 'DHMVIIDTMRAS1m_k22', 'DHMVIIDTMRAS1m_k23',
            'DHMVIIDTMRAS1m_k29', 'DHMVIIDTMRAS1m_k30', 'DHMVIIDTMRAS1m_k31',
        ]

        self.invalid_list = [
            # test
            (50.9761639541252, 3.50350061937995),
            (50.967587, 3.46339),
            (50.963005, 3.505905),
            (50.752506, 3.6088760000000004),
            # train
            (50.745453, 4.345417),
            (50.7535570834951, 4.26976461821304),
            (50.7486391043145, 4.35302730688102),
            ###
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

        # if not os.path.exists('data/selected_stats_rainfall_segment.pkl'): # True:#
        #     new_data = {}
        #     for k in data.keys():
        #         if k in self.stations_in_selected_areas and k not in self.invalid_list:
        #             new_data[k] = data[k]

        #     with open('data/selected_stats_rainfall_segment.pkl', 'wb') as f:
        #         pickle.dump(new_data, f)

        #     exit()
        self.set_elevation(data)

        self.using_keys = []
        self.data = {}

        for k in data.keys():
            if len(data[k]['week_idx']) < 10:
                continue

            new_nb = {
                nb_: data[k]['neighbor'][nb_] for nb_ in data[k]['neighbor'] if nb_ in data.keys() and len(data[nb_]['week_idx']) >= 10 # and nb_ in selected_stations
            }
            data[k]['neighbor'] = new_nb

            self.data[k] = data[k]
                
            if len(list(new_nb.keys())) > 0 and k in selected_stations and k not in self.invalid_list:
                self.using_keys.append(k)

        self.using_keys = sorted(self.using_keys)

        # count = [0, 0, 0, 0, 0, 0, 0, 0]
        # for k in data.keys():
        #     if len(data[k]['values']) > 0:
        #         corr = pearson_corrcoef(
        #             np.array(data[k]['values']).flatten(),
        #             np.array(data[k]['r1x1']).flatten(),
        #         )
        #     if corr < -0.75:
        #         count[0] += 1
        #     elif corr < -0.5:
        #         count[1] += 1
        #     elif corr < -0.25:
        #         count[2] += 1
        #     elif corr < 0:
        #         count[3] += 1
        #     elif corr < 0.25:
        #         count[4] += 1
        #     elif corr < 0.5:
        #         count[5] += 1
        #     elif corr < 0.75:
        #         count[6] += 1
        #     else:
        #         count[7] += 1
        # print([count[i] / np.sum(count) for i in range(8)])
        # exit()

        stage = 'train' if self.train else 'test'
        print(f'Final {stage} length: ', len(self.using_keys))

    def __len__(self):
        return len(self.using_keys)

    def get_neighbor_input(self, stats):
        # avg_elevation, std_elevation, max_z, mean_z, min_z, std_z
        distance = stats['distance']
        displacement = stats['displacement']
        delta_elevation = stats['key_elevation'] - stats['src_elevation']
        src_slope = stats['src_slope']
        slope_mean = stats['slope_mean']
        slope_std = stats['slope_std']
        slope_min = stats['slope_min']
        slope_max = stats['slope_max']
        slope_median = stats['slope_median']
        
        key_slope = stats['key_slope']
        key_elevation = stats['key_elevation']
        src_elevation = stats['src_elevation']

        elevation = stats['elevation']
        if isinstance(elevation, tuple) and len(elevation) == 6:
            elevation = elevation[:5]
        else:
            elevation = [elevation[k] for k in elevation.keys()]

        xs = [
            # distance features
            distance, displacement[0], displacement[1],
            # elevation features
            *elevation, delta_elevation,
            # slope features
            key_slope, src_slope,
            slope_mean, slope_std, slope_min, slope_max, slope_median
        ]

        return xs, elevation, key_elevation, src_elevation

    def __getitem__(self, idx):
        keep_running = True
        while keep_running:
            loc_key = self.get_random_loc(idx)
            nb, loc_vals, loc_rain, nb_vals, nb_rain = self.get_time_values_from_loc(loc_key)

            if nb is None:
                idx = random.randint(0, len(self) - 1)
                print(loc_key, 'not found')
                continue

            if 'mean' not in self.data[loc_key].keys():
                all_data = np.array(self.data[loc_key]['values'])
                self.data[loc_key]['mean'] = np.mean(all_data)

            xs = []
            es = []

            for i in range(len(nb)):
                _nb = nb[i]

                if os.path.exists(f'data/tmp/stats-{loc_key}-{_nb}.pkl'):
                    with open(f'data/tmp/stats-{loc_key}-{_nb}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, _, _, _ = self.get_neighbor_input(stats)
                        direction = -1
                        _xs = [*_xs, direction, *loc_key, *_nb]
                elif os.path.exists(f'data/tmp/stats-{_nb}-{loc_key}.pkl'):
                    with open(f'data/tmp/stats-{_nb}-{loc_key}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, _, _, _ = self.get_neighbor_input(stats)
                        direction = 1
                        _xs = [*_xs, direction, *loc_key, *_nb]
                else:
                    displacement = np.array(_nb) - np.array(loc_key)
                    delta_elevation = self.data[_nb]['elevation'] - self.data[loc_key]['elevation']
                    _xs = [
                        # distance features
                        -1, displacement[0], displacement[1],
                        # elevation features
                        -1, -1, -1, -1, -1, delta_elevation,
                        # slope features
                        -1, -1,
                        -1, -1, -1, -1, -1, 0, *loc_key, *_nb
                    ]
                if 'mean' not in self.data[_nb].keys():
                    self.data[_nb]['mean'] = np.mean(self.data[_nb]['values'])
                
                xs.append(_xs)
                es.append(self.data[_nb]['mean'])

            loc_vals = loc_vals - self.data[loc_key]['mean']

            if len(nb_vals) <= 0:
                print('======>', loc_key)
            else:
                keep_running = False

        loc_rain = np.array(loc_rain)
        nb_rain = np.array(nb_rain)

        x = np.array(nb_vals)
        xs = np.array(xs)
        es = np.array(es)

        x = torch.tensor(x).float()
        xs = torch.tensor(xs).float()
        
        es = torch.tensor(es).float().unsqueeze(-1)
        valid = (x > 0).float()
        x = x - es
        y = torch.tensor(loc_vals).float()

        loc_rain = torch.tensor(loc_rain).float()
        nb_rain = torch.tensor(nb_rain).float()

        """
        xs = [
            # distance features
            distance, displacement[0], displacement[1],
            # elevation features
            *elevation, delta_elevation,
            # slope features
            key_slope, src_slope,
            slope_mean, slope_std, slope_min, slope_max, slope_median,
            direction, gps1, gps2
        ]
        """
        # mask = xs[:, 0] != -1
        # # Loop through each column (feature dimension)
        # for i in range(xs.shape[1]):
        #     col = xs[mask, i]
        #     max_val = col.abs().max()

        #     if max_val > 0:  # Avoid division by zero
        #         xs[mask, i] = xs[mask, i] / max_val

        return x, xs, y, es, loc_rain, nb_rain, valid

    def get_time_values_from_loc(self, loc_key):
        nb = self.data[loc_key]['neighbor'].keys()
        nb = [nb_ for nb_ in nb if nb_ in self.data.keys() and nb_ != loc_key]
        common_weeks = set(self.data[loc_key]['week_idx'])
        for _nb in nb:
            nb_weeks = set(self.data[_nb]['week_idx'])
            common_weeks = common_weeks.intersection(nb_weeks)
        common_weeks = sorted(common_weeks)

        if common_weeks == []:
            return None, None, None, None, None

        if self.train:
            # Pick 1 week only
            selected_weeks = [random.choice(common_weeks)]
        else:
            # Use all weeks
            selected_weeks = common_weeks

        # Get loc data
        loc_idx = [self.data[loc_key]['week_idx'].index(w) for w in selected_weeks]
        loc_vals = np.array([self.data[loc_key]['values'][i] for i in loc_idx]).flatten()
        # loc_rain = np.array([self.data[loc_key]['r1x1'][i] for i in loc_idx]).flatten()
        loc_rain = loc_vals

        # Get neighbor data
        nb_vals = []
        nb_rain = []
        for _nb in nb:
            nb_idx = [self.data[_nb]['week_idx'].index(w) for w in selected_weeks]
            _nb_vals = np.array([self.data[_nb]['values'][i] for i in nb_idx]).flatten()
            # _nb_rain = np.array([self.data[_nb]['r1x1'][i] for i in nb_idx]).flatten()
            nb_vals.append(_nb_vals)
            nb_rain.append(_nb_vals)
            # nb_rain.append(_nb_rain)

        return nb, loc_vals, loc_rain, nb_vals, nb_rain

    def get_random_loc(self, idx):
        while True:
            loc_key = self.using_keys[idx]

            if self.train:
                if len(self.data[loc_key]['week_idx']) >= self.length:
                    break
                else:
                    idx = random.randint(0, len(self.using_keys) - 1)
            else:
                break
        return loc_key


class GWaterDataset(WaterDataset):
    def __getitem__(self, idx):
        keep_running = True
        while keep_running:
            loc_key = self.get_random_loc(idx)
            nb, loc_vals, loc_rain, nb_vals, nb_rain = self.get_time_values_from_loc(loc_key)
            if nb is None:
                idx = random.randint(0, len(self) - 1)
                print(loc_key, 'not found')
                continue

            updated = False
            ndict = {loc_key: {'idx': 0, 'f': np.zeros_like(loc_vals).tolist()}}
            edict = {}

            es = []
            src_values = []
            good_nb = []
            valid = [np.zeros_like(loc_vals, dtype=np.int32).tolist()]

            src_values = np.array(nb_vals)
            nb_rain = np.array(nb_rain)

            for idx, _nb in enumerate(nb):
                nb_values = nb_vals[idx].copy()
                
                if os.path.exists(f'data/tmp/stats-{loc_key}-{_nb}.pkl'):
                    with open(f'data/tmp/stats-{loc_key}-{_nb}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, _, _, _ = self.get_neighbor_input(stats)
                        direction = -1
                        _xs = [*_xs, direction]
                elif os.path.exists(f'data/tmp/stats-{_nb}-{loc_key}.pkl'):
                    with open(f'data/tmp/stats-{_nb}-{loc_key}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, _, _, _ = self.get_neighbor_input(stats)
                        direction = 1
                        _xs = [*_xs, direction]
                else:
                    displacement = np.array(_nb) - np.array(loc_key)
                    delta_elevation = self.data[_nb]['elevation'] - self.data[loc_key]['elevation']
                    _xs = [
                        # distance features
                        -1, displacement[0], displacement[1],
                        # elevation features
                        -1, -1, -1, -1, -1, delta_elevation,
                        # slope features
                        -1, -1,
                        -1, -1, -1, -1, -1, 0
                    ]

                # mean handling
                if 'mean' not in self.data[_nb]:
                    self.data[_nb]['mean'] = np.mean(self.data[_nb]['values'])

                valid.append((nb_values > 0).astype(np.int32))
                es.append(self.data[_nb]['mean'])
                nb_values = nb_values - self.data[_nb]['mean']

                if not updated:
                    updated = True

                    if 'mean' not in self.data[loc_key].keys():
                        all_data = np.array(self.data[loc_key]['values'])
                        self.data[loc_key]['mean'] = np.mean(all_data)
                    loc_vals -= self.data[loc_key]['mean']

                if self.train:
                    if np.mean(np.abs(loc_vals)) <= 0.2 or np.mean(np.abs(loc_vals)) >= 4:
                        continue

                ndict = self.update_node_dict(ndict, _nb, nb_values)
                edict[(_nb, loc_key)] = _xs
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
                            _xs = [*_xs, edge[-1]]
                        
                        if (edge[1], edge[0]) not in edict.keys():
                            edict[(edge[0], edge[1])] = _xs
                            prev_len = len(ndict.keys())
                            ndict = self.update_node_dict(
                                ndict, edge[0], np.array([0 for _ in range(len(loc_vals))])
                            )
                            if prev_len < len(ndict.keys()):
                                valid.append(
                                    np.array([0 for _ in range(len(loc_vals))])
                                )
                                prev_len += 1
                            
                            ndict = self.update_node_dict(
                                ndict, edge[1], np.array([0 for _ in range(len(loc_vals))])
                            )
                            if prev_len < len(ndict.keys()):
                                valid.append(
                                    np.array([0 for _ in range(len(loc_vals))])
                                )

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

        edge_attr = 1 / (edge_attr + 1e-8)

        # Convert edge list to tensor format
        edge_index = torch.tensor(edge_list, dtype=torch.long).T  # Transpose to shape [2, num_edges]
        # visualize_edge_index(edge_index, directed=True, title="Graph Visualization")

        # Create PyTorch Geometric Data object
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        valid = torch.tensor(np.array(valid))

        es = np.array(es)
        es = torch.tensor(es).float().unsqueeze(-1)
        src_values = torch.tensor(src_values).float() - es

        return torch.tensor(loc_vals), src_values, valid, graph_data

    def update_node_dict(self, ndict, _nb, nb_values):
        if _nb not in ndict.keys():
            ndict[_nb] = {'idx': len(ndict.keys()), 'f': nb_values}
        return ndict


class WaterDatasetX(WaterDataset):
    def __init__(self, path, train=False, selected_stations=None): # ='tdata'
        super().__init__(path, train, selected_stations)

        new_using_keys = []
        for k in self.using_keys:
            if len(self.data[k]['values']) > 39:
                new_using_keys.append(k)
        self.using_keys = new_using_keys

        new_using_keys = []
        for k in self.data.keys():
            if len(self.data[k]['values']) > 39:
                new_using_keys.append(k)
        self.all_using_stations = new_using_keys

        common_weeks = set(self.data[self.using_keys[0]]['week_idx'])
        for k in self.all_using_stations:
            nb_weeks = set(self.data[k]['week_idx'])
            common_weeks = common_weeks.intersection(nb_weeks)
        common_weeks = sorted(common_weeks)
        self.common_weeks = common_weeks

    def get_time_values_from_loc(self, loc_key):
        selected_weeks = self.common_weeks

        # Get loc data
        loc_idx = [self.data[loc_key]['week_idx'].index(w) for w in selected_weeks]
        loc_vals = np.array([self.data[loc_key]['values'][i] for i in loc_idx]).flatten()
        loc_rain = np.array([self.data[loc_key]['r1x1'][i] for i in loc_idx]).flatten()

        return loc_vals, loc_rain

    def __getitem__(self, idx):
        loc_key = self.get_random_loc(idx)

        loc_vals, loc_rain = self.get_time_values_from_loc(loc_key)

        if 'mean' not in self.data[loc_key].keys():
            all_data = np.array(self.data[loc_key]['values'])
            self.data[loc_key]['mean'] = np.mean(all_data)

        xs = [
            # loc features
            *loc_key,
            # elevation & slope features
            self.data[loc_key]['elevation'],
            self.data[loc_key]['slope'],
        ]
        loc_vals = loc_vals - self.data[loc_key]['mean']
        loc_rain = np.array(loc_rain)
        xs = np.array(xs)
        xs = torch.tensor(xs).float()
        y = torch.tensor(loc_vals).float()
        loc_rain = torch.tensor(loc_rain).float()
        return xs, y, loc_rain
    
    def __len__(self):
        return len(self.using_keys)
    

class WaterDatasetY(WaterDataset):
    def get_time_values_from_loc(self, loc_key):
        nb = self.data[loc_key]['neighbor'].keys()
        nb = [nb_ for nb_ in nb if nb_ in self.data.keys() and nb_ != loc_key]
        common_weeks = set(self.data[loc_key]['week_idx'])
        for _nb in nb:
            nb_weeks = set(self.data[_nb]['week_idx'])
            common_weeks = common_weeks.intersection(nb_weeks)
        common_weeks = sorted(common_weeks)

        if common_weeks == []:
            return None, None, None, None, None

        selected_weeks = common_weeks

        # Get loc data
        loc_idx = [self.data[loc_key]['week_idx'].index(w) for w in selected_weeks]
        loc_vals = np.array([self.data[loc_key]['values'][i] for i in loc_idx]).flatten()
        loc_rain = np.array([self.data[loc_key]['r1x1'][i] for i in loc_idx]).flatten()

        # Get neighbor data
        nb_vals = []
        nb_rain = []
        for _nb in nb:
            nb_idx = [self.data[_nb]['week_idx'].index(w) for w in selected_weeks]
            _nb_vals = np.array([self.data[_nb]['values'][i] for i in nb_idx]).flatten()
            _nb_rain = np.array([self.data[_nb]['r1x1'][i] for i in nb_idx]).flatten()
            nb_vals.append(_nb_vals)
            nb_rain.append(_nb_rain)

        return nb, loc_vals, loc_rain, nb_vals, nb_rain

    def __getitem__(self, idx):
        loc_key = self.get_random_loc(idx)

        nb, loc_vals, loc_rain, nb_vals, nb_rain = self.get_time_values_from_loc(loc_key)

        if 'mean' not in self.data[loc_key].keys():
            all_data = np.array(self.data[loc_key]['values'])
            self.data[loc_key]['mean'] = np.mean(all_data)

        lxs = [
            # loc features
            *loc_key,
            # elevation & slope features
            self.data[loc_key]['elevation'],
            self.data[loc_key]['slope'],
        ]
        loc_vals = loc_vals - self.data[loc_key]['mean']
        ly = torch.tensor(loc_vals).float()

        lxs = np.array(lxs)
        lxs = torch.tensor(lxs).float()
        
        loc_rain = np.array(loc_rain)
        loc_rain = torch.tensor(loc_rain).float()

        nbxs = []
        es = []
        for _nb in nb:
            nbxs.append([
                # loc features
                *_nb,
                # elevation & slope features
                self.data[_nb]['elevation'],
                self.data[_nb]['slope'],
            ])

            if 'mean' not in self.data[_nb].keys():
                all_data = np.array(self.data[_nb]['values'])
                self.data[_nb]['mean'] = np.mean(all_data)
            
            es.append(self.data[_nb]['mean'])

        es = np.array(es)
        nb_vals = np.array(nb_vals)
        nb_vals = nb_vals - es.reshape(-1, 1)
        nby = torch.tensor(nb_vals).float()

        nbxs = np.array(nbxs)
        nbxs = torch.tensor(nbxs).float()
        
        nb_rain = np.array(nb_rain)
        nb_rain = torch.tensor(nb_rain).float()

        return lxs, ly, loc_rain, nbxs, nby, nb_rain
    
    def __len__(self):
        return len(self.using_keys)