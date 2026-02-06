from scipy.signal import savgol_filter
import pandas as pd
import pytz
import torch

import os
import csv
import math

from collections import defaultdict

import random
import pickle
import numpy as np

import networkx as nx

from geopy.distance import geodesic
from dateutil import parser
from torch.utils.data import Dataset

from torch_geometric.data import Data

import matplotlib.pyplot as plt
from scipy.signal import medfilt

import geohash


def suppress_spike_segments(
    signal,
    z_thresh=6.0,
    max_segment_len=8,
    context=5
):
    """
    Suppress short contiguous spike segments using local median.

    Parameters
    ----------
    signal : 1D np.ndarray
    z_thresh : float
        MAD-based outlier threshold
    max_segment_len : int
        Maximum length of spike segment to correct
    context : int
        Number of neighbors on each side for median replacement

    Returns
    -------
    filtered : np.ndarray
    """
    x = np.array(signal.copy())
    n = len(x)

    # Robust stats
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-8
    z = np.abs(x - med) / mad

    is_outlier = z > z_thresh

    i = 0
    while i < n:
        if not is_outlier[i]:
            i += 1
            continue

        # Find contiguous segment
        start = i
        while i < n and is_outlier[i]:
            i += 1
        end = i  # [start, end)

        seg_len = end - start

        # Only fix *short* pathological bursts
        if seg_len <= max_segment_len:
            left = max(0, start - context)
            right = min(n, end + context)

            neighborhood = np.concatenate([
                x[left:start],
                x[end:right]
            ])

            if len(neighborhood) > 0:
                replacement = np.median(neighborhood)
                x[start:end] = replacement

    return list(x)


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


def l2_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


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
            
        with open('data/data_with_width_segment.pkl', 'wb') as f:
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
            
        with open('data/data_with_width_segment_2.pkl', 'wb') as f:
            pickle.dump(data, f)
        exit()

    def create_segments3(self, data):
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
            
        with open('data/data_with_width_segment_2.pkl', 'wb') as f:
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
                    break

    def show_data(self):
        for k in self.data.keys():
            old_data = np.array(self.data[k]['values'])
            N, L = old_data.shape
            old_data = old_data.reshape(-1)

            # --- visualization ---
            plt.figure(figsize=(75, 5))
            plt.plot(old_data - np.median(old_data), label="Original", alpha=0.7)
            for _nb in self.data[k]['neighbor'].keys():
                tmp = np.array(self.data[_nb]['values']).reshape(-1)
                tmp -= np.median(tmp)
                plt.plot(tmp, label=f"Neighbor {_nb}", alpha=0.7)
            plt.title(f"Key={k}")
            plt.xlabel("Time step")
            plt.ylabel("Value")
            plt.legend()
            plt.tight_layout()
            plt.savefig(f"xem.png")
            plt.close()

            input(f'Stop {k}: ')

    def __init__(self, path, train=False, selected_stations=None, input_type='pte', length=12 * 3, n_neighbors=4): # ='tdata'
        self.train = train
        self.input_type = input_type
        self.length = length
        self.n_neighbors = n_neighbors

        # with open('data/selected_stats_rainfall_segment.pkl', 'rb') as f:
        with open('data/data_with_width.pkl', 'rb') as f:
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
            # (50.9761639541252, 3.50350061937995),
            # (50.967587, 3.46339),
            # (50.963005, 3.505905),
            # (50.752506, 3.6088760000000004),
            # (51.148039, 3.864885),      # too few data points
            # (51.14826076428264, 3.902723574459568),  # too few data points
            # (51.03393855766628, 3.551301116668707),
            (50.75034473379703, 3.979828442354752),
            (50.751208179845, 3.971109149875746),
            (50.75330774120651, 3.963098792810351),
            (50.963005, 3.505905),
            (50.967587, 3.46339),
            (50.9761639541252, 3.50350061937995),
            # train
            (50.752506, 3.6088760000000004),
            (50.7915540177442, 3.62257299330959),
            (50.87282736801505, 4.056547787684996),
            (50.80790167884659, 3.626698144970669),
            (50.8296965118308, 4.00959912439749),
            ###
            (50.984913, 4.156368),
            (50.805603, 4.166977),
            ###
            (50.76482189211035, 4.271663187398281),
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

        self.set_elevation(data)

        self.using_keys = []
        self.data = {}

        for k in data.keys():
            mean_val = np.median(data[k]['values'])
            new_values = []
            new_week_idx = []
            for i in range(len(data[k]['values'])):
                mean_shift = abs(data[k]['values'][i] - mean_val)
                diff = np.mean(mean_shift)
                if diff < 4 and np.max(np.diff(mean_shift)) < 0.5:
                    new_values.append(suppress_spike_segments(data[k]['values'][i]))
                    new_week_idx.append(data[k]['week_idx'][i])
                # else:
                #     print('Removed spike at ', k, i, diff)
            data[k]['values'] = new_values
            data[k]['week_idx'] = new_week_idx

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

        stage = 'train' if self.train else 'test'
        print(f'Final {stage} length: ', len(self.using_keys))

    def __len__(self):
        return len(self.using_keys)

    def get_neighbor_input(self, stats):
        # avg_elevation, std_elevation, max_z, mean_z, min_z, std_z
        distance = stats['distance']
        displacement = stats['displacement']
        delta_elevation = stats['key_elevation'] - stats['src_elevation']
        
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
                exit()

            if 'mean' not in self.data[loc_key].keys():
                all_data = np.array(self.data[loc_key]['values'])
                self.data[loc_key]['mean'] = np.median(all_data)

            loc_width = self.data[loc_key]['width']

            xs = []
            es = []
            for i in range(len(nb)):
                _nb = nb[i]

                nb_width = self.data[_nb]['width']

                if os.path.exists(f'data/tmp/stats-{loc_key}-{_nb}.pkl'):
                    with open(f'data/tmp/stats-{loc_key}-{_nb}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, _, _, _ = self.get_neighbor_input(stats)
                        direction = -1
                        _xs = [*_xs, direction, nb_width, loc_width, *loc_key, *_nb]
                elif os.path.exists(f'data/tmp/stats-{_nb}-{loc_key}.pkl'):
                    with open(f'data/tmp/stats-{_nb}-{loc_key}.pkl', 'rb') as f:
                        stats = pickle.load(f)
                        _xs, _, _, _ = self.get_neighbor_input(stats)
                        direction = 1
                        _xs = [*_xs, direction, nb_width, loc_width, *loc_key, *_nb]
                else:
                    displacement = np.array(_nb) - np.array(loc_key)
                    delta_elevation = self.data[_nb]['elevation'] - self.data[loc_key]['elevation']
                    _xs = [
                        # distance features
                        -1, displacement[0], displacement[1],
                        # elevation features
                        -1, -1, -1, -1, -1, delta_elevation,
                        # others
                        0,
                        nb_width, loc_width, *loc_key, *_nb
                    ]
                if 'mean' not in self.data[_nb].keys():
                    self.data[_nb]['mean'] = np.median(self.data[_nb]['values'])

                xs.append(_xs)
                es.append(self.data[_nb]['mean'])

            nb_vals = np.array(nb_vals)
            loc_vals = loc_vals - self.data[loc_key]['mean']

            if len(nb_vals) <= 0:
                # print('======>', loc_key)
                pass
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

        return x, xs, y, es, loc_rain, nb_rain, valid, torch.tensor(np.array(loc_key))

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
        # print('length: ', len(self.data[loc_key]['week_idx']))
        return loc_key


class GWaterDataset(WaterDataset):
    def build_trim_graph(self, loc_key):
        # Allowed “selected” nodes = loc_key + its neighbors
        nodes_to_keep = set([loc_key] + list(self.data[loc_key]['neighbor'].keys()))

        # Build full directed graph from sim_graph edges
        G = nx.DiGraph()
        
        # print('Building graph for: ', loc_key)
        if len(self.data[loc_key]['sim_graph']['edges'][0]) > 3:
            self.data[loc_key]['sim_graph']['edges'] = [
                (e[0], e[1], e[2]) for e in self.data[loc_key]['sim_graph']['edges']
            ] 
        for (src, dst, dir_flag) in self.data[loc_key]['sim_graph']['edges']:
            if dir_flag == 1:
                G.add_edge(src, dst)
            elif dir_flag == -1:
                G.add_edge(dst, src)
            else:
                # If dir_flag is 0 or missing, treat as bidirectional
                G.add_edge(src, dst)
                G.add_edge(dst, src)

        # Collect all nodes from connected components that contain selected nodes
        sub_nodes = set()
        for node in nodes_to_keep:
            if node in G:
                sub_nodes |= set(nx.node_connected_component(G.to_undirected(), node))

        G_trimmed = G.subgraph(sub_nodes).copy()

        selected_nodes = set(G_trimmed.nodes())
        selected_edges = set(G_trimmed.edges())

        new_selected_edges = []
        for (src, dst, dir_flag) in self.data[loc_key]['sim_graph']['edges']:
            if (src, dst) in selected_edges or (dst, src) in selected_edges:
                new_selected_edges.append((src, dst, dir_flag))
        
        return selected_nodes, new_selected_edges

    def _compute_delta(self, loc_key):
        v = np.asarray(self.data[loc_key]['values'], dtype=float)
        if v.size == 0:
            return None
        m = np.median(v)
        v0 = v - m
        return np.max(np.abs(v0)) - np.min(np.abs(v0))

    def __getitem__(self, idx):
        keep_running = True
        # ----------------------------
        # 1) Pick a valid location and prepare node-level series/features
        # ----------------------------
        while keep_running:
            loc_key = self.get_random_loc(idx)
            
            # print('loc_key: ', loc_key)
            nb, loc_vals, loc_rain, nb_vals, nb_rain = self.get_time_values_from_loc(loc_key)
            
            if nb is None:
                idx = random.randint(0, len(self) - 1)
                continue

            selected_nodes, selected_edges = self.build_trim_graph(loc_key)
            
            loc_width = self.data[loc_key]['width']
            updated = False

            # node dict: stores time series 'f', rainfall 'r', fixed features 'fx'
            node_list = [loc_key]
            ndict = {
                loc_key: {
                    'idx': 0,
                    'f': np.zeros_like(loc_vals).tolist(),
                    'r': loc_rain,
                    'fx': [
                        *gps_encode(loc_key),
                        self.data[loc_key]['elevation'] if 'elevation' in self.data[loc_key] else 0.0,
                        self.data[loc_key]['width'] if 'width' in self.data[loc_key] else 0.0,  # width placeholder
                    ],
                }
            }

            edict = {}          # (src_coord, dst_coord) -> edge feature vector
            earray = []
            es = []             # neighbor means
            valid = [np.zeros_like(loc_vals, dtype=np.int32).tolist()]
            good_nb = []
            src_values = np.array(nb_vals)
            nb_rain = np.array(nb_rain)

            if 'mean' not in self.data[loc_key]:
                self.data[loc_key]['mean'] = np.median(np.array(self.data[loc_key]['values']))

            # add neighbors
            for i_nb, _nb in enumerate(nb):
                nb_values = nb_vals[i_nb].copy()
                nb_width = self.data[_nb]['width']

                # edge features from cached stats if present
                _xs = self.load_edge_feats(loc_key, loc_width, _nb, nb_width)

                # ensure neighbor mean exists
                if 'mean' not in self.data[_nb]:
                    self.data[_nb]['mean'] = np.median(self.data[_nb]['values'])

                valid.append((nb_values > 0).astype(np.int32))
                es.append(self.data[_nb]['mean'])
                nb_values = nb_values - self.data[_nb]['mean']

                if not updated:
                    updated = True
                    loc_vals = loc_vals - self.data[loc_key]['mean']

                if self.train:
                    delta = np.max(np.abs(loc_vals)) - np.min(np.abs(loc_vals))
                    if delta <= 0.3 or delta >= 4:
                        # skip neighbor if unstable sample
                        idx = np.random.randint(0, len(self))  # re-pick idx
                        continue
                    # elif 0.3 < delta <= 0.6:
                    #     # Keep with 50% probability
                    #     if np.random.rand() > 0.5:
                    #         idx = np.random.randint(0, len(self))
                    #         continue

                # register neighbor node and this (nb -> loc_key) edge
                node_list.append(_nb)
                ndict = self.update_node_dict(ndict, _nb, nb_values, nb_rain[i_nb])
                edict[(_nb, loc_key)] = _xs
                earray.append(_xs)
                good_nb.append(_nb)

            if len(good_nb) <= 0:
                # print('======>', loc_key)
                idx = np.random.randint(0, len(self))  # re-pick idx
                continue
            else:
                keep_running = False

        # ----------------------------
        # 3) Ensure all trimmed nodes exist in ndict (add placeholders if needed)
        # ----------------------------
        T = len(loc_vals)  # time length
        for n in selected_nodes:
            if n not in ndict:
                ndict[n] = {
                    'idx': 0,
                    'f': [0.0] * T,                # no series -> zeros
                    'r': [0.0] * T,                # no rain -> zeros
                    'fx': [
                        *gps_encode(n),
                        self.data[n]['elevation'] if n in self.data.keys() and 'elevation' in self.data[n] else 0.0,
                        self.data[n]['width'] if n in self.data.keys() and 'width' in self.data[n] else 0.0,  # width placeholder
                    ],        # at least keep gps encoding
                }
                node_list.append(n)

        # consistent node order & index map
        node_to_idx = {node: i for i, node in enumerate(node_list)}

        # pack node tensors
        x = torch.tensor(np.array([ndict[n]['f'] for n in node_list]), dtype=torch.float)

        # unify rainfall length for any placeholders
        temp_len = None
        for n in node_list:
            if not isinstance(ndict[n]['r'], int):
                temp_len = len(ndict[n]['r'])
                break
        if temp_len is None:
            temp_len = T
        for n in node_list:
            if isinstance(ndict[n]['r'], int):
                ndict[n]['r'] = [0.0] * temp_len

        fx = torch.tensor(np.array([ndict[n]['fx'] for n in node_list]), dtype=torch.float)
        r  = torch.tensor(np.array([ndict[n]['r']  for n in node_list]), dtype=torch.float)

        # ----------------------------
        # 4) Edge list + attributes aligned to node_to_idx
        # ----------------------------
        # Feature dimension for edges (from any entry in edict, else 0)
        feat_dim = (len(next(iter(edict.values()))) if len(edict) else 0)

        edge_list = []
        edge_attr_list = []

        for u, v, d in selected_edges:
            if u in node_to_idx and v in node_to_idx:
                edge_list.append((node_to_idx[u], node_to_idx[v]))

                u_width = self.data[u]['width'] if u in self.data else 0.0
                v_width = self.data[v]['width'] if v in self.data else 0.0

                edge_feats = self.load_edge_feats(u, u_width, v, v_width)  # ensure stats are prepared
                edge_attr_list.append(edge_feats)

        edge_index = torch.tensor(edge_list, dtype=torch.long).T  # shape [2, E]
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float) if feat_dim > 0 else torch.zeros((len(edge_list), 0))

        # Optional edge-attr normalization (kept from your code; no-op if 0 dims)
        if edge_attr.numel() > 0:
            mask = edge_attr[:, 0] != -10
            for i in range(edge_attr.shape[1]):
                col = edge_attr[mask, i]
                if col.numel() == 0:
                    continue
                max_val = col.abs().max()
                if max_val > 0:
                    edge_attr[mask, i] = edge_attr[mask, i] / max_val
            edge_attr = 1 / (edge_attr + 1e-8)

        # ----------------------------
        # 5) Pack outputs
        # ----------------------------
        graph_data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)

        for _ in range(len(valid), x.shape[0]):
            valid.append(np.zeros_like(valid[0]))
        valid = torch.tensor(np.array(valid))
        es = torch.tensor(np.array(es)).float().unsqueeze(-1)
        src_values = torch.tensor(src_values).float() - es

        earray = torch.tensor(np.array(earray)).float()

        return (
            torch.tensor(loc_vals).float(),     # target series at loc_key (mean-removed)
            src_values.float(),                 # neighbor series (mean-removed)
            valid,                      # availability mask
            graph_data,                 # PyG graph
            fx,                         # node fixed features
            r,                          # node rainfall features
            torch.tensor(np.array(loc_key)).float(),
            earray
        )

    def load_edge_feats(self, loc_key, loc_width, _nb, nb_width):
        if os.path.exists(f'data/tmp/stats-{loc_key}-{_nb}.pkl'):
            with open(f'data/tmp/stats-{loc_key}-{_nb}.pkl', 'rb') as f:
                stats = pickle.load(f)
            _xs, _, _, _ = self.get_neighbor_input(stats) # direction -1
            _xs = [*_xs, -1, nb_width, loc_width]
        elif os.path.exists(f'data/tmp/stats-{_nb}-{loc_key}.pkl'):
            with open(f'data/tmp/stats-{_nb}-{loc_key}.pkl', 'rb') as f:
                stats = pickle.load(f)
            _xs, _, _, _ = self.get_neighbor_input(stats)  # direction +1
            _xs = [*_xs, 1, nb_width, loc_width]
        else:
            displacement = np.array(_nb) - np.array(loc_key)
            nb_elevation = self.data[_nb]['elevation'] if _nb in self.data and 'elevation' in self.data[_nb] else 0.0
            loc_elevation = self.data[loc_key]['elevation'] if loc_key in self.data and 'elevation' in self.data[loc_key] else 0.0
            delta_elev = nb_elevation - loc_elevation
            _xs = [
                -1, displacement[0], displacement[1],               # distance-ish
                -1, -1, -1, -1, -1, delta_elev,                     # elevation stats + delta
                0,                                                  # flow dir placeholder
                nb_width, loc_width,                                # coast placeholders
            ]
            
        return _xs
    
    def revert_edge_feats(self, edge_feats):
        new_edge_feats = [
            edge_feats[0], -edge_feats[1], -edge_feats[2],  # distance features
            # elevation features
            edge_feats[3], edge_feats[4], edge_feats[5], edge_feats[6], edge_feats[7],
            -edge_feats[8],
            # direction, widths
            -edge_feats[16], edge_feats[18], edge_feats[17]
        ]
        return new_edge_feats

    def update_node_dict(self, ndict, _nb, nb_values, nb_rain=None):
        if _nb not in ndict.keys():
            if nb_rain is None:
                ndict[_nb] = {
                    'idx': len(ndict.keys()),
                    'f': nb_values,
                }
            else:
                ndict[_nb] = {
                    'idx': len(ndict.keys()),
                    'f': nb_values,
                    'fx': [
                        *gps_encode(_nb),
                        self.data[_nb]['elevation'] if 'elevation' in self.data[_nb] else 0.0,
                        self.data[_nb]['width'] if 'width' in self.data[_nb] else 0.0,
                    ],
                    'r': nb_rain,
                }
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
            self.data[loc_key]['mean'] = np.median(all_data)

        xs = [
            # loc features
            *loc_key,
            # elevation features
            self.data[loc_key]['elevation'],
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
            self.data[loc_key]['mean'] = np.median(all_data)

        lxs = [
            # loc features
            *loc_key,
            # elevation features
            self.data[loc_key]['elevation'],
        ]
        loc_vals = loc_vals - self.data[loc_key]['mean']
        ly = torch.tensor(loc_vals).float()

        lxs = np.array(lxs)
        lxs = torch.tensor(lxs).float()
        
        loc_rain = np.array(loc_rain)
        loc_rain = torch.tensor(loc_rain).float()

        nbxs = []
        es = []
        for i, _nb in enumerate(nb):
            if 'mean' not in self.data[_nb].keys():
                all_data = np.array(self.data[_nb]['values'])
                self.data[_nb]['mean'] = np.median(all_data)

            nbxs.append([
                # loc features
                *_nb,
                # elevation features
                self.data[_nb]['elevation'],
            ])
            
            es.append(self.data[_nb]['mean'])

        es = np.array(es)
        nb_vals = np.array(nb_vals)
        nb_vals = nb_vals - es.reshape(-1, 1)
        nby = torch.tensor(nb_vals).float()

        nbxs = np.array(nbxs)
        nbxs = torch.tensor(nbxs).float()
        
        nb_rain = np.array(nb_rain)
        nb_rain = torch.tensor(nb_rain).float()

        return lxs, ly, loc_rain, nbxs, nby, nb_rain, torch.tensor(np.array(loc_key))
    
    def __len__(self):
       return len(self.using_keys)


class WaterDatasetYAllStations(WaterDataset):
    """
    Modified dataset that loads all available stations for KED training
    while keeping the same center station for consistent evaluation.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Pre-filter valid stations that have temporal overlap with others
        self._filter_valid_stations()
    
    def _filter_valid_stations(self):
        """
        Pre-filter stations to only include those with temporal overlap.
        This prevents runtime errors from stations with no common weeks.
        """
        print("Pre-filtering stations for temporal overlap...")
        valid_keys = []
        
        for loc_key in self.using_keys:
            if loc_key not in self.data:
                continue
            
            # Check if this station has overlap with at least one other station
            center_weeks = set(self.data[loc_key]['week_idx'])
            has_overlap = False
            
            for other_key in self.data.keys():
                if other_key == loc_key:
                    continue
                other_weeks = set(self.data[other_key]['week_idx'])
                if len(center_weeks.intersection(other_weeks)) > 0:
                    has_overlap = True
                    break
            
            if has_overlap:
                valid_keys.append(loc_key)
            else:
                print(f"  Filtering out station {loc_key}: no temporal overlap")
        
        original_count = len(self.using_keys)
        self.using_keys = valid_keys
        print(f"Filtered {original_count} → {len(self.using_keys)} stations ({original_count - len(self.using_keys)} removed)")
    
    def get_time_values_from_loc(self, loc_key):
        """
        Get all stations' data at common weeks defined by neighbors.
        
        Process:
        1. Find common weeks between center and its NEIGHBORS (original logic)
        2. For those common weeks, include ALL stations that have data (not just neighbors)
        
        This preserves the neighbor-based timeframe while maximizing training data.
        """
        # STEP 1: Get common weeks from neighbors (ORIGINAL LOGIC - DON'T CHANGE)
        nb = self.data[loc_key]['neighbor'].keys()
        nb = [nb_ for nb_ in nb if nb_ in self.data.keys() and nb_ != loc_key]
        
        if len(nb) == 0:
            return None, None, None, None, None
        
        common_weeks = set(self.data[loc_key]['week_idx'])
        for _nb in nb:
            nb_weeks = set(self.data[_nb]['week_idx'])
            common_weeks = common_weeks.intersection(nb_weeks)
        
        common_weeks = sorted(common_weeks)
        
        if common_weeks == []:
            return None, None, None, None, None
        
        selected_weeks = common_weeks
        
        # STEP 2: Get center station data (ORIGINAL LOGIC)
        loc_idx = [self.data[loc_key]['week_idx'].index(w) for w in selected_weeks]
        loc_vals = np.array([self.data[loc_key]['values'][i] for i in loc_idx]).flatten()
        loc_rain = np.array([self.data[loc_key]['r1x1'][i] for i in loc_idx]).flatten()
        
        # STEP 3: Find ALL stations (not just neighbors) that have data for ALL common_weeks
        all_stations = []
        for station_key in self.data.keys():
            if station_key == loc_key:
                continue
            
            station_weeks = set(self.data[station_key]['week_idx'])
            # Check if this station has data for ALL common weeks
            if set(selected_weeks).issubset(station_weeks):
                all_stations.append(station_key)
        
        if len(all_stations) == 0:
            return None, None, None, None, None
        
        # STEP 4: Get data for ALL stations at common_weeks
        all_vals = []
        all_rain = []
        for station in all_stations:
            station_idx = [self.data[station]['week_idx'].index(w) for w in selected_weeks]
            station_vals = np.array([self.data[station]['values'][i] for i in station_idx]).flatten()
            station_rain = np.array([self.data[station]['r1x1'][i] for i in station_idx]).flatten()
            all_vals.append(station_vals)
            all_rain.append(station_rain)
        
        return all_stations, loc_vals, loc_rain, all_vals, all_rain
    
    def __getitem__(self, idx):
        # Track attempted indices to prevent infinite recursion
        max_attempts = min(len(self), 100)  # Try up to 100 stations or all stations
        attempted = 0
        current_idx = idx
        
        while attempted < max_attempts:
            loc_key = self.get_random_loc(current_idx)
            all_stations, loc_vals, loc_rain, all_vals, all_rain = self.get_time_values_from_loc(loc_key)
            
            if all_stations is not None:
                break
            
            # Try next index
            current_idx = (current_idx + 1) % len(self)
            attempted += 1
        
        if all_stations is None:
            raise ValueError(
                f"Could not find valid data after {max_attempts} attempts. "
                f"Station {loc_key} has no common weeks with other stations. "
                f"Please check your data or filter out stations with no temporal overlap."
            )
        
        # Calculate center station mean if not cached
        if 'mean' not in self.data[loc_key].keys():
            all_data = np.array(self.data[loc_key]['values'])
            self.data[loc_key]['mean'] = np.median(all_data)
        
        # Center station features
        lxs = [
            # loc features
            *loc_key,
            # elevation features
            self.data[loc_key]['elevation'],
        ]
        loc_vals = loc_vals - self.data[loc_key]['mean']
        ly = torch.tensor(loc_vals).float()
        
        lxs = np.array(lxs)
        lxs = torch.tensor(lxs).float()
        
        loc_rain = np.array(loc_rain)
        loc_rain = torch.tensor(loc_rain).float()
        
        # All stations features and values
        all_xs = []
        all_means = []
        for i, station in enumerate(all_stations):
            if 'mean' not in self.data[station].keys():
                all_data = np.array(self.data[station]['values'])
                self.data[station]['mean'] = np.median(all_data)
            
            all_xs.append([
                # loc features
                *station,
                # elevation features
                self.data[station]['elevation'],
            ])
            
            all_means.append(self.data[station]['mean'])
        
        # Normalize all station values
        all_means = np.array(all_means)
        all_vals = np.array(all_vals)
        all_vals = all_vals - all_means.reshape(-1, 1)
        ally = torch.tensor(all_vals).float()
        
        all_xs = np.array(all_xs)
        all_xs = torch.tensor(all_xs).float()
        
        all_rain = np.array(all_rain)
        all_rain = torch.tensor(all_rain).float()
        
        return lxs, ly, loc_rain, all_xs, ally, all_rain, torch.tensor(np.array(loc_key))
    
    def __len__(self):
        return len(self.using_keys)
    

class GeoWaterDataset(WaterDataset):
    def __init__(self, path, train=False, selected_stations=None): # ='tdata'
        super().__init__(path, train, selected_stations)

        # ensure mean values exist
        for loc, d in self.data.items():
            vals = np.asarray(d['values']).flatten()
            d.setdefault('mean', np.median(vals))

        # --- Precompute: week -> list of (loc, pos) ---
        week_to_entries = defaultdict(list)
        for loc, d in self.data.items():
            weeks = np.asarray(d['week_idx']).flatten().astype(int)
            for pos, w in enumerate(weeks):
                week_to_entries[int(w)].append((loc, pos))

        self.weeks = sorted(week_to_entries.keys())
        self.week_entries = week_to_entries  # dictionary for fast lookup

    def __len__(self):
        return len(self.weeks)

    def __getitem__(self, idx):
        week = self.weeks[idx]
        entries = self.week_entries[week]

        x_list, es_list, rain_list, xs_list, keys_list = [], [], [], [], []
        y_list = []
        for loc, pos in entries:
            d = self.data[loc]

            y = np.asarray(d['values']).flatten()[pos]
            es   = np.asarray(d['values']).flatten()[pos]
            rain = np.asarray(d['r1x1']).flatten()[pos]

            elevation = d['elevation']
            width     = d['width']

            xs = [elevation, width]
            x  = es - d['mean']

            y_list.append(y)
            x_list.append(x)
            es_list.append(es)
            rain_list.append(rain)
            xs_list.append(xs)
            keys_list.append(loc)

        out = {
            "week": week,
            "loc_keys": torch.tensor(np.array(keys_list)),
            "x":   torch.tensor(x_list, dtype=torch.float32),
            "es":  torch.tensor(es_list, dtype=torch.float32),
            "rain":torch.tensor(rain_list, dtype=torch.float32),
            "xs":  torch.tensor(xs_list, dtype=torch.float32),  # (N,5)
            "y":   torch.tensor(y_list, dtype=torch.float32),
        }

        return out['x'], out['xs'], out['y'], out['es'], out['rain'], out['loc_keys']
