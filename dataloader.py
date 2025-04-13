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
from scipy.signal import find_peaks

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
    

class WaterLevelDataset(Dataset):
    def __init__(self, path, neighbor_idx, n_points=1, train=False, strategy='fixed'): # ='tdata'
        self.n_points = n_points
        self.strategy = strategy

        sensors = [os.path.join(path, d) for d in os.listdir(path)]
        sensors.sort()

        src_sensors = [sensors[idx] for idx in neighbor_idx]
        dst_sensor = sensors[0]

        self.src_xy = [
            [50.9433484538354, 2.67218829689136],
            [50.9801372834513, 2.74188851184371],
            [50.91342278544826, 2.740715491046123],
            [51.01981875223706, 2.801108964697574],
            [50.9201498250921, 2.62306717051891],
            [51.0009785859662, 2.82874141859661],
            [50.92287013183257, 2.825129902359957],
            [50.9103217034162, 2.60800111173615],
            [50.91828617352338, 2.84176626642422],
            [50.98331265035559, 2.866412315226182],
            [50.96803117176623, 2.885084969193244],
            [50.93122339433107, 2.887343968299596],
            [50.91908182371857, 2.890628773829795],
        ]
        self.dst_xy = [50.9571484191633, 2.73619409474557]

        with open('00_rainfall_accu.pkl', 'rb') as f:
            self.rainfall_accu = pickle.load(f)

        self.times = None

        self.list_data_dst = []
        t, v = self.extract_values_from_file(dst_sensor)
        self.list_data_dst = v

        spike_periods = self.detect_peaks_and_custom_valleys(t, v)
        self.times = self.get_interesting_times(t, spike_periods)

        self.list_data_src = []
        for src in src_sensors:
            t, v = self.extract_values_from_file(src)
            self.list_data_src.append(v)

        if train:
            start = 0
            end = len(self.times) // 2
        else:
            start = 0#len(self.list_data_dst[0]) // 2
            end = len(self.times) - 1

        self.times = self.times[start:end]

    def get_interesting_times(self, times, spike_periods):
        merged_list = []
        for sp in spike_periods:
            start = sp['start_idx']
            end = sp['end_idx']
            new_list = [times[i] for i in range(start, end + 1)]
            merged_list = merged_list + [item for item in new_list if item not in merged_list]
        merged_list.sort()
        return merged_list

    def detect_peaks_and_custom_valleys(self, times, values, prominence=0.90, height_ratio=0.85):
        """
        Detects peaks and dynamically adjusts the valleys around them based on a relative height threshold.

        Args:
            times (list or array): List of timestamps.
            values (list or array): List of data values.
            prominence (float): Minimum prominence of peaks to consider.
            height_ratio (float): Percentage of peak height to define valleys.

        Returns:
            pd.DataFrame: DataFrame containing peaks and their custom valleys.
        """
        # Convert inputs to numpy arrays for processing
        values = np.array(values)

        # Detect peaks using scipy's find_peaks
        peaks, properties = find_peaks(values, prominence=prominence)

        # Dynamically define valleys around each peak
        max_peak = max(values[peak_idx] for peak_idx in peaks)
        valleys = []
        for peak_idx in peaks:
            peak_value = values[peak_idx]

            # Adjust height dynamically based on the peak value and max peak
            dynamic_height = height_ratio * (0.15 + 0.6 * max_peak / peak_value) * peak_value

            # Expand left valley
            start_idx = peak_idx
            while start_idx > 0 and values[start_idx] > dynamic_height:
                start_idx -= 1

            # Expand right valley
            end_idx = peak_idx
            while end_idx < len(values) - 1 and values[end_idx] > dynamic_height:
                end_idx += 1

            # Append results
            valleys.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'Peak': times[peak_idx],
                'Start': times[start_idx],
                'End': times[end_idx],
                'Peak_Value': peak_value
            })

        return valleys

    def extract_values_from_file(self, file_path):
        """
        Reads a file with a semi-colon delimiter and extracts values from the second column.
        
        Args:
            file_path (str): Path to the file to be read.
        
        Returns:
            list: A list of values from the second column.
        """
        values = []
        times = []
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                # Skip lines that are metadata or headers
                if len(row) > 1 and not row[0].startswith('#'):
                    times.append(row[0])
                    str_value = row[1].replace(',', '.')
                    str_value = str_value if str_value != '' else '-1'
                    values.append(float(str_value))  # Extract the value in the second column
        return times, values

    def __len__(self):
        return len(self.times)

    def __getitem__(self, idx):
        x = []
        xs = []
        ds = []
        tds = []
        for _ in range(self.n_points):
            src_idx = np.random.randint(0, len(self.list_data_src))
            rand_idx = np.random.randint(idx - 96 * 4, idx + 1)
            x.append(self.list_data_src[src_idx][rand_idx])
            d = math.sqrt((self.dst_xy[0] - self.src_xy[src_idx][0]) ** 2 + (self.dst_xy[1] - self.src_xy[src_idx][1]) ** 2)
            td = 1 / ((idx - rand_idx) + 1)
            xs.extend([self.list_data_src[src_idx][rand_idx], *self.src_xy[src_idx], td])

            ds.append(d)
            tds.append(td)

        x = torch.tensor(x).float()
        xs = torch.tensor(xs).float()
        ds = torch.tensor(ds).float()
        tds = torch.tensor(tds).float()
        y = torch.tensor([self.list_data_dst[idx]]).float()
        r = torch.tensor([self.rainfall_accu[idx * 3]]).float()
        t = (self.times[idx])

        return x, y, t, ds, tds, xs, r


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


class WaterDataset(Dataset):
    def retain_overlapping(self):
        data = self.data

        for k in data.keys():
            list_time = [set(data[k]['time']), *[set(data[nb]['time']) for nb in data[k]['neighbor'] if nb in data.keys()]]
            common_times = set.intersection(*list_time)
            
            data[k]['common_time'] = common_times
            data[k]['common_time_idx'] = [data[k]['time'].index(t) for t in common_times]
            
            data[k]['nb_common_time_idx'] = {}
            for nb in data[k]['neighbor']:
                if nb in data.keys():
                    data[k]['nb_common_time_idx'][nb] = [data[nb]['time'].index(t) for t in common_times]
        
        self.data = data

    def retain_top_overlapping(self):
        data = self.data

        for k in data.keys():
            list_time = [set(data[k]['top_25_time']), *[set(data[nb]['top_25_time']) for nb in data[k]['neighbor'] if nb in data.keys()]]
            common_times = set.intersection(*list_time)
            common_times = list(common_times)
            common_times.sort()

            data[k]['common_time'] = common_times
            data[k]['common_time_idx'] = [data[k]['top_25_time'].index(t) for t in common_times]

            data[k]['nb_common_time_idx'] = {}
            for nb in data[k]['neighbor']:
                if nb in data.keys():
                    data[k]['nb_common_time_idx'][nb] = [data[nb]['top_25_time'].index(t) for t in common_times]
        
        self.data = data
    
    def __init__(self, path, train=False, selected_stations=None, input_type='pte', length=12*8, n_neighbors=4): # ='tdata'
        self.train = train
        self.nb = selected_stations
        self.input_type = input_type
        self.length = length
        self.n_neighbors = n_neighbors

        self.tz = pytz.FixedOffset(60)  # 120 minutes = 2 hours
        self.tz2 = pytz.FixedOffset(120)  # 120 minutes = 2 hours
        start_dt = datetime(year=2023, month=12, day=10, hour=0, tzinfo=self.tz2)

        # sensors = [os.path.join(path, d) for d in os.listdir(path) if 'csv' in d]
        # sensors.sort()

        # locations = [os.path.join(path, d) for d in os.listdir(path) if 'pkl' in d]
        # locations.sort()

        with open(path, 'rb') as f:
            self.data = pickle.load(f)

        if selected_stations is None:
            selected_stations = list(self.data.keys())

        # self.retain_top_overlapping()
        # print('Done loading: ', len(self.data.keys()))
        # with open('all_data_new_median.pkl', 'wb') as f:
        #     pickle.dump(self.data, f)
        # print('Done saving: ', len(self.data.keys()))
        # exit()

        # self.get_proper_time()
        # print('Done loading: ', len(self.data.keys()))
        # with open('all_data_new5.pkl', 'wb') as f:
        #     pickle.dump(self.data, f)
        # print('Done saving: ', len(self.data.keys()))
        # exit()

        # self.data = {}
        # for s, l in zip(sensors, locations):
        #     with open(l, 'rb') as f:
        #         data = pickle.load(f)
        #         latitude, longitude = data['latitude'], data['longitude']

        #     if (latitude, longitude) not in subset_locations:
        #         continue

        #     print(f'{s}-{l}: {(latitude, longitude)}')

        #     ts, vs = self.extract_values_from_file(s)
        #     if ts is None:
        #         continue
            
        #     nts = []
        #     for t in ts:
        #         splitted = t.split('T')
        #         date = splitted[0].split('-')
        #         hours = splitted[1].split('.')[0].split(':')

        #         year = int(date[0])
        #         month = int(date[1])
        #         day = int(date[2])
        #         hour = int(hours[0])
        #         minute = int(hours[1])

        #         if minute != 0:
        #             continue

        #         dt = datetime(
        #             year=year, month=month, day=day, hour=hour, tzinfo=self.tz2
        #       print  )
        #         nts.append(dt)
        #     # print(len(ts))
        #     # '2024-12-06T14:00:00.000+01:00', '2024-12-06T14:15:00.000+01:00'
        #     # print(nts[0], start_dt, nts[0] >= start_dt)
        #     for idx, t in enumerate(nts):
        #         if t >= start_dt:
        #             break
        #     ts = ts[idx:]
        #     vs = vs[idx:]
        #     # vs = [v for t, v in zip(nts, vs) if t >= start_dt]
        #     # ts = [t for t in nts if t >= start_dt]

        #     self.data[(latitude, longitude)] = {
        #         'neighbor': neighbor_dict[(latitude, longitude)],
        #         'series': vs,
        #         'time': ts,
        #         'rainfall': self.rainmap[(latitude, longitude)]
        #     }

        # with open('all_data.pkl', 'wb') as f:
        #     pickle.dump(self.data, f)
        # exit()
        self.set_nb(selected_stations)

    def get_proper_time(self):
        for k in self.data.keys():
            self.data[k]['series'] = [v for v, t in zip(self.data[k]['series'], self.data[k]['time']) if t[14:19] == '00:00']
            self.data[k]['time'] = [t for t in self.data[k]['time'] if t[14:19] == '00:00']

    def set_nb(self, nb):
        if self.nb is not None:
            list_keys = list(self.data.keys())
            for k in list_keys:
                if k not in nb:
                    del self.data[k]

            self.nb = nb
            for loc in self.nb:
                new_nb = [nb_ for nb_ in self.data[loc]['neighbor'] if nb_ in self.data.keys()]
                self.data[loc]['neighbor'] = new_nb

    def extract_values_from_file(self, file_path):
        """
        Reads a file with a semi-colon delimiter and extracts values from the second column.
        
        Args:
            file_path (str): Path to the file to be read.
        
        Returns:
            list: A list of values from the second column.
        """
        values = []
        times = []
        with open(file_path, mode='r', encoding='utf-8') as f:
            reader = csv.reader(f, delimiter=';')
            for row in reader:
                if 'ts_name' in row and 'Pv.15' not in row and 'P.15' not in row:
                    return None, None
                # Skip lines that are metadata or headers
                if len(row) > 1 and not row[0].startswith('#'):
                    times.append(row[0])
                    str_value = row[1].replace(',', '.')
                    str_value = str_value if str_value != '' else '-1'
                    values.append(float(str_value))  # Extract the value in the second column
        return times, values

    def detect_peaks_and_custom_valleys(self, times, values, prominence=0.90, height_ratio=0.85):
        """
        Detects peaks and dynamically adjusts the valleys around them based on a relative height threshold.

        Args:
            times (list or array): List of timestamps.
            values (list or array): List of data values.
            prominence (float): Minimum prominence of peaks to consider.
            height_ratio (float): Percentage of peak height to define valleys.

        Returns:
            pd.DataFrame: DataFrame containing peaks and their custom valleys.
        """
        # Convert inputs to numpy arrays for processing
        values = np.array(values)

        # Detect peaks using scipy's find_peaks
        peaks, properties = find_peaks(values, prominence=prominence)

        # Dynamically define valleys around each peak
        max_peak = max(values[peak_idx] for peak_idx in peaks)
        valleys = []
        for peak_idx in peaks:
            peak_value = values[peak_idx]

            # Adjust height dynamically based on the peak value and max peak
            dynamic_height = height_ratio * (0.15 + 0.6 * max_peak / peak_value) * peak_value

            # Expand left valley
            start_idx = peak_idx
            while start_idx > 0 and values[start_idx] > dynamic_height:
                start_idx -= 1

            # Expand right valley
            end_idx = peak_idx
            while end_idx < len(values) - 1 and values[end_idx] > dynamic_height:
                end_idx += 1

            # Append results
            valleys.append({
                'start_idx': start_idx,
                'end_idx': end_idx,
                'Peak': times[peak_idx],
                'Start': times[start_idx],
                'End': times[end_idx],
                'Peak_Value': peak_value
            })

        return valleys

    def __len__(self):
        return len(self.data.keys()) if not self.train else len(self.data.keys()) * 10

    def get_training_item(self):
        all_x = []
        all_xs = []
        all_ys = []
        for _, loc_key in enumerate(self.nb):
            ys = []
            lx = []
            lxs = []
            
            for _ in range(8):
                sample_idx = np.random.randint(0, len(self.data[loc_key]['common_time']))
                v = self.data[loc_key]['top_25_series'][
                    self.data[loc_key]['common_time_idx'][sample_idx]
                ]

                nb = self.data[loc_key]['neighbor']
                nb_keys = [nb_ for nb_ in nb if nb_ in self.nb]
                nb = [nb_ for nb_ in nb if nb_ in nb_keys]

                x = []
                xs = []
                
                for idx in range(self.n_points):
                    src_loc = nb[idx % len(nb)]

                    max_idx = len(self.data[src_loc]['top_25_series']) - self.data[loc_key]['nb_common_time_idx'][src_loc][sample_idx] - 1
                    delta = np.random.randint(-max_idx, 96)
                    rand_idx = self.data[loc_key]['nb_common_time_idx'][src_loc][sample_idx] - delta
                    sv = self.data[src_loc]['top_25_series'][rand_idx]

                    x.append(sv)
                    xs = self.get_neighbor_input(loc_key, src_loc, xs, sv, delta)

                paired = list(zip(x, xs))
                random.shuffle(paired)
                x, xs = zip(*paired)

                lx.append(x)
                lxs.append(xs)
                ys.append(v)

            all_x.append(lx)
            all_xs.append(lxs)
            all_ys.append(ys)

        all_x = torch.tensor(np.array(all_x)).float()
        all_xs = torch.tensor(np.array(all_xs)).float()
        all_ys = torch.tensor(np.array(all_ys)).float()

        if self.noise:
            all_c = torch.zeros_like(all_x)
            for i in range(all_x.shape[0]):
                for j in range(all_x.shape[1]):
                    for k in range(len(all_x[i, j])):
                        if random.random() > 0.75:
                            all_x[i, j, k] += np.random.normal(0, 0.20)
                            all_c[i, j, k] = 0.2
                        elif random.random() > 0.50:
                            all_x[i, j, k] += np.random.normal(0, 0.10)
                            all_c[i, j, k] = 0.1
                        elif random.random() > 0.25:
                            all_x[i, j, k] += np.random.normal(0, 0.05)
                            all_c[i, j, k] = 0.05
            all_xs = torch.cat([all_xs, all_c.unsqueeze(-1)], dim=-1)

        return all_x, all_xs, all_ys

    def get_neighbor_input(self, loc_key, src_loc, xs, sv, dt):
        dt = dt / 96

        # is_connected = int(self.data[loc_key]['neighbor_stats']['is_connected'][src_loc])
        distance = self.data[loc_key]['neighbor_stats']['distance'][src_loc]
        is_connected = self.data[loc_key]['neighbor_stats']['is_connected'][src_loc]
        distance = -1 if distance is None else distance / 100
        bidistance = self.data[loc_key]['neighbor_stats']['bidistance'][src_loc]
        bidistance = -1 if bidistance is None else bidistance / 100

        is_connected = 1 if is_connected else 0

        dp = np.array(loc_key) - np.array(src_loc)
        
        if self.input_type == 'p':
            xs.append([*gps_encode(loc_key), *gps_encode(src_loc)])
        elif self.input_type == 'pt':
            xs.append([*gps_encode(loc_key), *gps_encode(src_loc), dt])
        elif self.input_type == 'ptn':
            xs.append([*gps_encode(loc_key), *gps_encode(src_loc), dt])
        elif self.input_type == 'pte':
            xs.append([*gps_encode(loc_key), *gps_encode(src_loc), dt, c1, c2, w1 / 66, w2 / 66, distance, bidistance, is_connected])
        elif self.input_type == 'ptev':
            xs.append([sv, *gps_encode(loc_key), *gps_encode(src_loc), dt, c1, c2, w1 / 66, w2 / 66, distance, bidistance, is_connected])
        elif self.input_type == 'ptev':
            xs.append([sv, *gps_encode(loc_key), *gps_encode(src_loc), dt, c1, c2, w1 / 66, w2 / 66, distance, bidistance, is_connected])
        elif self.input_type == 'shortv':
            xs.append([sv, dt, w1 / 66, w2 / 66, distance, bidistance, h1, h2])
        elif self.input_type == 'short':
            xs.append([dt, w1 / 66, w2 / 66, distance, bidistance, h1, h2])
        elif self.input_type == 'update':
            xs.append([sv, dt, w1 / 66, w2 / 66, distance, bidistance, *dp, h1, h2, mean_w, std_w, max_zw])  # *dp, 
        return xs

    def get_testing_item(self, idx):
        loc_key = list(self.data.keys())[idx]

        sample_idx = random.randint(0, len(self.data[loc_key]['time']) - self.length - 1)
        loc_time = self.data[loc_key]['time'][sample_idx:sample_idx+self.length]
        loc_vals = self.data[loc_key]['values'][sample_idx:sample_idx+self.length]

        nb = self.data[loc_key]['sim_graph']['nodes']
        nb = [nb_ for nb_ in nb if nb_ in self.data.keys()]

        x = []
        xs = []
        for _ in range(self.n_neighbors):
            _nb = random.choice(nb)

            time_to_idx_nb = {t: idx for idx, t in enumerate(self.data[_nb]['time'])}
            idx_matches = [time_to_idx_nb.get(t, -1) for t in loc_time]
            nb_values = [self.data[_nb]['values'][idx] if idx != -1 else -1 for idx in idx_matches]

            x.append(nb_values)
            xs = self.get_neighbor_input(loc_key, src_loc, xs, sv, delta)

        paired = list(zip(x, xs))
        random.shuffle(paired)
        x, xs = zip(*paired)
        x = np.array(x)
        xs = np.array(xs)

        x = torch.tensor(x).float()
        xs = torch.tensor(xs).float()
        y = torch.tensor([v]).float()

        if self.noise:
            c = torch.zeros_like(x)
            for k in range(len(x)):
                if random.random() > 0.75:
                    x[k] += np.random.normal(0, 0.20)
                    c[k] = 0.2
                elif random.random() > 0.50:
                    x[k] += np.random.normal(0, 0.10)
                    c[k] = 0.1
                elif random.random() > 0.25:
                    x[k] += np.random.normal(0, 0.05)
                    c[k] = 0.05
            xs = torch.cat([xs, c.unsqueeze(-1)], dim=-1)

        return x, xs, y

    def __getitem__(self, idx):
        if self.train:
            return self.get_training_item()
        return self.get_testing_item(idx)


class GWaterDataset(WaterDataset):
    def get_input_vector_from_node(self, k, idx):
        v = self.data[k]['top_25_series'][
            idx
        ]
        c = FCLASS[self.data[k]['fclass']]
        w = self.data[k]['width']
        loc = gps_encode(k)

        return v, [v, c, w, *loc]
    
    def get_input_vector_from_path(self, k, nb, idx):
        loc = gps_encode(self.data[k]['neighbor_path'][nb]['nodes'][idx])
        c = FCLASS[self.data[k]['neighbor_path'][nb]['fclass'][idx]]
        w = self.data[k]['neighbor_path'][nb]['width'][idx]
        v = -1

        return [v, c, w, *loc]
    
    def find_edge(self, src, edges):
        for e in edges:
            if src == e[0]:
                return e
            
    def get_list_loc_and_time(self, loc_key, sample_idx, nb):
        list_src_time_pairs = set()
        src_loc_time_count = {}
        list_src_locs = []
        list_time = []

        attempts = 0
        max_attempts = self.n_points * 20  # Increased attempts to ensure filling up

        while len(list_src_locs) < self.n_points and attempts < max_attempts:
            attempts += 1

            idx = np.random.randint(0, len(nb))
            src_loc = nb[idx % len(nb)]

            delta = np.random.randint(-96, 96)
            time_rand_idx = min(
                self.data[loc_key]['nb_common_time_idx'][src_loc][sample_idx] - delta,
                max(self.data[loc_key]['nb_common_time_idx'][src_loc])
            )

            pair = (src_loc, time_rand_idx)

            # Ensure no duplicate pairs and at most 3 time idx per src_loc
            if pair not in list_src_time_pairs:
                if src_loc not in src_loc_time_count:
                    src_loc_time_count[src_loc] = []

                if len(src_loc_time_count[src_loc]) < 3:
                    src_loc_time_count[src_loc].append(time_rand_idx)
                    src_loc_time_count[src_loc].sort()

                    list_src_time_pairs.add(pair)
                    list_src_locs.append(src_loc)
                    list_time.append(time_rand_idx)

                else:
                    # Replace if the new time_rand_idx is closer to 0 (smaller)
                    if time_rand_idx < src_loc_time_count[src_loc][-1]:
                        removed_time_idx = src_loc_time_count[src_loc].pop()
                        src_loc_time_count[src_loc].append(time_rand_idx)
                        src_loc_time_count[src_loc].sort()

                        list_src_time_pairs.remove((src_loc, removed_time_idx))
                        removal_idx = next((i for i, (l, t) in enumerate(zip(list_src_locs, list_time)) if l == src_loc and t == removed_time_idx), None)
                        if removal_idx is not None:
                            del list_src_locs[removal_idx]
                            del list_time[removal_idx]

                        list_src_time_pairs.add(pair)
                        list_src_locs.append(src_loc)
                        list_time.append(time_rand_idx)

        return list_src_locs, list_time

    def __getitem__(self, idx):
        station_idx = 0
        sample_idx = idx
        while sample_idx > self.n_samples[station_idx]:
            sample_idx -= self.n_samples[station_idx]
            station_idx += 1

        loc_key = self.nb[station_idx]

        if sample_idx >= len(self.data[loc_key]['common_time']):
            sample_idx = np.random.randint(0, len(self.data[loc_key]['common_time']))
        time_sample_idx = self.data[loc_key]['common_time_idx'][sample_idx]

        nb = self.data[loc_key]['neighbor']
        if self.train:
            nb_keys = [nb_ for nb_ in nb if nb_ in self.nb]
            nb = [nb_ for nb_ in nb if nb_ in nb_keys]

        v = self.data[loc_key]['top_25_series'][time_sample_idx]
        vs = [v]

        ndict = {loc_key: {'idx': 0, 'time': [], 'f': [[0, 0], [0, 0], [0, 0]]}}
        edict = {}
        edge_list = []

        list_src_locs, list_time = self.get_list_loc_and_time(loc_key, sample_idx, nb)

        for src_loc, time_rand_idx in zip(list_src_locs, list_time):
            dt = get_time_difference_in_hours(
                self.data[loc_key]['top_25_time'][time_sample_idx],
                self.data[src_loc]['top_25_time'][time_rand_idx]
            ) / 96
            v_src = self.data[src_loc]['top_25_series'][time_rand_idx]
            ndict = self.update_node_dict(ndict, src_loc, dt, v_src)

            if self.data[loc_key]['neighbor_path'][src_loc]['edges'] is not None:
                start = src_loc
                flag = True
                while flag:
                    edges = self.data[loc_key]['neighbor_path'][src_loc]['edges'][::-1]

                    area = 0
                    distance = 0
                    nev_distance = 0
                    new_e = self.find_edge(start, self.data[loc_key]['graph']['edges'])

                    if new_e[1] != loc_key and new_e[1] not in ndict.keys():
                        ndict = self.update_node_dict(ndict, new_e[1], dt, v_src)

                    if new_e[1] == loc_key:
                        if start == src_loc:
                            idx = 0
                        else:
                            idx = self.data[loc_key]['neighbor_path'][src_loc]['nodes'].index(
                                (start[1], start[0])
                            )
                        for _i in range(max(idx - 1, 0), len(self.data[loc_key]['neighbor_path'][src_loc]['area'])):
                            area += self.data[loc_key]['neighbor_path'][src_loc]['area'][_i]
                        
                        flow = True
                        last_node = None
                        for _i in range(max(idx - 1, 0), len(edges)):
                            if last_node is not None and last_node != e[0]:
                                flow = not flow
                            
                            e = edges[_i]
                            d = abs(e[0][1] - e[1][1]) + abs(e[0][0] - e[1][0])
                            if flow:
                                distance += d
                            else:
                                nev_distance += d
                        
                        flag = False

                    elif (start[1], start[0]) not in self.data[loc_key]['neighbor_path'][src_loc]['nodes']:
                        idx = self.data[loc_key]['neighbor_path'][src_loc]['nodes'].index((new_e[1][1], new_e[1][0]))
                        for _i in range(idx - 1):
                            area += self.data[loc_key]['neighbor_path'][src_loc]['area'][_i]

                        flow = True
                        last_node = None
                        for _i in range(idx - 1):
                            if last_node is not None and last_node != e[0]:
                                flow = not flow
                            
                            e = edges[_i]
                            d = abs(e[0][1] - e[1][1]) + abs(e[0][0] - e[1][0])
                            if flow:
                                distance += d
                            else:
                                nev_distance += d
                    else:
                        sidx = self.data[loc_key]['neighbor_path'][src_loc]['nodes'].index((start[1], start[0]))
                        eidx = self.data[loc_key]['neighbor_path'][src_loc]['nodes'].index((new_e[1][1], new_e[1][0]))
                        for _i in range(sidx - 1, eidx):
                            area += self.data[loc_key]['neighbor_path'][src_loc]['area'][_i]

                        flow = True
                        last_node = None
                        for _i in range(sidx - 1, eidx):
                            if last_node is not None and last_node != e[0]:
                                flow = not flow
                            
                            e = edges[_i]
                            d = abs(e[0][1] - e[1][1]) + abs(e[0][0] - e[1][0])
                            if flow:
                                distance += d
                            else:
                                nev_distance += d

                    feats = [area / (distance + 0.01), distance / 100, nev_distance / 100, new_e[1][0] - new_e[0][0] / 100, new_e[1][1] - new_e[0][1] / 100, new_e[1][0] - loc_key[0] / 100, new_e[1][1] - loc_key[1] / 100]
                    edict[(start, new_e[1])] = feats

                    start = new_e[1]
            else:
                feats = [-10, -10, -10, -10, -10, -10, -10]
                edict[(src_loc, loc_key)] = feats

        # Convert node dictionary to tensor
        node_list = list(ndict.keys())
        for n in node_list:
            if len(ndict[n]['f']) == 1:
                ndict[n]['f'].extend([[0, -1], [0, -1]])
            elif len(ndict[n]['f']) == 2:
                ndict[n]['f'].append([0, -1])
        x = torch.tensor(np.array([ndict[n]['f'][:3] for n in node_list]), dtype=torch.float)
        x = x.view(-1, 6)

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
        
        for i, k in enumerate(list(ndict.keys())):
            if k != loc_key:
                if k in self.data.keys():
                    time_rand_idx = self.data[loc_key]['nb_common_time_idx'][k][sample_idx]
                    v = self.data[k]['top_25_series'][time_rand_idx]
                else:
                    v = -1
                vs.append(v)
        return torch.tensor(vs), graph_data

    def update_node_dict(self, ndict, src_loc, dt, v_src):
        if src_loc not in ndict.keys():
            ndict[src_loc] = {'idx': len(ndict.keys()), 'time': [dt], 'f': [[v_src, dt]]}
        elif dt in ndict[src_loc]['time']:
            idx = ndict[src_loc]['time'].index(dt)
            if ndict[src_loc]['f'][idx][0] == -1:
                ndict[src_loc]['f'][idx] = [v_src, dt]
        elif len(ndict[src_loc]['f']) > 0:
            ndict[src_loc]['f'].append([v_src, dt])
            ndict[src_loc]['f'].sort(key=lambda x: abs(x[1]))
            ndict[src_loc]['time'] = [item[1] for item in ndict[src_loc]['f']]
        return ndict
    


def visualize_edge_index(edge_index, directed=True, title="Graph Visualization"):
    """
    Visualize a graph from a PyTorch-style edge_index tensor using matplotlib and networkx.

    Parameters:
        edge_index (torch.Tensor): Tensor of shape [2, num_edges] representing edges.
        directed (bool): Whether to treat the graph as directed.
        title (str): Title for the plot.
    """
    assert edge_index.shape[0] == 2, "edge_index must be of shape [2, num_edges]"
    
    # Choose graph type
    G = nx.DiGraph() if directed else nx.Graph()

    # Convert tensor to list of edge tuples
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    G.add_edges_from(edges)

    # Layout and drawing
    plt.figure(figsize=(6, 4))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(
        G, pos,
        with_labels=True,
        node_color='lightcoral',
        edge_color='gray',
        node_size=1000,
        font_size=14,
        arrows=directed
    )
    plt.title(title)
    plt.show()