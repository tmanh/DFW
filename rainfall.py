from dateutil import parser
import pickle

import numpy as np
import pandas as pd
import pytz


def check_rainfall_time_breaks(rainfall_df):
    """
    Check for time breaks in rainfall data and print info.
    Assumes 'rainfall_df' has a datetime index.
    """
    times_pd = rainfall_df.index.to_series()
    
    # Calculate time differences in hours
    deltas = (times_pd.shift(-1) - times_pd).dropna().dt.total_seconds() / 3600.0

    # Identify breaks where the gap is not 1 hour
    breaks = deltas[np.abs(deltas - 1.0) > 1e-3]

    print(f"Number of time breaks in rainfall data: {len(breaks)}")
    if not breaks.empty:
        print("Break times and gap sizes (in hours):")
        for timestamp, gap in breaks.items():
            print(f"  At {timestamp} → {gap:.2f} hours gap")
    else:
        print("No time breaks found (hourly resolution is consistent).")

    return breaks


def check_time_breaks_in_merged_dict(merged_dict):
    """
    Check for time breaks in a merged dictionary with 'time' as a datetime list.
    """
    times = pd.to_datetime(merged_dict['time'])
    deltas = (times[1:] - times[:-1]).total_seconds() / 3600.0  # in hours

    breaks = []
    for i, delta in enumerate(deltas):
        if abs(delta - 1.0) > 1e-3:
            breaks.append((times[i], delta))

    print(f"Number of time breaks: {len(breaks)}")
    if breaks:
        print("Break times and gap sizes (in hours):")
        for t, gap in breaks:
            print(f"  At {t} → {gap:.2f} hours gap")
    else:
        print("No breaks detected (hourly resolution is consistent).")

    return breaks


def check_15min_time_breaks(water_df):
    """
    Checks for time breaks in a DataFrame with 15-minute expected intervals.
    """
    times = water_df.index.to_series().sort_values()
    deltas = times.diff().dropna()

    expected_delta = pd.Timedelta(minutes=15)
    breaks = deltas[deltas != expected_delta]

    print(f"Number of time breaks: {len(breaks)}")
    if not breaks.empty:
        print("Breaks found at:")
        for t, delta in breaks.items():
            print(f"  At {t} → Gap of {delta}")
    else:
        print("No breaks detected (15-minute resolution is consistent).")

    return breaks


def align_and_filter(data, rainfall, k):
    # Step 1: Create rainfall DataFrame
    rainfall_df = pd.DataFrame({
        'time': pd.to_datetime([t.replace(tzinfo=None) for t in pd.to_datetime(list(rainfall[k].keys())).to_pydatetime()]),
        'r1x1': list(rainfall[k][t]['r1x1'] for t in rainfall[k].keys()),
        'r3x3': list(rainfall[k][t]['r3x3'] for t in rainfall[k].keys()),
        'r5x5': list(rainfall[k][t]['r5x5'] for t in rainfall[k].keys()),
        'r7x7': list(rainfall[k][t]['r7x7'] for t in rainfall[k].keys()),
        'r9x9': list(rainfall[k][t]['r9x9'] for t in rainfall[k].keys()),
    }).set_index('time').sort_index()
    
    check_rainfall_time_breaks(rainfall_df)

    water_df = pd.DataFrame({
        'time': pd.to_datetime([t.replace(tzinfo=None) for t in pd.to_datetime(data[k]['time']).to_pydatetime()]),
        'water_level': data[k]['values']
    }).set_index('time').sort_index()

    check_15min_time_breaks(water_df)

    # Step 3: Resample rainfall to daily to check completeness
    rainfall_daily = rainfall_df.resample('D').mean()

    # Step 4: Count days per month and expected days
    days_per_month = rainfall_daily.index.to_period('M').value_counts().sort_index()
    month_lengths = rainfall_daily.index.to_series().groupby(rainfall_daily.index.to_period('M')).apply(
        lambda x: x.index[-1].day
    )

    # Step 5: Keep only full months
    valid_months = days_per_month[days_per_month == month_lengths].index

    # Step 6: Filter rainfall and water level by valid months
    rainfall_valid = rainfall_df[rainfall_df.index.to_period('M').isin(valid_months)]
    water_valid = water_df[water_df.index.to_period('M').isin(valid_months)]

    # Step 7: Merge (inner join) on timestamps
    merged = water_valid.join(rainfall_valid, how='inner')

    merged_dict = {
        'time': merged.index.to_pydatetime().tolist(),
        'values': merged['water_level'].tolist(),
        'r1x1': merged['r1x1'].tolist(),
        'r3x3': merged['r3x3'].tolist(),
        'r5x5': merged['r5x5'].tolist(),
        'r7x7': merged['r7x7'].tolist(),
        'r9x9': merged['r9x9'].tolist(),
    }

    check_time_breaks_in_merged_dict(merged_dict)

    return merged_dict, valid_months.to_list()


def main():
    with open('data/rainfall2.pkl', 'rb') as f:
        rainfall = pickle.load(f)

    with open('data/selected_stats.pkl', 'rb') as f:
        data = pickle.load(f)

    new_data = {}
    for k in data.keys():
        if k not in rainfall.keys():
            print(k)
            continue
        
        merge_dict, _ = align_and_filter(data, rainfall, k)
        
        new_data[k] = data[k]
        new_data[k]['time'] = merge_dict['time']
        new_data[k]['values'] = merge_dict['values']
        new_data[k]['r1x1'] = merge_dict['r1x1']
        new_data[k]['r3x3'] = merge_dict['r3x3']
        new_data[k]['r5x5'] = merge_dict['r5x5']
        new_data[k]['r7x7'] = merge_dict['r7x7']
        new_data[k]['r9x9'] = merge_dict['r9x9']

    with open('data/selected_stats_rainfall.pkl', 'wb') as f:
        pickle.dump(new_data, f)


def main2():
    with open('data/rainfall2.pkl', 'rb') as f:
        rainfall = pickle.load(f)

    with open('data/selected_stats.pkl', 'rb') as f:
        data = pickle.load(f)

    new_data = {}
    for k in data.keys():
        merge_dict, _ = align_and_filter2(data, k)
        
        new_data[k] = data[k]
        new_data[k]['time'] = merge_dict['time']
        new_data[k]['values'] = merge_dict['values']

    with open('data/selected_stats_segment.pkl', 'wb') as f:
        pickle.dump(new_data, f)
    
main2()