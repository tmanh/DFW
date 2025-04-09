import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Union, List


def save_time_series_plot(
    timestamps: Union[List[str], List[pd.Timestamp]],
    values: List[float],
    output_path: str = 'time_series_plot.png',
    title: str = 'Time Series Plot',
    xlabel: str = 'Time',
    ylabel: str = 'Value'
):
    """
    Save a time series plot to an image file.

    Parameters:
    - timestamps: List of timestamps (strings or pd.Timestamp)
    - values: List of numerical values
    - output_path: Path to save the image file (e.g. 'plot.png')
    - title: Plot title
    - xlabel: Label for the x-axis
    - ylabel: Label for the y-axis
    """
    # Convert string timestamps to pandas datetime if needed
    timestamps = pd.to_datetime(timestamps, utc=True)

    plt.figure(figsize=(10, 5))
    plt.plot(timestamps, values, marker='o')
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    print(f"Saved plot to {output_path}")


with open('data/raw.pkl', 'rb') as f:
    loaded_data = pickle.load(f)


for k in loaded_data.keys():
    time = loaded_data[k]['time']
    values = loaded_data[k]['values']

    save_time_series_plot(time, values)
    input('Press Enter to continue:')