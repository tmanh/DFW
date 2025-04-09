import pickle
import numpy as np
import pandas as pd


with open('data/raw.pkl', 'rb') as f:
    loaded_data = pickle.load(f)


for k in loaded_data.keys():
    time = loaded_data[k]['time']
    values = loaded_data[k]['values']

    # Convert to datetime (if not already) and create a DataFrame
    time = pd.to_datetime(loaded_data[k]['time'], utc=True)
    values = np.array(loaded_data[k]['values'], dtype=np.float32)

    # Store pre-parsed time and float values only
    loaded_data[k]['time'] = time.values  # NumPy datetime64
    loaded_data[k]['values'] = values


with open('data/processed.pkl', 'wb') as f:
    pickle.dump(loaded_data, f)
