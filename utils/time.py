import pandas as pd

from datetime import timedelta


def get_slices(loaded_data, k, days_period):
    time = loaded_data[k]['time']
    values = loaded_data[k]['values']

    df = pd.DataFrame({'value': values}, index=pd.to_datetime(time))

    # Extract 30-day window
    start_time = df.index[1000]  # or random.choice(df.index)
    end_time = start_time + timedelta(days=days_period)
    
    return df[start_time:end_time]
