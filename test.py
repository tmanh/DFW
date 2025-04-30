import numpy as np
import pandas as pd
from pykrige.ok import OrdinaryKriging

# Example sensor coordinates:
sensor_coords = {
    'Sensor1': (106.67, 10.76),
    'Sensor2': (106.68, 10.75),
    'Sensor3': (106.65, 10.74),
    'Sensor4': (106.66, 10.77)
}

# Load your time series data (example format):
df = pd.DataFrame({
    'Timestamp': pd.date_range('2024-04-01', periods=3, freq='H'),
    'Sensor1': [2.5, 2.6, 2.7],
    'Sensor2': [2.7, 2.8, 2.9],
    'Sensor3': [2.6, 2.7, 2.8],
    'Sensor4': [2.4, 2.5, 2.6]
})

# Target interpolation location:
target_lon, target_lat = 106.665, 10.755

# Initialize a list to store interpolated results:
interpolated_results = []

# Perform kriging interpolation at each timestamp
for idx, row in df.iterrows():
    values = row[['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']].values
    lons = [sensor_coords[s][0] for s in ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']]
    lats = [sensor_coords[s][1] for s in ['Sensor1', 'Sensor2', 'Sensor3', 'Sensor4']]
    
    values = np.array(values)
    lons = np.array(lons)
    lats = np.array(lats)

    print(values, values.dtype)
    print(lons, lons.dtype)
    print(lats, lats.dtype)
    exit()
    # Kriging
    OK = OrdinaryKriging(
        lons, lats, values,
        variogram_model='linear',
        verbose=False,
        enable_plotting=False,
    )
    
    # Interpolation at the target point
    z_interp, ss = OK.execute('points', [target_lon], [target_lat])
    
    interpolated_results.append({
        'Timestamp': row['Timestamp'],
        'Interpolated_Water_Level': z_interp[0]
    })

# Convert to DataFrame
interpolated_df = pd.DataFrame(interpolated_results)

print(interpolated_df)