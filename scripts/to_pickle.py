import csv
import pickle

from pathlib import Path


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


def get_all_sensor_locations():
    file_paths = get_all_files('data/raw', ext='pkl')

    all_dicts = {}
    for k, fp in enumerate(file_paths):
        with open(fp, 'rb') as f:
            data = pickle.load(f)
            date_time, values = extract_values_from_file(fp.replace('pkl', 'csv'))
            data['time'] = date_time
            data['values'] = values
            all_dicts[(data['latitude'], data['longitude'])] = data


    with open('data/raw.pkl', 'wb') as f:
        pickle.dump(all_dicts, f)


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


get_all_sensor_locations()