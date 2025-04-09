import sys
import pickle

sys.path.append('common')


from loc import create_neighboring_dict


with open('data/processed.pkl', 'rb') as f:
    loaded_data = pickle.load(f)


neighbor_dict = create_neighboring_dict(loaded_data)
with open('data/neighbor.pkl', 'wb') as f:
    pickle.dump(neighbor_dict, f)