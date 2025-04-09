import numpy as np


def create_neighboring_dict(loc_dict, k=10):
    """
    Create a neighboring dictionary where the key is the keys of loc_dict
    and the values are the 10 closest keys in loc_dict.

    Args:
        loc_dict: Dictionary with keys as (latitude, longitude) tuples.
        k: Number of closest neighbors to find for each key.

    Returns:
        A dictionary where the keys are the same as loc_dict, and the values are
        lists of 10 closest keys.
    """
    keys = list(loc_dict.keys())
    coords = np.array(keys)  # Convert keys to a numpy array for easier distance computation
    neighbor_dict = {}

    for i, key in enumerate(keys):
        distances = np.linalg.norm(coords - coords[i], axis=1)  # Compute distances to all other keys
        closest_indices = np.argsort(distances)[1:k+1]  # Sort distances and get indices of 10 closest, excluding itself
        neighbor_dict[key] = [keys[j] for j in closest_indices]
    
    return neighbor_dict
