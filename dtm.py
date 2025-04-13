import os
import time
import pickle
import numpy as np

from affine import Affine
from rasterio.merge import merge
from skimage.restoration import denoise_tv_chambolle
import richdem as rd
from scipy.ndimage import zoom

from common.dsm import *
from common.graph import *
from common.vis import *


def compute_waterway_elevation_statistics(cropped_dsm, waterway_mask):
    """
    Compute elevation statistics for the waterway portion of a cropped DSM.
    
    The function calculates statistics only for the pixels that are part of the waterway.
    It returns:
      - avg_elevation: Average elevation over the waterway.
      - std_elevation: Standard deviation of the elevation over the waterway.
      - max_z: Maximum z-score among waterway pixels.
      - mean_z: Mean z-score among waterway pixels.
      - min_z: Minimum z-score among waterway pixels.
      - std_z: Standard deviation of the z-scores.
    
    The z-score is computed as:
        z = (elevation - avg_elevation) / std_elevation
    over only the waterway pixels.
    
    :param cropped_dsm: 2D numpy array of elevation values.
    :param waterway_mask: Boolean 2D array (same shape as cropped_dsm) indicating the waterway.
    :return: Tuple (avg_elevation, std_elevation, max_z, mean_z, min_z, std_z)
    """
    # Select only the waterway elevations.
    waterway_elevations = cropped_dsm[waterway_mask]
    
    if waterway_elevations.size == 0:
        # If there are no waterway pixels, return zeros.
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    
    # Compute average and standard deviation for the waterway elevations.
    avg_elevation = np.mean(waterway_elevations)
    std_elevation = np.std(waterway_elevations)
    
    # Compute z-scores for waterway pixels.
    if std_elevation > 0:
        z_scores = (waterway_elevations - avg_elevation) / std_elevation
    else:
        z_scores = np.zeros_like(waterway_elevations)
    
    z_scores = np.abs(z_scores)
    
    max_z = np.max(z_scores)
    mean_z = np.mean(z_scores)
    min_z = np.min(z_scores)
    std_z = np.std(z_scores)
    
    return avg_elevation, std_elevation, max_z, mean_z, min_z, std_z


def create_dict(path):
    names = [d for d in os.listdir(path)]
    parts = [os.path.join(path, d, 'GeoTIFF', f'{d}.tif') for d in os.listdir(path)]
    rpj_parts = [os.path.join(path, d, 'GeoTIFF', f'{d}_rpj.tif') for d in os.listdir(path)]

    for p, rpj_p in zip(parts, rpj_parts):
        if not os.path.exists(rpj_p):
            reproject_dsm(p, rpj_p)

    dict_bounds = {}
    for n, rpj_p in zip(names, rpj_parts):
        _, _, bounds = load_dsm(rpj_p)
        min_lon, min_lat = bounds.left, bounds.bottom
        max_lon, max_lat = bounds.right, bounds.top
        dict_bounds[n] = [min_lon, min_lat, max_lon, max_lat]

        with open('data/bounds.pkl', 'wb') as f:
            pickle.dump(dict_bounds, f)


def load_and_merge_dsms(dsm_paths):
    """
    Load and merge only the required DSM tiles, preserving spatial alignment.
    
    :param dsm_paths: List of DSM file paths that need merging
    :return: Merged DSM array and its transform
    """
    dsm_data = []
    
    for dsm_path in dsm_paths:
        src = rasterio.open(dsm_path)  # Keep dataset open for merging
        crs = src.crs
        dsm_data.append(src)

    # Merge all DSMs together using rasterio's merge function
    merged_dsm, merged_transform = merge(dsm_data)  # No "transforms" parameter

    # Close all opened datasets
    for src in dsm_data:
        src.close()

    return merged_dsm[0], merged_transform, crs  # Return first (and only) band


def load_dtm(dict_bounds, start_gps, end_gps, dsm_root='data/dtm'):
    """
    Load and merge relevant DSM tiles, crop to region of interest.
    """
    tile1 = find_tile_for_location(dict_bounds, start_gps[::-1])
    tile2 = find_tile_for_location(dict_bounds, end_gps[::-1])
    tiles = set([tile1, tile2])  # Avoid duplication if same tile

    reprojected_paths = []

    for tile in tiles:
        if tile is None:
            return None, None, None
        dtm_path = os.path.join(dsm_root, tile, 'GeoTIFF', tile + '.tif')
        rpj_path = dtm_path.replace(".tif", "_rpj.tif")

        if not os.path.exists(rpj_path):
            reproject_dsm(dtm_path, rpj_path)

        if rpj_path not in reprojected_paths:
            reprojected_paths.append(rpj_path)

    # Load and merge the tiles
    datasets = [rasterio.open(path) for path in reprojected_paths]
    merged_dtm, merged_transform = merge(datasets)
    for ds in datasets:
        ds.close()

    return merged_dtm, merged_transform


def crop_dtm(merged_dtm, merged_transform, start_gps, end_gps, margin=100):
    # Convert GPS to pixel coordinates on merged raster
    start_px = gps_to_pixel(merged_transform, *start_gps)
    end_px = gps_to_pixel(merged_transform, *end_gps)

    cropped_dtm, top_left_offset = crop_dsm(merged_dtm[0], start_px, end_px, margin=margin)

    return cropped_dtm, top_left_offset


def load_and_crop_dtm(dict_bounds, nodes, dsm_root='data/dtm'):
    """
    Load and merge relevant DSM tiles, crop to region of interest.
    """
    tiles = []
    for n in nodes:
        tile = find_tile_for_location(dict_bounds, n[::-1])
        if tile not in tiles:
            tiles.append(tile)

    reprojected_paths = []

    for tile in tiles:
        if tile is None:
            return None, None, None
        
        dtm_path = os.path.join(dsm_root, tile, 'GeoTIFF', tile + '.tif')
        rpj_path = dtm_path.replace(".tif", "_rpj.tif")

        if not os.path.exists(rpj_path):
            reproject_dsm(dtm_path, rpj_path)

        if rpj_path not in reprojected_paths:
            reprojected_paths.append(rpj_path)

    # Load and merge the tiles
    datasets = [rasterio.open(path) for path in reprojected_paths]
    merged_dtm, merged_transform = merge(datasets)
    merged_dtm = merged_dtm[0]

    for ds in datasets:
        ds.close()

    return merged_dtm.astype(np.float32), merged_transform


def compute_coarse_path_from_nodes(
    nodes, graph_nodes, transform, coarse_dtm
):
    """
    Computes a merged coarse path from a sequence of GPS nodes using Dijkstra on a downsampled DSM.

    Args:
        nodes (list of tuples): List of (lat, lon) GPS coordinates.
        transform (Affine): Affine transform of the DSM.
        coarse_dtm (np.ndarray): Downsampled DSM.

    Returns:
        list: Merged list of coarse path points (pixel coordinates).
    """
    final_merge_path = []
    merge_coarse_path = []
    for i in range(len(nodes) - 1):
        # Get current segment GPS
        gps1 = (nodes[i][1], nodes[i][0])
        gps2 = (nodes[i + 1][1], nodes[i + 1][0])

        # Convert to full-res pixel positions
        coarse1 = gps_to_pixel(transform, *gps1[::-1])
        coarse2 = gps_to_pixel(transform, *gps2[::-1])

        if coarse1 == coarse2:
            continue

        # Compute path for this segment
        path = find_line_path(coarse_dtm, coarse1, coarse2)

        merge_coarse_path.append(path)
        if nodes[i + 1] in graph_nodes:
            final_merge_path.append((merge_coarse_path.copy(), nodes[i + 1]))

    return final_merge_path


def refine_and_analyze_path(cropped_dtm, nodes, coarse_path, slope_deg):
    """
    Refines a coarse path, expands the waterway, smooths the path, computes stats, and saves visualizations.

    Args:
        cropped_dtm (np.ndarray): Cropped high-resolution DSM.
        coarse_path (list): List of coarse path coordinates.
        elevation_threshold (float): Elevation difference threshold for expansion.
        pixel_size (float): Pixel resolution (meters per pixel) for stats.

    Returns:
        final_path (list): Interpolated path.
        smooth_path (list): Smoothed and updated path.
        skeleton (np.ndarray): Skeleton of the expanded waterway.
    """
    flattened = [item for sublist in coarse_path for item in sublist]

    # === Expand waterway ===
    distance = total_distance(nodes)

    skeleton = np.zeros(cropped_dtm.shape, dtype=bool)
    flattened = np.array(flattened)  # shape = (N, 2)
    xs, ys = flattened[:, 0], flattened[:, 1]
    skeleton[xs, ys] = True

    slope_stats = slope_statistics_along_centerline(slope_deg, flattened)
    
    results = {}
    stats = compute_waterway_elevation_statistics(cropped_dtm, skeleton)
    results['elevation'] = stats
    results['distance'] = distance
    results['delta_elevation'] = cropped_dtm[
        flattened[-1][0], flattened[-1][1]] - cropped_dtm[
            flattened[0][0], flattened[0][1]]
    results['slope'] = slope_deg[flattened[-1][0], flattened[-1][1]]

    for k in slope_stats.keys():
        results[k] = slope_stats[k]

    return slope_deg[flattened[-1][0], flattened[-1][1]], cropped_dtm[flattened[-1][0], flattened[-1][1]], results


def slope_statistics_along_centerline(slope_map, centerline_coords):
    """
    Compute slope and its statistical measures along a centerline.

    Args:
        dtm (np.ndarray): Digital Terrain Model.
        centerline_coords (list of (row, col)): Points along the centerline.

    Returns:
        dict: Slope statistics: mean, std, min, max, median, percentiles.
    """
    slope_values = []
    h, w = dtm.shape

    for row, col in centerline_coords:
        ri, ci = int(round(row)), int(round(col))
        if 0 <= ri < h and 0 <= ci < w:
            slope_values.append(slope_map[ri, ci])

    slope_values = np.array(slope_values)
    return {
        'slope_mean': np.mean(slope_values),
        'slope_std': np.std(slope_values),
        'slope_min': np.min(slope_values),
        'slope_max': np.max(slope_values),
        'slope_median': np.median(slope_values),
    }


if __name__ == "__main__":
    # === Load Data ===
    if not os.path.exists('data/bounds.pkl'):
        create_dict('data/dtm')

    with open('data/bounds.pkl', 'rb') as f:
        dict_bounds = pickle.load(f)

    with open('data/processed.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    selected_areas = [
        'DHMVIIDTMRAS1m_k13', 'DHMVIIDTMRAS1m_k14', 'DHMVIIDTMRAS1m_k15',
        'DHMVIIDTMRAS1m_k21', 'DHMVIIDTMRAS1m_k22', 'DHMVIIDTMRAS1m_k23',
        'DHMVIIDTMRAS1m_k29', 'DHMVIIDTMRAS1m_k30', 'DHMVIIDTMRAS1m_k31',
    ]

    loading_dict = {}
    for station_key in data_dict.keys():
        station_data = data_dict[station_key]
        for ref_location in station_data['graph'].keys():
            if 'nodes' not in station_data['graph'][ref_location].keys():
                continue
            nodes = station_data['graph'][ref_location]['nodes']
            if nodes is None:
                continue
    
            nodes = [station_key, *nodes, ref_location]
            tiles = []
            for n in nodes: 
                tile = find_tile_for_location(dict_bounds, n[::-1])
                if tile not in tiles:
                    tiles.append(tile)
            
            flag = True
            for tile in tiles:
                if tile not in selected_areas:
                    flag = False

            if len(tiles) < 3 and flag:
                tiles.sort()
                tiles = tuple(tiles)
                if tiles not in loading_dict.keys():
                    loading_dict[tiles] = []
                loading_dict[tiles].append((station_key, ref_location, nodes))

    with open('data/loading.pkl', 'wb') as f:
        pickle.dump(loading_dict, f)

    # === Select nodes ===
    for kdict in loading_dict.keys():
        dtm, transform = None, None
        for pair in loading_dict[kdict]:
            station_key, ref_location, nodes = pair
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
            print(station_key)
            print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')

            station_data = data_dict[station_key]
            graph_nodes = station_data['sim_graph']['nodes']

            if 'nodes' in list(station_data['graph'][ref_location].keys()):
                nodes = station_data['graph'][ref_location]['nodes']
            else:
                continue

            if nodes is None:
                continue

            t0 = time.time()
            if dtm is None:
                dtm, transform = load_and_crop_dtm(
                    dict_bounds, nodes
                )
                t0 = time.time()
                slope_deg = compute_slope(dtm)
                print('Slope: ', time.time() - t0)

            os.makedirs(f"vis/{station_key}", exist_ok=True)
            print(f'End loading - {time.time() - t0}')

            t0 = time.time()
            # === Downsample DSM for coarse path planning ===
            list_merge_coarse_path = compute_coarse_path_from_nodes(
                nodes, graph_nodes, transform, dtm
            )
            print(f'End coarse - {time.time() - t0}')

            t0 = time.time()
            for merge_coarse_path, graph_node in list_merge_coarse_path:
                if os.path.exists(f'data/tmp/stats-{station_key}-{graph_node}.pkl'):
                    continue
                
                slope, elevation, nb_results = refine_and_analyze_path(
                    dtm, nodes, merge_coarse_path, slope_deg
                )

                data_dict[station_key]['slope'] = slope
                data_dict[station_key]['elevation'] = elevation

                nb_results['displacement'] = np.array(graph_node) - np.array(station_key) 
                nb_results['key_slope'] = slope
                nb_results['key_elevation'] = elevation
                with open(f'data/tmp/stats-{station_key}-{graph_node}.pkl', 'wb') as f:
                    pickle.dump(nb_results, f)
            
            print(f'End analysis - {time.time() - t0}')

    with open(f'data/processed_stats.pkl', 'wb') as f:
        pickle.dump(data_dict, f)
