import os
import time
import pickle
import numpy as np

from rasterio.merge import merge
from skimage.restoration import denoise_tv_chambolle

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
        dsm_data.append(src)

    # Merge all DSMs together using rasterio's merge function
    merged_dsm, merged_transform = merge(dsm_data)  # No "transforms" parameter

    # Close all opened datasets
    for src in dsm_data:
        src.close()

    return merged_dsm[0], merged_transform  # Return first (and only) band


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


def load_and_crop_dtm(dict_bounds, start_gps, end_gps, margin=500, dsm_root='data/dtm'):
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

    # Convert GPS to pixel coordinates on merged raster
    start_px = gps_to_pixel(merged_transform, *start_gps)
    end_px = gps_to_pixel(merged_transform, *end_gps)

    cropped_dtm, top_left_offset = crop_dsm(merged_dtm[0], start_px, end_px, margin=margin)

    return cropped_dtm, merged_transform, top_left_offset


def compute_coarse_path_from_nodes(
    nodes, graph_nodes, transform, top_left_offset, coarse_dtm
):
    """
    Computes a merged coarse path from a sequence of GPS nodes using Dijkstra on a downsampled DSM.

    Args:
        nodes (list of tuples): List of (lat, lon) GPS coordinates.
        transform (Affine): Affine transform of the DSM.
        top_left_offset (tuple): Offset of cropped DSM relative to full DSM.
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
        px1 = gps_to_pixel(transform, *gps1[::-1])
        px2 = gps_to_pixel(transform, *gps2[::-1])

        # Apply crop offset
        crop1 = (px1[0] - top_left_offset[0], px1[1] - top_left_offset[1])
        crop2 = (px2[0] - top_left_offset[0], px2[1] - top_left_offset[1])

        # Convert to coarse-resolution pixels
        coarse1 = (crop1[0], crop1[1])
        coarse2 = (crop2[0], crop2[1])

        if coarse1 == coarse2:
            continue

        # Compute path for this segment
        # t0 = time.time()
        # path = dijkstra_guided_water_cost(coarse_dtm, coarse1, coarse2)
        path = find_line_path(coarse_dtm, coarse1, coarse2)
        # print(f"→ Segment {i} coarse path in {time.time() - t0:.2f}s")

        merge_coarse_path.append(path)
        if nodes[i + 1] in graph_nodes:
            final_merge_path.append((merge_coarse_path.copy(), nodes[i + 1]))

    return final_merge_path


def compute_coarse_path_from_two_nodes(
    nodes, transform, top_left_offset, coarse_dtm
):
    """
    Computes a merged coarse path from a sequence of GPS nodes using Dijkstra on a downsampled DSM.

    Args:
        nodes (list of tuples): List of (lat, lon) GPS coordinates.
        transform (Affine): Affine transform of the DSM.
        top_left_offset (tuple): Offset of cropped DSM relative to full DSM.
        coarse_dtm (np.ndarray): Downsampled DSM.

    Returns:
        list: Merged list of coarse path points (pixel coordinates).
    """
    final_merge_path = []
    for i in range(len(nodes) - 1):
        # Get current segment GPS
        gps1 = (nodes[i][1], nodes[i][0])
        gps2 = (nodes[i + 1][1], nodes[i + 1][0])

        # Convert to full-res pixel positions
        px1 = gps_to_pixel(transform, *gps1[::-1])
        px2 = gps_to_pixel(transform, *gps2[::-1])

        # Apply crop offset
        crop1 = (px1[0] - top_left_offset[0], px1[1] - top_left_offset[1])
        crop2 = (px2[0] - top_left_offset[0], px2[1] - top_left_offset[1])

        # Convert to coarse-resolution pixels
        coarse1 = (crop1[0], crop1[1])
        coarse2 = (crop2[0], crop2[1])

        if coarse1 == coarse2:
            continue

        # Compute path for this segment
        # t0 = time.time()
        path = find_line_path(coarse_dtm, coarse1, coarse2)
        final_merge_path.extend(path)
        
    return final_merge_path


def refine_and_analyze_path(station_key, ref_location, cropped_dtm, coarse_path, elevation_threshold=1.05, pixel_size=1):
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
    t0 = time.time()
    # expanded = expand_water_surface_from_line(cropped_dtm, flattened, coarse_path, elevation_threshold=elevation_threshold)
    expanded = waterway_surface_mask(cropped_dtm, flattened)
    print(f"→ Waterway expanded in {time.time() - t0:.2f}s")

    results = {}
    # === Compute stats: Width ===
    t0 = time.time()
    skeleton = skeletonize(expanded.astype(bool))
    stats = compute_waterway_statistics(expanded, skeleton, pixel_size=pixel_size)
    print(f"→ Width stats in {time.time() - t0:.2f}s")
    print(f"Width - Mean: {stats[0]:.2f} m, Std: {stats[1]:.2f} m")
    print(f"Z-score - Max: {stats[2]:.2f}, Mean: {stats[3]:.2f}, Min: {stats[4]:.2f}, Std: {stats[5]:.2f}")
    results['width'] = stats

    # === Compute stats: Elevation ===
    t0 = time.time()
    stats = compute_waterway_elevation_statistics(cropped_dtm, expanded)
    print(f"→ Elevation stats in {time.time() - t0:.2f}s")
    print(f"Elevation - Mean: {stats[0]}, Std: {stats[1]}")
    print(f"Z-score - Max: {stats[2]}, Mean: {stats[3]}, Min: {stats[4]}, Std: {stats[5]}")
    results['elevation'] = stats

    # === Save visualizations ===
    os.makedirs(f"vis/{station_key}", exist_ok=True)
    save_expanded_waterway_plot(cropped_dtm, expanded, f"vis/{station_key}/expanded_waterway-{ref_location}.jpg")

    if coarse_path:
        save_path(cropped_dtm, flattened, f"vis/{station_key}/plot_smooth_path-{ref_location}.jpg")

    return results


def compute_bounding_box_gps(nodes):
    """
    Computes the top-left and bottom-right GPS coordinates from a list of (lat, lon) nodes.

    Args:
        nodes (list of tuples): List of (lat, lon) coordinates.

    Returns:
        top_left_gps (tuple): (lat, lon) of the top-left corner.
        bottom_right_gps (tuple): (lat, lon) of the bottom-right corner.
    """
    lats = [lat for lat, lon in nodes]
    lons = [lon for lat, lon in nodes]

    max_lat = max(lats)
    min_lat = min(lats)
    max_lon = max(lons)
    min_lon = min(lons)

    top_left_gps = (max_lat, min_lon)
    bottom_right_gps = (min_lat, max_lon)

    return top_left_gps, bottom_right_gps


def get_analysis_results(nodes, dict_bounds):
    t0 = time.time()
    top_left_gps, bottom_right_gps = compute_bounding_box_gps(nodes)
    cropped_dtm, transform, top_left_offset = load_and_crop_dtm(
        dict_bounds, top_left_gps, bottom_right_gps
    )

    # cropped_dtm = denoise_bilateral(cropped_dtm)
    cropped_dtm = denoise_tv_chambolle(cropped_dtm)
    save_raw(cropped_dtm, f'test.jpg')

    if cropped_dtm is None:
        return None

    os.makedirs(f"vis/{station_key}", exist_ok=True)
    save_raw(cropped_dtm, f'vis/{station_key}/plot_raw-{ref_location}.jpg')
    print(f'End loading - {time.time() - t0}')

    coarse_path = compute_coarse_path_from_two_nodes(
        nodes, transform, top_left_offset, cropped_dtm
    )
    print(coarse_path)
    exit()
    nb_results = refine_and_analyze_path(
        station_key, ref_location, cropped_dtm, coarse_path,
        elevation_threshold=1.07, pixel_size=1
    )

    return nb_results


if __name__ == "__main__":
    # === Load Data ===
    if not os.path.exists('data/bounds.pkl'):
        create_dict('data/dtm')

    with open('data/bounds.pkl', 'rb') as f:
        dict_bounds = pickle.load(f)

    with open('data/processed.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    # === Select nodes ===
    for station_key in data_dict.keys():
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(station_key)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        if os.path.exists(f'data/tmp/{station_key}.pkl'):
            continue
        
        station_data = data_dict[station_key]
        graph_nodes = station_data['sim_graph']['nodes']

        results = {}
        for ref_location in station_data['graph'].keys():
            if 'nodes' in list(station_data['graph'][ref_location].keys()):
                nodes = station_data['graph'][ref_location]['nodes']
            else:
                continue

            if nodes is None:
                continue

            nodes = [station_key, *nodes, ref_location]

            t0 = time.time()
            top_left_gps, bottom_right_gps = compute_bounding_box_gps(nodes)
            cropped_dtm, transform, top_left_offset = load_and_crop_dtm(
                dict_bounds, top_left_gps, bottom_right_gps
            )

            # cropped_dtm = denoise_bilateral(cropped_dtm)
            cropped_dtm = denoise_tv_chambolle(cropped_dtm)
            save_raw(cropped_dtm, f'test.jpg')

            if cropped_dtm is None:
                continue

            os.makedirs(f"vis/{station_key}", exist_ok=True)
            save_raw(cropped_dtm, f'vis/{station_key}/plot_raw-{ref_location}.jpg')
            print(f'End loading - {time.time() - t0}')

            # === Downsample DSM for coarse path planning ===
            coarse_dtm = downsample_dsm(cropped_dtm)

            list_merge_coarse_path = compute_coarse_path_from_nodes(
                nodes, graph_nodes, transform, top_left_offset, coarse_dtm
            )

            for merge_coarse_path, graph_node in list_merge_coarse_path:
                nb_results = refine_and_analyze_path(
                    station_key, ref_location, cropped_dtm, merge_coarse_path,
                    elevation_threshold=1.07, pixel_size=1
                )
                results[graph_node] = nb_results

        with open(f'data/tmp/{station_key}.pkl', 'wb') as f:
            pickle.dump(results, f)


def main_legacy():
    # === Load Data ===
    if not os.path.exists('data/bounds.pkl'):
        create_dict('data/dtm')

    with open('data/bounds.pkl', 'rb') as f:
        dict_bounds = pickle.load(f)

    with open('data/processed.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    # === Select nodes ===
    for station_key in data_dict.keys():
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        print(station_key)
        print('xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx')
        if os.path.exists(f'data/tmp/{station_key}.pkl'):
            continue
        
        station_data = data_dict[station_key]
        graph_nodes = station_data['sim_graph']['nodes']

        results = {}
        for ref_location in station_data['graph'].keys():
            if 'nodes' in list(station_data['graph'][ref_location].keys()):
                nodes = station_data['graph'][ref_location]['nodes']
            else:
                continue

            if nodes is None:
                continue

            nodes = [station_key, *nodes, ref_location]

            t0 = time.time()
            top_left_gps, bottom_right_gps = compute_bounding_box_gps(nodes)
            cropped_dtm, transform, top_left_offset = load_and_crop_dtm(
                dict_bounds, top_left_gps, bottom_right_gps
            )

            # cropped_dtm = denoise_bilateral(cropped_dtm)
            cropped_dtm = denoise_tv_chambolle(cropped_dtm)
            save_raw(cropped_dtm, f'test.jpg')

            if cropped_dtm is None:
                continue

            os.makedirs(f"vis/{station_key}", exist_ok=True)
            save_raw(cropped_dtm, f'vis/{station_key}/plot_raw-{ref_location}.jpg')
            print(f'End loading - {time.time() - t0}')

            # === Downsample DSM for coarse path planning ===
            coarse_dtm = downsample_dsm(cropped_dtm)

            list_merge_coarse_path = compute_coarse_path_from_nodes(
                nodes, graph_nodes, transform, top_left_offset, coarse_dtm
            )

            for merge_coarse_path, graph_node in list_merge_coarse_path:
                nb_results = refine_and_analyze_path(
                    station_key, ref_location, cropped_dtm, merge_coarse_path,
                    elevation_threshold=1.07, pixel_size=1
                )
                results[graph_node] = nb_results

        with open(f'data/tmp/{station_key}.pkl', 'wb') as f:
            pickle.dump(results, f)