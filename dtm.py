import os
import time
import pickle
import numpy as np
import gc
import psutil
from rasterio.warp import transform_bounds
from rasterio.transform import from_origin

import scipy
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds
from rasterio.io import MemoryFile
from shapely.geometry import MultiPoint

from common.dsm import *
from common.graph import *
from common.vis import *


def compute_waterway_elevation_statistics(cropped_dsm, centerline_coords):
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
    :param centerline_coords: centerline_coords
    :return: Tuple (avg_elevation, std_elevation, max_z, mean_z, min_z, std_z)
    """
    # Select only the waterway elevations.
    waterway_elevations = []
    h, w = dtm.shape
    for row, col in centerline_coords:
        ri, ci = int(round(row)), int(round(col))
        if 0 <= ri < h and 0 <= ci < w:
            waterway_elevations.append(cropped_dsm[ri, ci])

    if len(waterway_elevations) == 0:
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


def get_bounds_from_nodes(nodes, buffer=0.001):
    """
    Get bounding box from nodes with optional buffer in degrees.
    """
    lats, lons = zip(*nodes)
    min_lon, max_lon = min(lons), max(lons)
    min_lat, max_lat = min(lats), max(lats)

    return (
        min_lon - buffer,  # left
        min_lat - buffer,  # bottom
        max_lon + buffer,  # right
        max_lat + buffer   # top
    )


def check_node_coverage(merged_transform, crs, merged_dtm, nodes):
    h, w = merged_dtm.shape
    outside = 0
    for n in nodes:
        p = gps_to_pixel(merged_transform, *n, crs)
        if p[0] < 0 or p[1] < 0 or p[0] >= h or p[1] >= w:
            outside += 1

    print(f"{outside}/{len(nodes)} nodes fall outside the cropped DTM.")

    return outside > 0


def load_and_crop_dtm(dict_bounds, nodes, dsm_root='data/dtm'):
    """
    Load and crop DTM tiles to the region defined by 'nodes'.
    Only reads the necessary parts of each tile to reduce memory usage.
    """
    tiles = []
    for n in nodes:
        tile = find_tile_for_location(dict_bounds, n[::-1])
        if tile is not None and tile not in tiles:
            tiles.append(tile)

    if not tiles:
        return None, None, None

    bounds = get_bounds_from_nodes(nodes)
    cropped_datasets = []
    memfiles = []
    ref_crs = None
    ref_res = None
    base_origin = None

    for tile in tiles:
        dtm_path = os.path.join(dsm_root, tile, 'GeoTIFF', tile + '.tif')
        rpj_path = dtm_path.replace(".tif", "_rpj.tif")
        if not os.path.exists(rpj_path):
            reproject_dsm(dtm_path, rpj_path)

        with rasterio.open(rpj_path) as src:
            try:
                if ref_crs is None:
                    ref_crs = src.crs
                    ref_res = src.res
                    base_origin = (src.transform.c, src.transform.f)

                src_bounds = transform_bounds("EPSG:4326", src.crs, *bounds)

                minx = max(src_bounds[0], src.bounds.left)
                miny = max(src_bounds[1], src.bounds.bottom)
                maxx = min(src_bounds[2], src.bounds.right)
                maxy = min(src_bounds[3], src.bounds.top)

                if minx >= maxx or miny >= maxy:
                    continue

                # Align bounds to full-grid origin
                xres, yres = ref_res
                origin_x, origin_y = base_origin
                minx = math.floor((minx - origin_x) / xres) * xres + origin_x
                miny = math.floor((miny - origin_y) / yres) * yres + origin_y
                maxx = math.ceil((maxx - origin_x) / xres) * xres + origin_x
                maxy = math.ceil((maxy - origin_y) / yres) * yres + origin_y

                safe_bounds = (minx, miny, maxx, maxy)
                window = from_bounds(*safe_bounds, transform=src.transform)
                data = src.read(1, window=window)

                transform = from_origin(minx, maxy, xres, yres)

                memfile = MemoryFile()
                dataset = memfile.open(
                    driver='GTiff',
                    height=data.shape[0],
                    width=data.shape[1],
                    count=1,
                    dtype=data.dtype,
                    transform=transform,
                    crs=src.crs
                )
                memfiles.append(memfile)
                dataset.write(data, 1)
                cropped_datasets.append(dataset)
            except Exception:
                continue

    if not cropped_datasets:
        return None, None, None

    merged_dtm, merged_transform = merge(cropped_datasets)
    merged_dtm = merged_dtm[0]

    for ds in cropped_datasets:
        ds.close()
    for mf in memfiles:
        mf.close()

    gc.collect()
    return merged_dtm.astype(np.float32), merged_transform, ref_crs


def compute_coarse_path_from_nodes(
    nodes, graph_nodes, transform, raster_crs
):
    """
    Computes a merged coarse path from a sequence of GPS nodes using Dijkstra on a downsampled DSM.

    Args:
        nodes (list of tuples): List of (lat, lon) GPS coordinates.
        transform (Affine): Affine transform of the DSM.

    Returns:
        list: Merged list of coarse path points (pixel coordinates).
    """
    all_merge_path = []
    merge_path = []
    total_path = []
    start = 0
    for i in range(len(nodes) - 1):
        if merge_path == []:
            start = i

        # Get current segment GPS
        gps1 = (nodes[i][1], nodes[i][0])
        gps2 = (nodes[i + 1][1], nodes[i + 1][0])

        if gps1 == gps2:
            continue

        # Convert to full-res pixel positions
        coarse1 = gps_to_pixel(transform, *gps1[::-1], raster_crs)
        coarse2 = gps_to_pixel(transform, *gps2[::-1], raster_crs)

        # Compute path for this segment
        path = find_line_path(coarse1, coarse2)

        merge_path.append(path)
        total_path.append(path)
        if nodes[i + 1] in graph_nodes or i == len(nodes) - 2:
            all_merge_path.append((merge_path.copy(), nodes[start], nodes[i + 1]))
            merge_path = []

    if (total_path, nodes[0], nodes[-1]) not in all_merge_path:
        all_merge_path.append((total_path, nodes[0], nodes[-1]))
    return all_merge_path


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

    elevation_stats = slope_statistics_along_centerline(cropped_dtm, flattened)
    slope_stats = slope_statistics_along_centerline(slope_deg, flattened)
    
    results = {}
    # stats = compute_waterway_elevation_statistics(cropped_dtm, flattened)

    results['elevation'] = elevation_stats
    results['distance'] = distance
    results['delta_elevation'] = cropped_dtm[
        flattened[-1][0], flattened[-1][1]] - cropped_dtm[
            flattened[0][0], flattened[0][1]]
    results['slope'] = slope_deg[flattened[-1][0], flattened[-1][1]]

    for k in slope_stats.keys():
        results[k] = slope_stats[k]

    return flattened[0], flattened[-1], results


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

    with open('data/selected_stats_rainfall_segment.pkl', 'rb') as f:
        data_dict = pickle.load(f)

    selected_areas = [
        'DHMVIIDTMRAS1m_k13', 'DHMVIIDTMRAS1m_k14', 'DHMVIIDTMRAS1m_k15',
        'DHMVIIDTMRAS1m_k21', 'DHMVIIDTMRAS1m_k22', 'DHMVIIDTMRAS1m_k23',
        'DHMVIIDTMRAS1m_k29', 'DHMVIIDTMRAS1m_k30', 'DHMVIIDTMRAS1m_k31',
    ]

    loading_dict = {}
    for station_key in data_dict.keys():
        station_data = data_dict[station_key]
        for ref_location in station_data['neighbor'].keys():
            if ref_location not in station_data['neighbor'] or 'nodes' not in station_data['neighbor'][ref_location].keys():
                nodes = [station_key, ref_location]
            else:
                nodes = station_data['neighbor'][ref_location]['nodes']
                if nodes is not None:
                    nodes = [station_key, *nodes, ref_location]
                else:
                    nodes = [station_key, ref_location]
            
            tiles = []
            for n in nodes: 
                tile = find_tile_for_location(dict_bounds, n[::-1])
                if tile not in tiles:
                    tiles.append(tile)
                
            flag = True
            for tile in tiles:
                if tile not in selected_areas:
                    flag = False

            if len(tiles) < 4 and flag:
                tiles.sort()
                tiles = tuple(tiles)
                if tiles not in loading_dict.keys():
                    loading_dict[tiles] = []
                if os.path.exists(f'data/tmp/stats-{station_key}-{ref_location}.pkl'):
                    continue
                loading_dict[tiles].append((station_key, ref_location, nodes))

    # for kdict in loading_dict.keys():
    #     all_nodes = []
    #     for pair in loading_dict[kdict]:
    #         station_key, ref_location, nodes = pair
    #         all_nodes.extend(nodes)

    #     dtm, transform = load_and_crop_dtm(
    #         dict_bounds, nodes
    #     )

    #     for pair in loading_dict[kdict]:
    #         station_key, ref_location, nodes = pair

    #         # Convert to full-res pixel positions
    #         key_pos = gps_to_pixel(transform, *station_key)
    #         ref_pos = gps_to_pixel(transform, *ref_location)

    #         data_dict[station_key]['elevation'] = dtm[key_pos[0], key_pos[1]]
    #         if ref_location in data_dict:
    #             data_dict[ref_location]['elevation'] = dtm[ref_pos[0], ref_pos[1]]

    # with open('data/selected_stats_rainfall.pkl_new', 'wb') as f:
    #     pickle.dump(data_dict, f)
    # exit()

    # === Select nodes ===
    for kdict in loading_dict.keys():
        t0 = time.time()

        for pair in loading_dict[kdict]:
            station_key, ref_location, nodes = pair
            
            if os.path.exists(f'data/tmp/stats-{station_key}-{ref_location}.pkl'):
                continue

            print('Loading ...........')
            dtm, transform, raster_crs = load_and_crop_dtm(
                dict_bounds, nodes
            )
            dtm = scipy.ndimage.median_filter(dtm, size=5)
            slope_deg = compute_slope(dtm)
            print(f'Loaded {dtm.shape}...........')

            station_data = data_dict[station_key]
            graph_nodes = station_data['sim_graph']['nodes']

            path_flag = True
            if ref_location not in station_data['neighbor'] or 'nodes' not in station_data['neighbor'][ref_location].keys():
                path_flag = False
            elif station_data['neighbor'][ref_location]['nodes'] is  None:
                path_flag = False
            print('Has path ......')
            t0 = time.time()
            if path_flag:
                # === Downsample DSM for coarse path planning ===
                list_merge_coarse_path = compute_coarse_path_from_nodes(
                    nodes, graph_nodes, transform, raster_crs
                )

                for merge_coarse_path, start_node, end_node in list_merge_coarse_path:
                    if os.path.exists(f'data/tmp/stats-{start_node}-{end_node}.pkl'):
                        continue

                    src_loc, key_loc, nb_results = refine_and_analyze_path(
                        dtm, nodes, merge_coarse_path, slope_deg
                    )

                    nb_results['displacement'] = np.array(end_node) - np.array(start_node)
                    key_loc = gps_to_pixel(transform, *start_node, raster_crs)
                    src_loc = gps_to_pixel(transform, *end_node, raster_crs)
                    nb_results['key_slope'] = slope_deg[key_loc[0], key_loc[1]]
                    nb_results['key_elevation'] = dtm[key_loc[0], key_loc[1]]
                    nb_results['src_slope'] = slope_deg[src_loc[0], src_loc[1]]
                    nb_results['src_elevation'] = dtm[src_loc[0], src_loc[1]]

                    with open(f'data/tmp/stats-{start_node}-{end_node}.pkl', 'wb') as f:
                        pickle.dump(nb_results, f)
                    del nb_results
            else:
                print('---No path---')
                nb_results = {
                    'elevation': (-1, -1, -1, -1, -1, -1),
                    'distance': -1,
                    'delta_elevation': -1,
                    'slope_mean': -1,
                    'slope_std': -1,
                    'slope_min': -1,
                    'slope_max': -1,
                    'slope_median': -1,
                    'displacement': -1,
                    'key_slope': -1,
                    'key_elevation': -1,
                    'src_slope': -1,
                    'src_elevation': -1
                }

                nb_results['displacement'] = np.array(ref_location) - np.array(station_key)
                key_loc = gps_to_pixel(transform, *station_key, raster_crs)
                src_loc = gps_to_pixel(transform, *ref_location, raster_crs)
                nb_results['delta_elevation'] = dtm[src_loc[0], src_loc[1]] - dtm[key_loc[0], key_loc[1]]
                nb_results['key_slope'] = slope_deg[key_loc[0], key_loc[1]]
                nb_results['key_elevation'] = dtm[key_loc[0], key_loc[1]]
                nb_results['src_slope'] = slope_deg[src_loc[0], src_loc[1]]
                nb_results['src_elevation'] = dtm[src_loc[0], src_loc[1]]
                with open(f'data/tmp/stats-{station_key}-{ref_location}.pkl', 'wb') as f:
                    pickle.dump(nb_results, f)
                del nb_results
            
            print(f'End saving - {time.time() - t0}')
        
            mem = psutil.virtual_memory()
            print(f"Available RAM: {mem.available / (1024 ** 3):.2f} GB")
        
            del dtm
            del slope_deg
            gc.collect()
    
    # with open(f'data/selected_stats_rainfall.pkl', 'wb') as f:
    #     pickle.dump(data_dict, f)
