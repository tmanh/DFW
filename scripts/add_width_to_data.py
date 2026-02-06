import pickle
import geopandas as gpd
import numpy as np
from collections import deque

from shapely.geometry import LineString, Point


def save_pkl(graph, path):
    """Save a NetworkX graph to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_pkl(path):
    """Load a NetworkX graph from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def find_nearest_linestring(point, gdf, clone_gdf):
    # Calculate the minimum distance from the point to each geometry
    point_gs = gpd.GeoSeries([point], crs="EPSG:4326").to_crs("EPSG:32630")
    distances = clone_gdf.geometry.distance(point_gs.iloc[0])  # This will work as intended
    idx_min = distances.idxmin()
    nearest_row = gdf.loc[idx_min]
    return nearest_row


def find_containing_linestring(point, gdf, gdf_proj, sindex, buffer_dist=1.0):
    point_proj = gpd.GeoSeries([point], crs="EPSG:4326").to_crs(gdf_proj.crs).iloc[0]
    point_buffer = point_proj.buffer(buffer_dist)
    candidate_idx = list(sindex.intersection(point_buffer.bounds))
    if not candidate_idx:
        return None
    candidates = gdf_proj.iloc[candidate_idx]
    mask = candidates.geometry.intersects(point_buffer)
    if not mask.any():
        return None
    # Find closest
    distances = candidates[mask].geometry.distance(point_proj)
    idx_min = distances.idxmin()  # idx_min is the original DataFrame index!
    return gdf.loc[idx_min]       # <-- Use loc, not iloc!
    

def euclidean(p1, p2):
    # Assumes p1 and p2 are (x, y)
    return np.linalg.norm(np.array(p1) - np.array(p2))

def find_path_with_directions(edges_set, start, end):
    """
    BFS to find a path from start to end using edges_set, and record step directions.
    Returns: (list of nodes along path, list of (is_forward, distance, width) per step)
    """
    # edges_set: set of (u, v, w1, w2)
    # For lookup, create maps for both directions
    edge_map_fwd = {(u, v): (w1, w2) for (u, v, w1, w2) in edges_set}
    edge_map_bwd = {(v, u): (w2, w1) for (u, v, w1, w2) in edges_set}

    queue = deque()
    queue.append((start, [start], []))
    while queue:
        node, path, steps = queue.popleft()
        if node == end:
            return path, steps
       
        # Forward edges
        for (u, v), (w1, w2) in edge_map_fwd.items():
            if u == node and v not in path:
                d = euclidean(u, v)
                queue.append((v, path + [v], steps + [(True, d, w1)]))

        # Backward edges
        for (v, u), (w2, w1) in edge_map_bwd.items():
            if v == node and u not in path:
                d = euclidean(v, u)
                queue.append((u, path + [u], steps + [(False, d, w2)]))
    return None, None


def find_pairwise_first_intersections(neighbor_nodes, neighbors, neighbor_valid, k):
    intersection_nodes = set()
    for i in range(len(neighbors)):
        np1 = neighbors[i]
        if not neighbor_valid[np1]:
            continue
        
        p1 = neighbor_nodes[np1]
        for j in range(i + 1, len(neighbors)):
            np2 = neighbors[j]
            if not neighbor_valid[np2]:
                continue
            
            p2 = neighbor_nodes[np2]
            
            # Find first common node in p1 that is not k (target) and is in p2
            intersection_node = None
            for node in p1:
                if node in p2 and node != k:
                    intersection_node = node
                    break
            if intersection_node is not None:
                intersection_nodes.add(intersection_node)
    return intersection_nodes


def l2_distance(a, b):
    return np.sum((np.array(a) - np.array(b))**2)


def create_simplified_graph_true_direction(station_data, k):
    neighbor_path = station_data.get('graph', {})
    neighbors = list(neighbor_path.keys())

    neighbor_valid = {}
    neighbor_nodes = {}
    neighbor_edges = {}

    for np_id in neighbors:
        nodes = neighbor_path[np_id].get('nodes', [])
        edges = neighbor_path[np_id].get('edges', [])

        if not nodes:
            neighbor_valid[np_id] = False
            neighbor_nodes[np_id] = []
            neighbor_edges[np_id] = set()
        else:
            neighbor_valid[np_id] = True
            nodes = list(nodes)
            edges = set(tuple(e) for e in edges)

            e_f, e_l = list(edges)[0], list(edges)[-1]
            if l2_distance(nodes[0], k) < l2_distance(nodes[0], np_id):
                if k not in nodes:
                    edges = edges | {(k, nodes[0], e_f[-2], e_f[-1])}
                    nodes.insert(0, k)
                if np_id not in nodes:
                    edges = edges | {(nodes[-1], np_id, e_l[-2], e_l[-1])}
                    nodes.append(np_id)
                nodes = nodes[::-1]
            else:
                if np_id not in nodes:
                    edges = edges | {(np_id, nodes[0], e_f[-2], e_f[-1])}
                    nodes.insert(0, np_id)
                if k not in nodes:
                    edges = edges | {(nodes[-1], k, e_l[-2], e_l[-1])}
                    nodes.append(k)
            neighbor_nodes[np_id] = nodes.copy()
            neighbor_edges[np_id] = edges # (u, v, w1, w2)

    # --- YOUR intersection logic here ---
    intersection_nodes = find_pairwise_first_intersections(neighbor_nodes, neighbors, neighbor_valid, k)
    important_nodes = set(neighbors) | intersection_nodes | {k}
    graph_edges = set()

    for np_id in neighbors:
        if not neighbor_valid[np_id]:
            continue
        
        path = neighbor_nodes[np_id]
        edges_in_path = neighbor_edges[np_id]
        
        important_indices = [i for i, node in enumerate(path) if node in important_nodes]
        if path[important_indices[0]] == k:
            important_indices = important_indices[::-1]

        for idx in range(len(important_indices) - 1):
            n1 = path[important_indices[idx]]
            n2 = path[important_indices[idx + 1]]

            node_path, step_info = find_path_with_directions(edges_in_path, n1, n2)
            if node_path is not None and step_info is not None:
                forward_length = sum(d for is_fwd, d, w in step_info if is_fwd)
                backward_length = sum(d for is_fwd, d, w in step_info if not is_fwd)
                all_widths = [w for is_fwd, d, w in step_info]
                avg_width = np.mean(all_widths) if all_widths else None
                std_width = np.std(all_widths) if all_widths else None
                min_width = np.min(all_widths) if all_widths else None
                max_width = np.max(all_widths) if all_widths else None
                med_width = np.median(all_widths) if all_widths else None
                graph_edges.add((n1, n2, avg_width, std_width, min_width, max_width, med_width, forward_length, backward_length))
            else:
                graph_edges.add((n1, n2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0))

    graph_nodes = list(important_nodes)
    return graph_nodes, list(graph_edges)


def main():
    waterways = gpd.read_file("data/osm/gis_osm_waterways_free_1.shp")

    projected_crs = "EPSG:32630"
    clone_waterways = waterways.to_crs(projected_crs)

    print(waterways.head())
    print(waterways.columns)
    width = [v for v in list(waterways['width'].values) if v is not None]

    waterways_proj = waterways.to_crs("EPSG:32630")
    sindex = waterways_proj.sindex

    # data = load_pkl('data/selected_stats_rainfall_segment.pkl')
    # for k in data:
    #     count = 0
    #     total = 0

    #     # (1) Edges from your graph
    #     edge_geoms = []
    #     edge_status = []
    #     matched_names = []
    #     edge_ids = []

    #     # (2) Matched waterways linestrings
    #     matched_lines_geoms = []
    #     matched_lines_names = []
    #     matched_lines_role = []  # 'row1' or 'row2'
    #     matched_lines_edge_id = []

    #     for nb in data[k]['graph'].keys():
    #         if data[k]['graph'][nb]['edges'] is None:
    #             continue

    #         point1 = Point(k[::-1])
    #         point2 = Point(nb[::-1])
    #         row1 = find_containing_linestring(point1, waterways, waterways_proj, sindex, buffer_dist=10.0)
    #         row2 = find_containing_linestring(point2, waterways, waterways_proj, sindex, buffer_dist=10.0)

    #         for i in range(len((data[k]['graph'][nb]['edges']))):
    #             p1, p2 = data[k]['graph'][nb]['edges'][i]

    #             new_edges = []

    #             point1 = Point(p1[::-1])
    #             point2 = Point(p2[::-1])
    #             row1 = find_containing_linestring(point1, waterways, waterways_proj, sindex, buffer_dist=10.0)
    #             row2 = find_containing_linestring(point2, waterways, waterways_proj, sindex, buffer_dist=10.0)
    #             w1 = row1['width'] if row1 is not None else 0
    #             w2 = row2['width'] if row2 is not None else 0

    #             data[k]['width'] = w1
    #             data[nb]['width'] = w2

    #             # Store edge (user connection)
    #             edge_geoms.append(LineString([point1, point2]))
    #             edge_ids.append(i)
    #             if row1 is not None and row2 is not None and row1['name'] == row2['name']:
    #                 edge_status.append("matched")
    #                 matched_names.append(row1['name'])
    #             else:
    #                 edge_status.append("unmatched")
    #                 matched_names.append(None)

    #             # Store matched lines (from waterways)
    #             if row1 is not None:
    #                 matched_lines_geoms.append(row1.geometry)
    #                 matched_lines_names.append(row1['name'])
    #                 matched_lines_role.append("row1")
    #                 matched_lines_edge_id.append(i)
    #             if row2 is not None:
    #                 matched_lines_geoms.append(row2.geometry)
    #                 matched_lines_names.append(row2['name'])
    #                 matched_lines_role.append("row2")
    #                 matched_lines_edge_id.append(i)

    #             # print(row1)
    #             w1 = row1['width'] if row1 is not None else 0
    #             w2 = row2['width'] if row2 is not None else 0
    #             if w1 > 0 or w2 > 0:
    #                 print(w1, w2)
                
    #             new_edges.append((p1, p2, w1, w2))

    #         data[k]['graph'][nb]['edges'] = new_edges

    # save_pkl(data, 'new.pkl')

    data = load_pkl('new.pkl')

    keys = list(data.keys())
    for k in keys:
        k = keys[1]
        station_data = data[k]
        nodes, edges = create_simplified_graph_true_direction(station_data, k)
        data[k]['sim_graph'] = {'nodes': nodes, 'edges': edges}

    save_pkl(data, 'new_data.pkl')
    exit()