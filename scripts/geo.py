import os
import pickle
import networkx as nx
import geopandas as gpd
import numpy as np

from shapely.geometry import Point, MultiPoint
from shapely.ops import split
from geopy.distance import geodesic
from shapely.strtree import STRtree  # Faster spatial index for points

from scipy import spatial


def save_pkl(graph, path):
    """Save a NetworkX graph to a pickle file."""
    with open(path, "wb") as f:
        pickle.dump(graph, f)


def load_pkl(path):
    """Load a NetworkX graph from a pickle file."""
    with open(path, "rb") as f:
        return pickle.load(f)


def check_connection_and_distance(G, node1, node2):
    print('Node1:', (node1[1], node1[0]), ' --- ' , 'Node2:', (node2[1], node2[0]))

    # Check if nodes are connected
    is_connected = nx.has_path(G, node1, node2)
    is_connected_inv = nx.has_path(G, node2, node1)
    # print(f"Are the points connected? {is_connected or is_connected_inv}")

    # Find the shortest path and distance if connected
    if is_connected:
        path = nx.shortest_path(G, source=node1, target=node2)
        distance = nx.shortest_path_length(G, source=node1, target=node2)
        # print(f"Shortest path distance along the waterway: {distance} km")
        # print(f"Nodes along the shortest path: {path}")
        return is_connected, is_connected_inv, distance, path
    elif is_connected_inv:
        path = nx.shortest_path(G, source=node2, target=node1)
        distance = nx.shortest_path_length(G, source=node2, target=node1)
        # print(f"Shortest path distance along the waterway: {distance} km")
        # print(f"Nodes along the shortest path: {path}")
        return is_connected, is_connected_inv, distance, path

    return is_connected, is_connected_inv, None, None


# Convert GPS coordinates (lat, lon) to ECEF (Earth-Centered, Earth-Fixed) coordinates
def latlon_to_ecef(lat, lon, R=6371):  # Earth's radius in km
    lat, lon = np.radians(lat), np.radians(lon)
    x = R * np.cos(lat) * np.cos(lon)
    y = R * np.cos(lat) * np.sin(lon)
    z = R * np.sin(lat)
    return np.array([x, y, z])


# Compute the shortest distance from point P to line segment AB
def haversine_distance_to_segment(lat1, lon1, lat2, lon2, latp, lonp):
    # Convert to ECEF
    A = latlon_to_ecef(lat1, lon1)
    B = latlon_to_ecef(lat2, lon2)
    P = latlon_to_ecef(latp, lonp)

    # Vector AB and AP
    AB = B - A
    AP = P - A

    # Projection factor t (perpendicular projection)
    t = np.dot(AP, AB) / np.dot(AB, AB)

    # Find intersection point
    if 0 <= t <= 1:  # Projection falls inside the segment
        closest_point = A + t * AB
    else:  # Projection is outside, take the closest endpoint
        closest_point = A if np.linalg.norm(P - A) < np.linalg.norm(P - B) else B

    # Convert closest point back to lat/lon
    x, y, z = closest_point
    lat_closest = np.degrees(np.arcsin(z / 6371))
    lon_closest = np.degrees(np.arctan2(y, x))

    # Compute haversine distance from P to closest point
    distance = geodesic((latp, lonp), (lat_closest, lon_closest)).km

    return distance


# Find the nearest nodes in the graph
def find_nearest_edge(tree, gnodes, point, G):
    point_geom = Point(point[1], point[0])
    nearest_edge = None
    min_distance = float('inf')
    nearest_node = None

    # Use KDTree to find the nearest nodes first
    distances, indices = tree.query([(point[1], point[0])], k=10)
    nearest_nodes = [gnodes[i] for i in indices[0]]

    for node in nearest_nodes:
        for neighbor in G.neighbors(node):
            distance = haversine_distance_to_segment(node[1], node[0], neighbor[1], neighbor[0], point[0], point[1])
            if distance < min_distance:
                min_distance = distance
                nearest_edge = (node, neighbor)
                nearest_node = node if point_geom.distance(Point(node)) < point_geom.distance(Point(neighbor)) else neighbor

    return nearest_node, nearest_edge, min_distance


def create_simplified_graph_for_station(station_data, k):
    """
    For a given station_data (dictionary for station k containing:
      - 'neighbor_path': a dict mapping neighbor IDs (e.g. coordinates) to a dict that includes 
         'nodes' (list of nodes along the path from the neighbor to station k; may be empty)
      - 'intersections': will be computed by compute_intersections_for_station()
    
    Build a simplified directed graph:
      - Nodes: All neighbor station IDs plus any computed intersection nodes.
      - Edges: For each unordered pair of neighbor stations (np1, np2):
            * If both have valid (non-empty) paths:
                - Let p1 and p2 be the ordered lists of nodes from np1 and np2, respectively.
                - Find the first common node in p1 (scanning in order); call it X.
                  (If found, it is assumed that X is the unique intersection point.)
                - Let i1 = index of X in p1, and i2 = index of X in p2.
                  (Lower index means the node appears earlier in the path – i.e. farther from station k.)
                - Then, if i1 < i2, np1 is farther from station k than np2. 
                  Add a directed edge from np1 to np2 (weight = 1).
                  If i1 > i2, add an edge from np2 to np1.
                - In either case, add X as an intersection node to the node set.
            * If at least one neighbor’s path is empty:
                - If one neighbor has a valid path and the other does not, add an edge from the valid neighbor
                  (the one that “can reach” station k) to the invalid neighbor with weight -1.
                - If both are empty, no edge is added.
    Returns a tuple: (list of nodes, list of edges)
    """
    neighbor_path = station_data.get('graph', {})
    # Assume neighbor IDs are something like (lat, lon)
    neighbors = list(neighbor_path.keys())
    
    # Build dictionaries: neighbor_valid and neighbor_paths.
    neighbor_valid = {}
    neighbor_paths = {}
    for np_id in neighbors:
        p = neighbor_path[np_id].get('nodes')
        if p is None or len(p) == 0:
            neighbor_valid[np_id] = False
            neighbor_paths[np_id] = []
        else:
            neighbor_valid[np_id] = True
            neighbor_paths[np_id] = [_p for _p in p[::-1]]  # p is the ordered list of nodes from neighbor to station k

    # Initialize graph: start with all neighbor IDs as nodes.
    graph_nodes = set()
    graph_edges = []  # directed edges: (source, target, weight)
    
    graph_nodes.add(k)

    # Compare each unordered pair of neighbors.
    for i in range(len(neighbors)):
        np1 = neighbors[i]
        if neighbor_valid[np1]:
            i_intersections = []
            i_intersections_idx = []

            for j in range(i+1, len(neighbors)):
                np2 = neighbors[j]
                p1 = neighbor_paths[np1]
                p2 = neighbor_paths[np2]
                
                if neighbor_valid[np1] and neighbor_valid[np2]:
                    # Find the first common node in p1 that appears in p2.
                    intersection_node = None
                    for idx, node in enumerate(p1):
                        if node in p2 and node != p2[-1]:
                            intersection_node = node
                            break

                    if intersection_node is not None and intersection_node not in i_intersections:
                        i_intersections.append(intersection_node)
                        i_intersections_idx.append(p1.index(intersection_node))

            if i_intersections:
                sorted_pairs = sorted(zip(i_intersections_idx, i_intersections))
                sorted_i_intersections_idx, sorted_i_intersections = zip(*sorted_pairs)

                # Convert back to lists
                sorted_i_intersections = list(sorted_i_intersections)
                sorted_i_intersections_idx = list(sorted_i_intersections_idx)

                start = np1
                graph_nodes.add(np1)
                for intersection_node in sorted_i_intersections:
                    graph_edges.append((start, intersection_node, 1))
                    graph_nodes.add(intersection_node)
                    start = intersection_node
                
                if (intersection_node, k, 1) not in graph_edges:
                    graph_edges.append((intersection_node, k, 1))
            else:
                graph_nodes.add(np1)
                graph_edges.append((np1, k, 1))
        else:
            graph_edges.append((np1, k, -1))
            graph_nodes.add(np1)

    return list(graph_nodes), graph_edges


def compute_intersections_for_station(station_data):
    """
    For each unordered pair of neighbor stations in station_data['neighbor_path'],
    compute the unique intersection node (if any) as the first common node encountered
    when scanning the neighbor's path (assumed to be ordered from the neighbor to station k).
    Store the result in station_data['intersections'] with key (np1, np2) (using the original neighbor IDs).
    """
    neighbor_path = station_data.get('graph', {})
    station_data['intersections'] = {}
    neighbors = list(neighbor_path.keys())
    
    for i in range(len(neighbors)):
        np1 = neighbors[i]
        p1 = neighbor_path[np1].get('nodes') or []
        for j in range(i+1, len(neighbors)):
            np2 = neighbors[j]
            p2 = neighbor_path[np2].get('nodes') or []

            intersection_node = None
            for node in p1:
                if node in p2:
                    intersection_node = node
                    break

            if intersection_node is not None:
                station_data['intersections'][(np1, np2)] = intersection_node

    return station_data


def create_simplified_graphs(data):
    """
    Given a dictionary `data` where each key k represents a station (with its station_data
    containing at least 'neighbor_path'), compute intersections for each station and then
    create a simplified graph for each station.
    
    Each simplified graph is represented as a dictionary with:
        {'nodes': [...], 'edges': [...]}
    and will include only one directed edge per neighbor pair.
    Returns a tuple: (data, simplified_graphs)
    where simplified_graphs is a dict mapping each station k to its graph.
    """
    simplified_graphs = {}
    for k in data:
        station_data = data[k]

        # First, compute intersections for station k.
        station_data = compute_intersections_for_station(station_data)
        
        # Then, build the simplified graph.
        nodes, edges = create_simplified_graph_for_station(station_data, k)
        simplified_graphs[k] = {'nodes': nodes, 'edges': edges}
        data[k]['sim_graph'] = {'nodes': nodes, 'edges': edges}

    return data, simplified_graphs


def main():
    # --- Load waterway graphs and node spatial indexes
    # Load graphs and spatial indexes
    (
        waterways, directed_graph, directed_node_ids, directed_tree,
        undirected_graph, undirected_node_ids, undirected_tree,
    ) = create_waterway_graph()

    # --- Load station data
    data = load_pkl('data/selected_stats_rainfall_segment.pkl')

    neighbors_dict = get_neighbors_dict(data, n_neighbors=10)

    # Build graph connections for each station
    build_station_waterway_connections(
        data, neighbors_dict,
        undirected_tree, undirected_node_ids, undirected_graph, directed_graph
    )

    # Save updated data
    save_pkl(data, 'data/selected_stats_rainfall_segment.pkl')

    # Build simplified graphs and print summary
    data, simplified_graphs = create_simplified_graphs(data)
    print_simplified_graphs(simplified_graphs)

    # --- Save the final data again (if needed)
    save_pkl(data, 'data/selected_stats_rainfall_segment.pkl')


def build_node_kdtree(graph):
    """Build a KDTree for fast spatial queries of graph nodes."""
    nodes_array = np.array(list(graph.nodes))
    kdtree = spatial.KDTree(nodes_array)
    return nodes_array, kdtree


def find_intersection_points(all_lines, waterways):
    """Find all unique intersection points among the provided LineStrings."""
    sindex = waterways.sindex
    intersection_point_coords = set()

    for idx, line in enumerate(all_lines):
        candidate_indices = list(sindex.intersection(line.bounds))
        candidates = [all_lines[i] for i in candidate_indices if i > idx]
        for other_line in candidates:
            if line.intersects(other_line):
                intersection = line.intersection(other_line)
                if intersection.geom_type == "Point":
                    intersection_point_coords.add((intersection.x, intersection.y))
                elif intersection.geom_type == "MultiPoint":
                    for pt in intersection.geoms:
                        intersection_point_coords.add((pt.x, pt.y))
    return [Point(x, y) for x, y in intersection_point_coords]


def split_lines_at_points(all_lines, intersection_points):
    """Split all lines at the given intersection points."""
    intersection_index = STRtree(intersection_points)
    split_lines = []
    for line in all_lines:
        relevant_points = intersection_index.query(line)
        split_candidates = [pt for pt in relevant_points if isinstance(pt, Point)]
        if split_candidates:
            split_result = split(line, MultiPoint(split_candidates))
            split_lines.extend(split_result.geoms)
        else:
            split_lines.append(line)
    return split_lines


def build_graphs_from_lines(split_lines, intersection_points):
    """Build directed and undirected graphs from split LineStrings and add intersection points as nodes."""
    directed_graph = nx.DiGraph()
    undirected_graph = nx.Graph()
    for geom in split_lines:
        if geom.geom_type == "LineString":
            lines = [geom]
        elif geom.geom_type == "MultiLineString":
            lines = geom.geoms
        else:
            continue  # Ignore unsupported geometry types

        for line in lines:
            coords = list(line.coords)
            for i in range(len(coords) - 1):
                point1, point2 = coords[i], coords[i + 1]
                distance_km = geodesic(point1, point2).km
                directed_graph.add_edge(point1, point2, weight=distance_km)
                undirected_graph.add_edge(point1, point2, weight=distance_km)

    for point in intersection_points:
        directed_graph.add_node((point.x, point.y))
        undirected_graph.add_node((point.x, point.y))

    return directed_graph, undirected_graph


def get_neighbors_dict(data, n_neighbors=20):
    """Return a dictionary mapping each station to its n nearest neighbors."""
    station_coords = np.array(list(data.keys()))
    station_keys = list(data.keys())
    pairwise_distances = np.linalg.norm(
        station_coords[:, None, :] - station_coords[None, :, :], axis=-1
    )
    neighbors_dict = {}
    for idx, key in enumerate(station_keys):
        nearest_indices = np.argsort(pairwise_distances[idx])[1 : n_neighbors + 1]
        neighbors_dict[key] = [station_keys[j] for j in nearest_indices]
    return neighbors_dict


def build_station_waterway_connections(
    data, neighbors_dict, undirected_tree, undirected_node_ids, undirected_graph, directed_graph
):
    """Populate each station's 'graph' key with waterway connection info to its neighbors."""
    for station_key in data:
        print(f'<---- Station: {station_key} ---->')
        query_node, _, _ = find_nearest_edge(undirected_tree, undirected_node_ids, station_key, undirected_graph)

        data[station_key]['graph'] = {}
        for neighbor_key in neighbors_dict[station_key]:
            if neighbor_key not in data:
                continue
            data[station_key]['graph'][neighbor_key] = {}

            ref_node, _, _ = find_nearest_edge(undirected_tree, undirected_node_ids, neighbor_key, undirected_graph)
            _, _, _, path = check_connection_and_distance(undirected_graph, query_node, ref_node)

            if path is not None:
                # (lat, lon) ordering
                path_latlon = [(p[1], p[0]) for p in path]
                directed_edges = [
                    (path_latlon[i], path_latlon[i + 1])
                    if directed_graph.has_edge(path_latlon[i], path_latlon[i + 1]) else
                    (path_latlon[i + 1], path_latlon[i])
                    for i in range(len(path_latlon) - 1)
                ]
                data[station_key]['graph'][neighbor_key]['edges'] = directed_edges
                data[station_key]['graph'][neighbor_key]['nodes'] = path_latlon
            else:
                data[station_key]['graph'][neighbor_key]['edges'] = None
                data[station_key]['graph'][neighbor_key]['nodes'] = None


def print_simplified_graphs(simplified_graphs):
    """Print summary of nodes and edges for each simplified graph."""
    for station_key, graph in simplified_graphs.items():
        print(f"Station: {station_key}")
        print("Nodes:", graph['nodes'])
        print("Edges:")
        for edge in graph['edges']:
            print("  ", edge)
        print()


def create_waterway_graph():
    """Load and construct waterway graphs and spatial indexes from shapefile and/or pickles."""

    waterways = gpd.read_file("data/Vhag_prj.shp")

    # Initialize graphs: directed (G) and undirected (G2)
    directed_graph = nx.DiGraph()
    undirected_graph = nx.Graph()

    graph_pickle_path = "data/graph.pickle"
    undirected_pickle_path = "data/graph2.pickle"

    if not os.path.exists(graph_pickle_path):
        all_lines = waterways.geometry.tolist()
        intersection_points = find_intersection_points(all_lines, waterways)
        split_lines = split_lines_at_points(all_lines, intersection_points)
        directed_graph, undirected_graph = build_graphs_from_lines(split_lines, intersection_points)
        save_pkl(directed_graph, graph_pickle_path)
        save_pkl(undirected_graph, undirected_pickle_path)

    else:
        # --- Load precomputed graphs from disk
        directed_graph = load_pkl(graph_pickle_path)
        undirected_graph = load_pkl(undirected_pickle_path)

    # --- Build spatial KDTree indexes for nodes
    _, directed_kdtree = build_node_kdtree(directed_graph)
    _, undirected_kdtree = build_node_kdtree(undirected_graph)

    # --- Return all relevant data structures
    return (
        waterways,
        directed_graph, list(directed_graph.nodes), directed_kdtree,
        undirected_graph, list(undirected_graph.nodes), undirected_kdtree,
    )



if __name__ == "__main__":
    main()