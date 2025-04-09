import os
import pickle
import networkx as nx
import geopandas as gpd
import numpy as np

from shapely.geometry import LineString, Point, MultiPoint, Polygon
from shapely.ops import split
from geopy.distance import geodesic
from shapely.strtree import STRtree  # Faster spatial index for points

import matplotlib.pyplot as plt
from scipy import spatial
import time


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
    waterways, DiG, node_ids, tree, UndiG, node2_ids, tree2 = create_waterway_graph()

    with open('data/processed.pkl', 'rb') as f:
        data = pickle.load(f)

    with open('data/neighbor.pkl', 'rb') as f:
        neighbors = pickle.load(f)

    if 'graph' not in data[list(data.keys())[0]].keys():
        for k in data.keys():
            print(f'<>----{k}----<>')
            qry2_node, _, _ = find_nearest_edge(tree2, node2_ids, k, UndiG)

            data[k]['graph'] = {}

            for nb in neighbors[k]:
                if nb not in data.keys():
                    continue

                data[k]['graph'][nb] = {}

                # print('>-------<')
                ref2_node, _, _ = find_nearest_edge(tree2, node2_ids, nb, UndiG)

                _, _, _, path = check_connection_and_distance(UndiG, qry2_node, ref2_node)

                if path is not None:
                    # Extract directed edges from G2 that exist along the path
                    path = [(p[1], p[0]) for p in path]
                    directed_edges = [
                        (
                            path[i], path[i+1]
                        ) if DiG.has_edge(path[i], path[i+1]) else (
                            path[i+1], path[i]
                        )
                        for i in range(len(path) - 1)
                    ]

                    data[k]['graph'][nb]['edges'] = directed_edges
                    data[k]['graph'][nb]['nodes'] = path
                else:
                    data[k]['graph'][nb]['edges'] = None
                    data[k]['graph'][nb]['nodes'] = None

        with open(f'data/processed.pkl', 'wb') as f:
            pickle.dump(data, f)

    # dict_keys(
    # ['neighbor', 'series', 'time', 'fclass',
    # 'width', 'neighbor_stats', 'common_time',
    # 'common_time_idx', 'nb_common_time_idx',
    # 'adj_series', 'top_25_series', 'top_25_time',
    # 'neighbor_path', 'avg', 'mean_w', 'max_zw', 'std_w']
    # )
    data, simplified_graphs = create_simplified_graphs(data)
    for station, graph in simplified_graphs.items():
        print("Station:", station)
        print("Nodes:", graph['nodes'])
        print("Edges:")
        for edge in graph['edges']:
            print("  ", edge)
        print()
        # visualize_graph_opencv(graph)

    with open('data/processed.pkl', 'wb') as f:
        pickle.dump(data, f)
    exit()
    
    # for k in data.keys():
    #     del data[k]['rainfall']

    #     data[k]['mean_w'] = {}
    #     data[k]['max_zw'] = {}
    #     data[k]['std_w'] = {}

    #     for nb in data[k]['neighbor']:
    #         if nb in data[k]['neighbor_path'] and 'area' in data[k]['neighbor_path'][nb]:
    #             areas = data[k]['neighbor_path'][nb]['area']
    #             distances = [
    #                 abs(e[0][1] - e[1][1]) + abs(e[0][0] - e[1][0]) for e in data[k]['neighbor_path'][nb]['edges']
    #             ]

    #             avg_widths = np.array([a / (d * 1000 * 1000) for a, d in zip(areas, distances)])
                
    #             if len(areas) > 0:
    #                 mean_w = np.mean(avg_widths)
    #                 std_w = np.std(avg_widths)

    #                 if std_w == 0:
    #                     max_zw = 0
    #                 else:
    #                     max_zw = ((avg_widths - mean_w) / std_w).max()
    #             else:
    #                 mean_w = -1
    #                 std_w = -1
    #                 max_zw = -1
    #         else:
    #             mean_w = -1
    #             std_w = -1
    #             max_zw = -1

    #         data[k]['mean_w'][nb] = mean_w
    #         data[k]['max_zw'][nb] = max_zw
    #         data[k]['std_w'][nb] = std_w
    
    # with open('all_data_new_median3.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    # exit()

    # list_p = [((4.2270005, 50.9436381), (4.2270833, 50.9436798)), ((4.2269236, 50.9435815), (4.2270005, 50.9436381)), ((4.2268816, 50.9435505), (4.2269236, 50.9435815)), ((4.2267746, 50.9434946), (4.2268816, 50.9435505)), ((4.2266833, 50.9434312), (4.2267746, 50.9434946)), ((4.2265782, 50.9433274), (4.2266833, 50.9434312)), ((4.2265249, 50.9432497), (4.2265782, 50.9433274))]
    # p1 = (3.8715394, 50.8040931)
    # p2 = (3.8718304, 50.8042185)
    # print(f"Load Data Time: {time.time() - start}")

    # for pp in list_p:
    #     p1, p2 = pp[0], pp[1]
    #     start = time.time()
    #     area = find_area_path(p1, p2, water)
    #     end = time.time()
    #     print(f"({(p1[1], p1[0])}, {(p2[1], p2[0])}) - Time: {end - start} - Total water surface area: {area:.2f} square meters")
    # exit()

    # for k in data.keys():
    #     average = np.median(data[k]['series'])
    #     data[k]['avg'] = average

    # with open('all_data_new_median2.pkl', 'wb') as f:
    #     pickle.dump(data, f)

    # for k in data.keys():
    #     average = np.median(data[k]['series'])
    #     x = [v - average for v in data[k]['series']]
    #     y = data[k]['time']

    #     threshold = np.percentile(x, 75)

    #     filtered_pairs = [(v, t) for v, t in zip(x, y) if v >= threshold]
    #     data[k]['top_25_series'], data[k]['top_25_time'] = zip(*filtered_pairs) if filtered_pairs else ([], [])
    # with open('all_data_new_median2.pkl', 'wb') as f:
    #     pickle.dump(data, f)
    # exit()

    

    with open('all_data_new_median3.pkl', 'wb') as f:
        pickle.dump(data, f)


def create_waterway_graph():
    waterways = gpd.read_file("data/Vhag_prj.shp")

    # Create an empty graph
    G = nx.DiGraph()  # Create a directed graph
    G2 = nx.Graph()  # Create a directed graph

    if not os.path.exists("data/graph.pickle"):
        # Step 1: Create a Spatial Index for the LineStrings
        sindex = waterways.sindex  # Build spatial index for fast lookups
        all_lines = waterways.geometry.tolist()
        intersection_points = set()  # Use a set to store unique intersection points

        # Step 2: Efficiently Find Intersections Using Spatial Index
        for idx, line in enumerate(all_lines):
            possible_matches_idx = list(sindex.intersection(line.bounds))  # Get bounding box matches
            possible_matches = [all_lines[i] for i in possible_matches_idx if i > idx]  # Avoid duplicates

            for other_line in possible_matches:
                if line.intersects(other_line):
                    intersection = line.intersection(other_line)

                    if intersection.geom_type == "Point":
                        intersection_points.add((intersection.x, intersection.y))  # Store as tuple
                    elif intersection.geom_type == "MultiPoint":
                        for pt in intersection.geoms:
                            intersection_points.add((pt.x, pt.y))  # Store unique points

        # Convert intersection points to Shapely Point objects and build spatial index
        intersection_points = [Point(x, y) for x, y in intersection_points]
        point_index = STRtree(intersection_points)  # Fast lookup for relevant points

        # Step 3: Optimized Splitting of LineStrings
        new_lines = []
        for line in all_lines:
            # Find only nearby intersection points for this line
            nearby_points = point_index.query(line)  # STRtree query finds relevant points fast
            
            # Only split if there are relevant points
            if len(nearby_points) > 0:
                split_line = split(line, MultiPoint([intersection_points[p] for p in nearby_points if isinstance(intersection_points[p], Point)]))  # Ensure only Points are included
                new_lines.extend(split_line.geoms)
            else:
                new_lines.append(line)  # Keep original line if no split needed

        # Step 4: Add Nodes & Edges to the Graph
        for geom in new_lines:
            if geom.geom_type == "LineString":
                lines = [geom]
            elif geom.geom_type == "MultiLineString":
                lines = geom.geoms
            else:
                continue  # skip unknown geometry types

            for line in lines:
                coords = list(line.coords)
                for i in range(len(coords) - 1):
                    point1, point2 = coords[i], coords[i + 1]
                    G.add_edge(point1, point2, weight=geodesic(point1, point2).km)
                    G2.add_edge(point1, point2, weight=geodesic(point1, point2).km)

        # Step 5: Ensure All Intersection Points Are Nodes
        for point in intersection_points:
            G.add_node((point.x, point.y))  # Ensure intersection nodes exist in the graph
            G2.add_node((point.x, point.y))  # Ensure intersection nodes exist in the graph

        with open("data/graph.pickle", "wb") as f:
            pickle.dump(G, f)
        
        with open("data/graph2.pickle", "wb") as f:
            pickle.dump(G2, f)
    else:
        with open("data/graph.pickle", "rb") as f:
            G = pickle.load(f)

        with open("data/graph2.pickle", "rb") as f:
            G2 = pickle.load(f)
    
    nodes = np.array(G.nodes)
    node_ids = list(G.nodes)
    tree = spatial.KDTree(nodes)

    nodes2 = np.array(G2.nodes)
    node2_ids = list(G2.nodes)
    tree2 = spatial.KDTree(nodes2)
    
    return waterways, G, node_ids, tree, G2, node2_ids, tree2


if __name__ == "__main__":
    main()