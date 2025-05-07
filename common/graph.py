import heapq
import cv2
import numpy as np

from shapely import LineString
from skimage.draw import line
from scipy.ndimage import binary_dilation, label
from scipy.ndimage import distance_transform_edt
from skimage.morphology import skeletonize
from shapely.geometry import LineString, Polygon, Point
from shapely.ops import split
from skimage.segmentation import watershed
from scipy import ndimage as ndi


def dijkstra_water_cost(dsm, start, end):
    """ Dijkstra's Algorithm to find the best water flow path, using elevation cost without blocking movement. """
    rows, cols = dsm.shape

    def get_neighbors(i, j):
        """ Return valid 4-way neighbors, computing cost based on elevation difference. """
        neighbors = []
        current_elevation = dsm[i, j]

        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-way movement
            ni, nj = i + di, j + dj

            if 0 <= ni < rows and 0 <= nj < cols:
                neighbor_elevation = dsm[ni, nj]
                diff = (neighbor_elevation - current_elevation)

                cost = diff + 0.1
                neighbors.append(((ni, nj), cost))

        return neighbors

    # Priority queue (total cost, position)
    pq = [(0, start)]
    costs = {start: 0}
    predecessors = {start: None}

    while pq:
        current_cost, current = heapq.heappop(pq)

        if current == end:
            break  # Goal reached

        for neighbor, move_cost in get_neighbors(*current):
            new_cost = current_cost + move_cost

            if neighbor not in costs or new_cost < costs[neighbor]:
                costs[neighbor] = new_cost
                heapq.heappush(pq, (new_cost, neighbor))  # No heuristic bias!
                predecessors[neighbor] = current

    # Reconstruct path
    path = []
    current = end
    while current is not None:
        path.append(current)
        current = predecessors.get(current)

    path.reverse()
    return [(p[0], p[1]) for p in path]  # Convert back to full DSM indices


def find_line_path(start, end):
    """
    Generate points along the straight line between start and end.

    Args:
        start (tuple): (row, col) start point.
        end (tuple): (row, col) end point.
        num_points (int): Number of interpolated points along the line.

    Returns:
        List of (x, y) points in world coordinates.
    """
    # Ensure integer pixel coordinates
    start_i, start_j = map(int, start)
    end_i, end_j = map(int, end)

    # Get pixel coordinates the line traverses
    rr, cc = line(start_i, start_j, end_i, end_j)

    return [(i, j) for i, j in zip(rr, cc)]


def interpolate_path(cropped_dsm, coarse_path):
    """ Connect the points in the full-resolution DSM using line interpolation. """
    refined_path = []
    
    for i in range(len(coarse_path) - 1):
        r1, c1 = coarse_path[i]
        r2, c2 = coarse_path[i + 1]

        # Get all pixels along the line between two coarse points
        rr, cc = line(r1, c1, r2, c2)  # Bresenham's Line Algorithm

        # Add these points to the final path
        for r, c in zip(rr, cc):
            if 0 <= r < cropped_dsm.shape[0] and 0 <= c < cropped_dsm.shape[1]:
                refined_path.append((r, c))
    return refined_path


def expand_waterway(dsm, path, elevation_threshold=1.0, max_iterations=3):
    """
    Expand the waterway from the detected shortest path until it reaches 
    a significant rise in elevation (riverbanks), using the average elevation 
    of the path.
    
    :param dsm: Digital Surface Model (DSM)
    :param path: List of (row, col) coordinates representing the water path
    :param elevation_threshold: Maximum elevation ratio allowed for expansion
    :param max_iterations: Limit the number of iterations to prevent infinite expansion
    :return: Binary mask of the expanded waterway region
    """
    # Preallocate waterway region
    waterway_region = np.zeros_like(dsm, dtype=bool)
    
    # Convert path to a NumPy array for fast indexing
    path_arr = np.array(path, dtype=int)
    
    # Compute the mean elevation along the path (vectorized)
    path_elevations = dsm[path_arr[:, 0], path_arr[:, 1]]
    mean_elevation = np.mean(path_elevations)
    
    # Mark initial path as waterway
    waterway_region[path_arr[:, 0], path_arr[:, 1]] = True
    
    # Compute valid expansion mask once
    valid_expansion = (dsm / mean_elevation) < elevation_threshold
    
    # Precompute the dilation structure (7x7, as given)
    structure = np.array([
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0],
        [1, 1, 1, 1, 1, 1, 1],
        [0, 1, 1, 1, 1, 1, 0],
        [0, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
    ], dtype=bool)
    
    # Iterative expansion using binary dilation
    for _ in range(max_iterations):
        # Perform dilation with 4 iterations at once
        expanded = binary_dilation(waterway_region, structure=structure, iterations=4)
        # Only consider valid expansions that are not already part of the waterway
        new_waterway = expanded & valid_expansion & ~waterway_region
        
        # If nothing new was added, break out early
        if not np.any(new_waterway):
            break
        
        # Update the waterway mask
        waterway_region |= new_waterway
    
    # Label connected components and select the one containing the original waterway
    labeled_array, _ = label(waterway_region)
    # Use the first coordinate in the path to determine the waterway label
    path_label = labeled_array[path_arr[0, 0], path_arr[0, 1]]
    waterway_region = (labeled_array == path_label)
    
    return waterway_region


def get_perpendicular_unit_vector(p1, p2):
    """Returns a unit vector perpendicular to the line from p1 to p2."""
    vec = np.array(p2) - np.array(p1)
    perp = np.array([-vec[1], vec[0]])  # Rotate 90 degrees
    return perp / np.linalg.norm(perp)


def order_quad_points_clockwise(quad):
    """
    Given 4 (x, y) points, order them clockwise around their centroid.
    
    Args:
        quad: (4, 2) array of (x, y) points

    Returns:
        reordered_quad: (4, 2) array ordered clockwise
    """
    center = np.mean(quad, axis=0)

    def angle(p):
        return np.arctan2(p[1] - center[1], p[0] - center[0])

    ordered = sorted(quad, key=angle)
    return np.array(ordered, dtype=np.int32)


def mask_between_perpendicular_cuts(mask_shape, line_pixels, expand=5000):
    """
    Build a mask that selects only pixels between two perpendicular cuts
    at the start and end of a centerline path.
    """
    """
    Build a mask that selects only pixels between two perpendicular cuts
    placed at the start and end of a centerline path.
    """
    line_pixels = np.array(line_pixels)
    start = np.array(line_pixels[0], dtype=np.float32)
    end = np.array(line_pixels[-1], dtype=np.float32)

    # Local directions
    dir_start = line_pixels[1] - line_pixels[0]
    dir_end = line_pixels[-1] - line_pixels[-2]

    # Unit perpendiculars
    perp_start = get_perpendicular_unit_vector(start + dir_start, start)
    perp_end = get_perpendicular_unit_vector(end - dir_end, end)

    # Dynamically adjust cap lengths
    s1 = min(expand, max_safe_scale(start,  perp_start, mask_shape))
    s2 = min(expand, max_safe_scale(start, -perp_start, mask_shape))
    s3 = min(expand, max_safe_scale(end,    perp_end,  mask_shape))
    s4 = min(expand, max_safe_scale(end,   -perp_end,  mask_shape))

    p1a = start + perp_start * s1
    p1b = start - perp_start * s2
    p2a = end   + perp_end   * s3
    p2b = end   - perp_end   * s4
    
    # Make sure all points are int32 for OpenCV
    p1a = np.round(p1a).astype(np.int32)
    p1b = np.round(p1b).astype(np.int32)
    p2a = np.round(p2a).astype(np.int32)
    p2b = np.round(p2b).astype(np.int32)

    # Round and convert to (x, y)
    quad = np.array([
        [p1a[1], p1a[0]],
        [p2a[1], p2a[0]],
        [p2b[1], p2b[0]],
        [p1b[1], p1b[0]]
    ], dtype=np.int32)

    quad = order_quad_points_clockwise(quad)

    mask = np.zeros(mask_shape, dtype=np.uint8)
    cv2.fillPoly(mask, [quad], 1)

    return mask.astype(bool)


def perpendicular_cut_lines(centerline, length):
    """
    Generate two perpendicular cut lines at the start and end of a centerline.
    """
    def perp_line(pt1, pt2):
        direction = np.array(pt2) - np.array(pt1)
        direction = direction / np.linalg.norm(direction)
        perp = np.array([-direction[1], direction[0]])
        center = np.array(pt1)
        p1 = center + perp * length
        p2 = center - perp * length
        return LineString([tuple(p1), tuple(p2)])

    start_cut = perp_line(centerline[0], centerline[1])
    end_cut = perp_line(centerline[-1], centerline[-2])
    return start_cut, end_cut


def clip_polygon_with_cuts(polygon, start_cut, end_cut):
    """
    Clip a polygon using two perpendicular cut lines (first intersection pair only).
    """
    parts = split(polygon, start_cut)
    for part in parts:
        if part.intersects(end_cut):
            final_parts = split(part, end_cut)
            return max(final_parts, key=lambda p: p.area) if final_parts else None
    return None


def watershed_clip_contour(dsm, centerline_coords, cut_length=100):
    """
    Watershed segmentation followed by contour clipping with perpendicular cuts.
    """
    centerline_mask = np.zeros_like(dsm, dtype=np.uint8)
    pts = np.array([[int(round(x)), int(round(y))] for x, y in centerline_coords])
    cv2.polylines(centerline_mask, [pts], isClosed=False, color=1, thickness=1)

    markers = np.zeros_like(dsm, dtype=np.int32)
    markers[centerline_mask > 0] = 2
    markers[0, :] = markers[-1, :] = markers[:, 0] = markers[:, -1] = 1

    gradient = ndi.gaussian_gradient_magnitude(dsm, sigma=1)
    labels = watershed(gradient, markers=markers)
    water_mask = (labels == 2).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(water_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(dsm, dtype=np.uint8)

    # Assume largest contour is the water region
    contour = max(contours, key=cv2.contourArea)
    contour_poly = Polygon([tuple(pt[0]) for pt in contour])

    # Create perpendicular cut lines
    start_cut, end_cut = perpendicular_cut_lines(centerline_coords, length=cut_length)

    # Clip polygon
    clipped_poly = clip_polygon_with_cuts(contour_poly, start_cut, end_cut)
    if clipped_poly is None:
        return np.zeros_like(dsm, dtype=np.uint8)

    # Rasterize clipped polygon
    final_mask = np.zeros_like(dsm, dtype=np.uint8)
    clipped_pts = np.array([[[int(x), int(y)]] for x, y in clipped_poly.exterior.coords])
    cv2.fillPoly(final_mask, [clipped_pts], 1)

    return final_mask


def max_safe_scale(point, direction, shape):
    """
    Scale 'direction' vector from 'point' so that point + direction * scale
    stays within image bounds.
    
    Args:
        point: np.array([row, col])
        direction: np.array([dy, dx]) (unit vector)
        shape: (height, width)
    
    Returns:
        max_scale: scalar value such that point + direction * max_scale stays in bounds
    """
    scale = np.inf

    for i in range(2):  # 0=row, 1=col
        if direction[i] > 0:
            limit = (shape[i] - 1 - point[i]) / direction[i]
        elif direction[i] < 0:
            limit = -point[i] / direction[i]
        else:
            limit = np.inf
        scale = min(scale, limit)

    return scale


def draw_line(shape, points, thickness=1):
    mask = np.zeros(shape, np.uint8)
    pts = np.array([[c, r] for r, c in points], dtype=np.int32).reshape((-1, 1, 2))
    cv2.polylines(mask, [pts], isClosed=False, color=255, thickness=thickness)
    return mask


def expand_water_surface_from_line(dsm, flattened, line_pixels, elevation_threshold=0.5):
    """
    Efficiently expand from line pixels to surrounding low-elevation pixels.
    """
    final_mask = np.zeros(dsm.shape, dtype=bool)
    valid = draw_line(dsm.shape, flattened, thickness=75).astype(bool)

    # Pre-compute elevation means with rolling windows for performance
    elev_means = []

    num_lines = len(line_pixels)
    for idx in range(num_lines):
        window_indices = line_pixels[max(idx - 5, 0): min(idx + 5, num_lines)]
        multi_segments = [pt for segment in window_indices for pt in segment]

        elev_mean = np.mean([dsm[i, j] for i, j in multi_segments])
        elev_means.append(elev_mean)

    for idx, l in enumerate(line_pixels):
        elev_mean = elev_means[idx]
        threshold_mask = (dsm < elev_mean * elevation_threshold) & valid

        current_line = l.copy()
        if idx < num_lines - 1 and line_pixels[idx + 1][1] not in current_line:
            current_line.append(line_pixels[idx + 1][1])

        cut_mask = mask_between_perpendicular_cuts(dsm.shape, current_line)
        cut_mask &= threshold_mask

        labeled, _ = label(cut_mask)
        labels_to_keep = set(labeled[i, j] for i, j in current_line if labeled[i, j] > 0)

        component_mask = np.isin(labeled, list(labels_to_keep))

        merged_mask = np.logical_or(component_mask, final_mask)
        final_mask |= merged_mask

    return final_mask & valid


def waterway_surface_mask(dsm, centerline_coords):
    """
    Tile-based watershed segmentation using centerline as seed.
    Segmentation is run independently on each tile and results are merged.
    """

    """
    Tile-based watershed segmentation using filtered and sampled centerline as seed.
    Segmentation is run independently on each tile and results are merged.
    """
    markers = np.zeros(dsm.shape, dtype=np.int32)

    # Filter out centerline points with outlier elevations (top/bottom 5%)
    elevations = np.array([dsm[int(round(x)), int(round(y))] for x, y in centerline_coords])

    foreground, background = 2, 1
    markers[dsm > np.mean(elevations) * 1.1] = background

    for x, y in centerline_coords:
        markers[x, y] = foreground

    gradient = ndi.gaussian_gradient_magnitude(dsm, sigma=1).astype(np.float32)
    labels = watershed(gradient.astype(np.float32), markers=markers.astype(np.int32))

    # Extract waterway mask (label==2)
    waterway_mask = (labels == 2)

    cv2.imwrite('mask.jpg', (waterway_mask * 255).astype(np.uint8))
    cv2.imwrite('markers.jpg', (markers * 255).astype(np.uint8))

    return waterway_mask, (markers == foreground)


def compute_waterway_statistics(waterway_region, skeleton, pixel_size):
    """
    Compute the average width, standard deviation, and maximum Z-score
    of the waterway region.

    The method uses the Euclidean distance transform and skeletonization.
    Each skeleton pixel's distance value approximates half the local width.
    The full width at that location is 2 * distance * pixel_size.
    
    :param waterway_region: Binary mask (2D numpy array) of the waterway.
    :param pixel_size: Size of one pixel in meters.
    :return: Tuple (avg_width, std_width, max_z) where:
             - avg_width is the average waterway width in meters,
             - std_width is the standard deviation of the width (m),
             - max_z is the maximum Z-score among the width values.
    """
    # Ensure waterway_region is boolean.
    waterway_region = waterway_region.astype(bool)
    
    # Compute the Euclidean distance transform (in pixels).
    dist = distance_transform_edt(waterway_region)
    
    # If the skeleton is empty, return zeros.
    if np.sum(skeleton) == 0:
        return 0.0, 0.0, 0.0
    
    # Extract the distance values along the skeleton.
    half_width_pixels = dist[skeleton]
    
    # Compute the full widths in meters.
    widths = 2 * half_width_pixels * pixel_size
    
    # Compute average and standard deviation of the widths.
    avg_width = np.mean(widths)
    std_width = np.std(widths)

    # Compute z-scores for each skeleton pixel.
    # Avoid division by zero if std_width is zero.
    if std_width > 0:
        z_scores = (widths - avg_width) / std_width
    else:
        z_scores = np.zeros_like(widths)
    z_scores = np.abs(z_scores)

    # Compute statistics on the z-scores.
    max_z = np.max(z_scores)
    mean_z = np.mean(z_scores)
    min_z = np.min(z_scores)
    std_z = np.std(z_scores)
    
    return avg_width, std_width, max_z, mean_z, min_z, std_z


def update_waterway_path(waterway_region):
    """
    Given a binary waterway region, compute its skeleton and extract an ordered, 
    smoother centerline using graph connectivity and Dijkstra's algorithm.
    
    Steps:
      1. Compute the skeleton (medial axis) of the waterway region.
      2. Build a connectivity graph using 8-neighbor connectivity from the skeleton pixels.
         Each edge is weighted by the Euclidean distance between pixels.
      3. Identify endpoints (pixels with only one neighbor). If there are at least two endpoints,
         choose the pair with the maximum Euclidean distance between them.
      4. Compute the shortest path (by total distance) between the selected endpoints using Dijkstra's algorithm.
      5. Return the ordered list of (row, col) coordinates (the waterway centerline) and the skeleton.
    
    :param waterway_region: Binary (boolean) mask of the expanded waterway.
    :return: Tuple (ordered_path, skeleton) where ordered_path is a list of (row, col) coordinates.
    """
    # Step 1: Compute the skeleton.
    skeleton = skeletonize(waterway_region.astype(bool))
    
    # Extract coordinates of skeleton pixels.
    coords = np.argwhere(skeleton)
    coords_list = [tuple(pt) for pt in coords]
    coords_set = set(coords_list)
    if not coords_list:
        return [], skeleton
    
    # Step 2: Build the connectivity graph with 8-neighbor connectivity.
    graph = {}
    for pt in coords_list:
        r, c = pt
        neighbors = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                neighbor = (r + dr, c + dc)
                if neighbor in coords_set:
                    # Use Euclidean distance as the edge weight.
                    weight = np.hypot(dr, dc)
                    neighbors.append((neighbor, weight))
        graph[pt] = neighbors

    # Step 3: Pick a starting node.
    # Try to choose an endpoint (a node with only one neighbor) if possible.
    endpoints = [pt for pt, nbrs in graph.items() if len(nbrs) == 1]
    if endpoints:
        start_node = endpoints[0]
    else:
        start_node = coords_list[0]

    # Define a helper: Dijkstra's algorithm on our skeleton graph.
    def dijkstra(start, graph, nodes):
        dist = {pt: float('inf') for pt in nodes}
        prev = {pt: None for pt in nodes}
        dist[start] = 0.0
        heap = [(0.0, start)]
        while heap:
            current_d, pt = heapq.heappop(heap)
            if current_d > dist[pt]:
                continue
            for neighbor, weight in graph[pt]:
                new_d = current_d + weight
                if new_d < dist[neighbor]:
                    dist[neighbor] = new_d
                    prev[neighbor] = pt
                    heapq.heappush(heap, (new_d, neighbor))
        return dist, prev

    # Phase 1: Run Dijkstra from start_node to find the farthest node.
    dist1, _ = dijkstra(start_node, graph, coords_list)
    farthest_node = max(dist1, key=dist1.get)
    
    # Phase 2: Run Dijkstra from the farthest_node to get the longest path (diameter).
    dist2, prev2 = dijkstra(farthest_node, graph, coords_list)
    other_end = max(dist2, key=dist2.get)
    
    # Reconstruct the diameter path.
    path = []
    current = other_end
    while current is not None:
        path.append(current)
        current = prev2[current]
    path.reverse()
    
    return path, skeleton