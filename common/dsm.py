import math
import rasterio
import numpy as np

from rasterio.warp import transform as rasterio_transform
from rasterio.warp import calculate_default_transform, reproject, Resampling


def load_dsm(file_path):
    with rasterio.open(file_path) as src:
        print("CRS of the DSM file:", src.crs)

        dsm = src.read(1)  # Read first band (elevation)

        # Extract coordinates
        bounds = src.bounds  # (left, bottom, right, top)
        min_lon, min_lat = bounds.left, bounds.bottom
        max_lon, max_lat = bounds.right, bounds.top

        print(f"Min coordinates (lon/lat): ({min_lon}, {min_lat})")
        print(f"Max coordinates (lon/lat): ({max_lon}, {max_lat})")

        transform = src.transform  # Affine transform
        return dsm, transform, bounds
    

def reproject_dsm(input_path, output_path, target_crs="EPSG:4326"):
    with rasterio.open(input_path) as src:
        print("Original CRS:", src.crs)

        # Compute the transform and new shape for target CRS
        transform, width, height = calculate_default_transform(
            src.crs, target_crs, src.width, src.height, *src.bounds
        )

        # Define new metadata
        kwargs = src.meta.copy()
        kwargs.update({
            "crs": target_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        # Create output file
        with rasterio.open(output_path, "w", **kwargs) as dst:
            for i in range(1, src.count + 1):  # Reproject all bands
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=target_crs,
                    resampling=Resampling.bilinear
                )

        print(f"Reprojected DSM saved to {output_path}")
    

def downsample_dsm(dsm, factor=10):
    """ Downsample DSM by averaging every (factor x factor) block. """
    new_rows = dsm.shape[0] // factor
    new_cols = dsm.shape[1] // factor
    downsampled_dsm = dsm[:new_rows * factor, :new_cols * factor].reshape(new_rows, factor, new_cols, factor).mean(axis=(1, 3))
    return downsampled_dsm


def crop_dsm(dsm, start, end, margin=50):
    """ Crop the DSM around the bounding box of start & end points with a margin. """
    rows, cols = dsm.shape
    min_row = max(0, min(start[0], end[0]) - margin)
    max_row = min(rows, max(start[0], end[0]) + margin)
    min_col = max(0, min(start[1], end[1]) - margin)
    max_col = min(cols, max(start[1], end[1]) + margin)

    cropped_dsm = dsm[min_row:max_row, min_col:max_col]
    return cropped_dsm, (min_row, min_col)  # Return top-left offset


def gps_to_pixel(transform, lat, lon, raster_crs):
    # Convert (lat, lon) in EPSG:4326 to raster CRS (e.g. UTM)
    x, y = rasterio_transform("EPSG:4326", raster_crs, [lon], [lat])
    x, y = x[0], y[0]

    # Convert projected coordinates to pixel indices
    col, row = ~transform * (x, y)
    return int(round(row)), int(round(col))


def pixel_to_gps(transform, row, col):
    """ Convert DSM raster indices (row, col) back to GPS coordinates (lat, lon). """
    lon, lat = transform * (col, row)  # Apply affine transform
    return lat, lon


def find_tile_for_location(boundary_dict, loc):
    lon, lat = loc
    for key, (min_lon, min_lat, max_lon, max_lat) in boundary_dict.items():
        if min_lon <= lon <= max_lon and min_lat <= lat <= max_lat:
            return key
    return None


def haversine_distance(coord1, coord2):
    """
    Compute the great-circle distance between two GPS points using Haversine formula.
    Coordinates must be in (lat, lon) format in degrees.
    Returns distance in meters.
    """
    R = 6371000  # Radius of Earth in meters
    lat1, lon1 = coord1
    lat2, lon2 = coord2

    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

def total_distance(gps_coords):
    """
    gps_coords: list of (lat, lon) tuples
    Returns total distance in meters.
    """
    if len(gps_coords) < 2:
        return 0.0

    dist = 0.0
    for i in range(len(gps_coords) - 1):
        dist += haversine_distance(gps_coords[i], gps_coords[i+1])
    return dist / 1000.0


def compute_slope(dtm, cellsize=1.0):
    dy, dx = np.gradient(dtm, cellsize)
    slope_rad = np.arctan(np.hypot(dx, dy))
    return np.degrees(slope_rad)