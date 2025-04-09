import rasterio
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


def gps_to_pixel(transform, lat, lon):
    """ Convert GPS coordinates (lat, lon) to raster pixel (row, col). """
    col, row = ~transform * (lon, lat)  # Apply inverse affine transform
    row, col = int(round(row)), int(round(col))

    # print(f"GPS ({lat}, {lon}) → Pixel ({row}, {col})")  # Debugging output
    return row, col


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