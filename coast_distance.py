# pip install geopandas shapely pyproj pandas
import pickle
import pandas as pd
import geopandas as gpd
from shapely.ops import nearest_points


def parse_key_to_latlon(k):
    # Accept (lat, lon), (lon, lat), or "lat,lon"
    if isinstance(k, (tuple, list)) and len(k) == 2:
        a, b = float(k[0]), float(k[1])
    elif isinstance(k, str):
        parts = k.replace("(", "").replace(")", "").split(",")
        if len(parts) < 2:
            raise ValueError(f"Cannot parse coord from string key: {k}")
        a, b = float(parts[0].strip()), float(parts[1].strip())
    else:
        raise TypeError(f"Unsupported key type for coord: {type(k)} ({k})")

    is_lat = lambda x: -90 <= x <= 90
    is_lon = lambda x: -180 <= x <= 180

    # Heuristic to determine order
    if is_lat(a) and is_lon(b) and not is_lon(a):
        lat, lon = a, b
    elif is_lat(b) and is_lon(a) and not is_lon(b):
        lat, lon = b, a
    else:
        # Ambiguous: assume (lat, lon)
        lat, lon = a, b
    return lat, lon

# --- Inputs ---
SENSORS_CSV = "sensors.csv"          # columns: id, lat, lon
COASTLINE_FILE = "ne_10m_coastline/ne_10m_coastline.shp"  # Natural Earth 10m coastlines (download once)

# Belgian/Europe-friendly metric CRS (LAEA Europe). Distances in meters.
METRIC_CRS = "EPSG:3035"

# --- Load from pickle ---
with open('data/data_with_width.pkl', 'rb') as f:
    data = pickle.load(f)

keys = list(data.keys())

coords = [parse_key_to_latlon(k) for k in keys]
df = pd.DataFrame(coords, columns=["lat", "lon"])
df["key"] = keys  # preserve the original key for joining back later if needed

# Build GeoDataFrame (WGS84)
gdf_pts = gpd.GeoDataFrame(
    df,
    geometry=gpd.points_from_xy(df["lon"], df["lat"]),
    crs="EPSG:4326",
)

coast = gpd.read_file(COASTLINE_FILE)
# Keep only a reasonable region (optional but faster). For Europe:
coast = coast.to_crs("EPSG:4326")
coast_eur = coast.cx[-30:45, 30:72]  # rough lon/lat bounds for Europe
coast_eur = coast_eur.to_crs(METRIC_CRS)

# Project sensors
gdf_pts_m = gdf_pts.to_crs(METRIC_CRS)

# Union the coastline to a single geometry (ok for thousands of vertices; fast enough for ~1–10k points)
coast_union = coast_eur.unary_union  # MultiLineString/GeometryCollection

# --- Compute nearest coastline distance for each sensor ---
def dist_to_coast_m(point):
    # nearest point on the (multi)line to this sensor
    _, nearest_on_coast = nearest_points(point, coast_union)
    return point.distance(nearest_on_coast)  # meters in METRIC_CRS

gdf_pts_m["dist_to_coast_m"] = gdf_pts_m.geometry.apply(dist_to_coast_m)
gdf_pts_m["dist_to_coast_km"] = gdf_pts_m["dist_to_coast_m"] / 1000.0

# Back to WGS84 if you want to export
# out = gdf_pts_m.to_crs("EPSG:4326").drop(columns="geometry")

# --- Build dict keyed by the original GPS key ---
# Value includes both meters and kilometers (floats).
dist_by_key = {
    key: {
        "dist_m": float(dist_m),
        "dist_km": float(dist_km),
        "norm": float(dist_km) / 100.0,
    }
    for key, dist_m, dist_km in zip(
        gdf_pts_m["key"],
        gdf_pts_m["dist_to_coast_m"],
        gdf_pts_m["dist_to_coast_km"],
    )
}

# --- Save to pickle ---
out_pkl = "data/sensor_dist_to_coast.pkl"
with open(out_pkl, "wb") as f:
    pickle.dump(dist_by_key, f, protocol=pickle.HIGHEST_PROTOCOL)

print(f"Saved {len(dist_by_key)} distances to {out_pkl}")
# out.to_csv("sensors_with_coast_distance.csv", index=False)

# print(out.head())
