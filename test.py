import geopandas as gpd

# Load shapefile
gdf = gpd.read_file("data/Vhag.shp")

# Manually assign the correct CRS (EPSG:31370)
gdf = gdf.set_crs(epsg=31370)

# Now convert to WGS84 (Google Maps CRS)
gdf_wgs84 = gdf.to_crs(epsg=4326)

# Save or view the reprojected data
print(gdf_wgs84.head())

# Optional: Save to new file
gdf_wgs84.to_file("data/Vhag_prj.shp")  # Still a shapefile

