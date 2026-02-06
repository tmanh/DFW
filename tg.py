import geopandas as gpd
import numpy as np


def create_waterway_graph():
    waterways = gpd.read_file("data/osm/gis_osm_waterways_free_1.shp")

    print(waterways.columns)
    width = [v for v in list(waterways['width'].values) if v is not None]
    print(np.unique(width))


if __name__ == "__main__":
    create_waterway_graph()