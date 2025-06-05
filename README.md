### S2MGRS

A simple, lightweight, and fast Python package that returns intersecting Sentinel-2 MGRS tile codes for either:

    - a single latitude/longitude point, or
    - a GeoPandas GeoDataFrame (AOI).

USAGE:

from s2mgrs.s2tile import s2tile

# Call the function directly
tile_name = s2tile(13.7563, 100.5018)
print(f"Sentinel-2 tile at (13.7563, 100.5018): {tile_name}")

import geopandas as gpd

# Load a small GeoJSON polygon as GeoDataFrame
aoi_gdf = gpd.read_file("path/to/your/aoi_boundary.geojson")

# Get all Sentinel-2 MGRS tiles intersecting that polygon
tiles = s2tile(aoi_gdf)
print("Tiles covering AOI:", tiles)

TODO: 

- return WKT geometries for tiles


