import numpy as np
import os

def get_tilename(id):
    tiles_path = os.path.join(os.path.dirname(__file__), 'tiles.csv')    
    tiles = open(tiles_path, "r")
    tilelist = tiles.read().split("\n")
    return tilelist[id].split(",")[1]

def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum('ij,ij->i', deltas, deltas)
    return np.argmin(dist_2)

def s2tile_point(lat, lng):
    points_path = os.path.join(os.path.dirname(__file__), 'points.csv')
    nodes = np.loadtxt(points_path, delimiter=',')
    return get_tilename(closest_node([lng,lat], nodes))

def s2tile_features(aoi_gdf: gpd.GeoDataFrame) -> list:

    tile_name_col: str = 'Name'
    grid_path = os.path.join(os.path.dirname(__file__), 's2mgrs_s2_index.geojson')
    mgrs_grid_gdf = gpd.read_file(grid_path)

    
    # ensure both are in the same CRS
    if aoi_gdf.crs != mgrs_grid_gdf.crs:
        #mgrs_grid_gdf = mgrs_grid_gdf.to_crs(aoi_gdf.crs)
        aoi_gdf = aoi_gdf.to_crs(mgrs_grid_gdf.crs)
    
    # merge all AOI geometries into one for faster intersection
    aoi_union = unary_union(aoi_gdf.geometry)
    
    # filter grid by intersection
    hits = mgrs_grid_gdf[mgrs_grid_gdf.geometry.intersects(aoi_union)]
    
    # return unique names
    return sorted(hits[tile_name_col].unique().tolist())

def s2_tile(*args):
    # Case 1: single GeoDataFrame → AOI mode
    if len(args) == 1 and isinstance(args[0], gpd.GeoDataFrame):
        return s2tile_features(args[0])

    # Case 2: two numeric args → LAT/LON
    if len(args) == 2 and all(isinstance(a, (int, float)) for a in args):
        print("LAT/LON")
        return s2tile_point(args[0], args[1])

    # Fallback for anything else
    raise TypeError(
        "s2_tile() accepts either:\n"
        "  • one GeoDataFrame, or\n"
        "  • a point (lat, lon)"
    )
    