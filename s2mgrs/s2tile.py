import numpy as np
import geopandas as gpd
from shapely.ops import unary_union

# importlib.resources is part of the stdlib from Python 3.7 onwards.
# If you must support Python 3.7–3.8 on a system without it,
# you can install the backport 'importlib_resources', but here we assume ≥3.9.
import importlib.resources as pkg_resources

# The 'data' is the subpackage/folder where your files live.
from . import data  


def get_tilename(idx: int) -> str:
    # Open "tiles.csv" from my_s2tile_module/data/ within the installed package.
    with pkg_resources.open_text(data, "tiles.csv") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    # Each line is "0,TILE_NAME" etc., so split on comma and return the second field.
    return lines[idx].split(",")[1]


def closest_node(node, nodes):
    nodes = np.asarray(nodes)
    deltas = nodes - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return np.argmin(dist_2)


def s2tile_point(lat: float, lng: float) -> str:
    # Open "points.csv" from the data folder as a text stream.
    with pkg_resources.open_text(data, "points.csv") as fh:
        nodes = np.loadtxt(fh, delimiter=",")
    idx = closest_node([lng, lat], nodes)
    return get_tilename(idx)


def s2tile_features(aoi_gdf: gpd.GeoDataFrame) -> list:
    tile_name_col = "Name"
    # Get a pathlib.Path–like object pointing to s2mgrs_s2_index.geojson
    geojson_path = pkg_resources.files(data) / "s2mgrs_s2_index.geojson"
    mgrs_grid_gdf = gpd.read_file(geojson_path)

    # Reproject if needed
    if aoi_gdf.crs != mgrs_grid_gdf.crs:
        aoi_gdf = aoi_gdf.to_crs(mgrs_grid_gdf.crs)

    # Merge all input geometries into one
    aoi_union = unary_union(aoi_gdf.geometry)
    hits = mgrs_grid_gdf[mgrs_grid_gdf.geometry.intersects(aoi_union)]
    return sorted(hits[tile_name_col].unique().tolist())


def s2tile(*args):
    # −− Case 1: single GeoDataFrame → AOI mode
    if len(args) == 1 and isinstance(args[0], gpd.GeoDataFrame):
        return s2tile_features(args[0])

    # −− Case 2: lat, lon as two numeric args
    if len(args) == 2 and all(isinstance(a, (int, float)) for a in args):
        return s2tile_point(args[0], args[1])

    raise TypeError(
        "s2tile() accepts either:\n"
        "  • one GeoDataFrame, or\n"
        "  • two numeric args (lat, lon)."
    )