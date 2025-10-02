import numpy as np  # numerical operations
import geopandas as gpd  # for vector spatial data handling
from shapely.ops import unary_union  # to merge geometries
from shapely.geometry import box, Point  # for bounding boxes and points
from typing import List, Dict

# importlib.resources for accessing package data files
import importlib.resources as pkg_resources
# 'data' should be the subpackage where your geojson and CSV files reside
from . import data


def get_tilename(idx: int) -> str:
    """
    Return the MGRS tile name corresponding to a given index.
    Reads 'tiles.csv' from the data package, where each line is 'index,TILE_NAME'.
    """
    # Open the CSV as text and strip empty lines
    with pkg_resources.open_text(data, "tiles.csv") as fh:
        lines = [line.strip() for line in fh if line.strip()]
    # Split on comma and return the second field (tile name)
    return lines[idx].split(",")[1]


def closest_node(node: List[float], nodes: np.ndarray) -> int:
    """
    Find the index of the closest node in 'nodes' to the given 'node' coordinate.
    Uses squared Euclidean distance for efficiency.
    """
    nodes_arr = np.asarray(nodes)
    deltas = nodes_arr - node
    dist_2 = np.einsum("ij,ij->i", deltas, deltas)
    return int(np.argmin(dist_2))


def s2tile_point(lat: float, lng: float) -> str:
    """
    Map a lat/lon point to the nearest precomputed grid point in 'points.csv',
    then return its corresponding tile name.
    """
    # Load array of node coordinates (lon,lat) from CSV
    with pkg_resources.open_text(data, "points.csv") as fh:
        nodes = np.loadtxt(fh, delimiter=",")
    # Find nearest node index and get tile name
    idx = closest_node([lng, lat], nodes)
    return get_tilename(idx)


def s2tiles_from_point(lat: float, lon: float) -> List[str]:
    """
    Return all Sentinel-2 MGRS tile codes covering a single lat/lon point.
    Reads the MGRS grid GeoJSON and finds containing polygons.
    """
    tile_name_col = "Name"

    # Load the MGRS grid GeoJSON via package resources
    geojson_path = pkg_resources.files(data) / "s2mgrs_s2_index.geojson"
    mgrs_grid_gdf = gpd.read_file(geojson_path)

    # Create a GeoDataFrame for the input point (WGS84)
    pt_gdf = gpd.GeoDataFrame(
        geometry=[Point(lon, lat)],
        crs="EPSG:4326"
    )

    # Reproject point to match grid CRS if needed
    if pt_gdf.crs != mgrs_grid_gdf.crs:
        pt_gdf = pt_gdf.to_crs(mgrs_grid_gdf.crs)

    # Find grid cells that contain the point geometry
    hits = mgrs_grid_gdf[mgrs_grid_gdf.geometry.contains(pt_gdf.geometry.iloc[0])]

    # Return sorted unique tile names
    return sorted(hits[tile_name_col].unique().tolist())


def s2tile_features(aoi_gdf: gpd.GeoDataFrame) -> List[str]:
    """
    Return all Sentinel-2 MGRS tile codes intersecting the given AOI GeoDataFrame.
    Merges all geometries in 'aoi_gdf' and performs spatial intersection with the grid.
    """
    tile_name_col = "Name"

    # Load the MGRS grid
    geojson_path = pkg_resources.files(data) / "s2mgrs_s2_index.geojson"
    mgrs_grid_gdf = gpd.read_file(geojson_path)

    # Reproject AOI to grid CRS if needed
    if aoi_gdf.crs != mgrs_grid_gdf.crs:
        aoi_gdf = aoi_gdf.to_crs(mgrs_grid_gdf.crs)

    # Merge all input geometries into one union
    aoi_union = unary_union(aoi_gdf.geometry)

    # Find intersecting grid cells
    hits = mgrs_grid_gdf[mgrs_grid_gdf.geometry.intersects(aoi_union)]
    # Calculate the extent of intersection as percentages for each mgrs tile.
    result = []
    for _, row in hits.iterrows():
        tile_geom = row.geometry
        tile_code = row[tile_name_col]
        intersection = tile_geom.intersection(aoi_union)
        if intersection.is_empty:
            continue
        intersection_area = intersection.area
        tile_area = tile_geom.area
        percentage = (intersection_area / tile_area) * 100 if tile_area > 0 else 0

        result.append((tile_code, percentage))
    sorted_results = sorted(result, key=lambda x: (-x[1], x[0]))
    return [code for code, _ in sorted_results]


def get_countries(tile_codes: List[str]) -> Dict[str, List[str]]:
    """
    Given a list of MGRS tile codes, return a mapping from each code
    to the list of country names (with ISO2) intersecting that tile.
    Reads both 's2mgrs_s2_index.geojson' and 'countries.geojson'.
    """
    tile_name_col = "Name"

    # Paths to the GeoJSON data files
    mgrs_geojson = pkg_resources.files(data) / "s2mgrs_s2_index.geojson"
    countries_geojson = pkg_resources.files(data) / "countries.geojson"

    # Load GeoDataFrames
    mgrs_grid = gpd.read_file(mgrs_geojson)
    countries = gpd.read_file(countries_geojson)

    # Ensure both layers share the same CRS for spatial operations
    if countries.crs != mgrs_grid.crs:
        countries = countries.to_crs(mgrs_grid.crs)

    result: Dict[str, List[str]] = {}

    # Loop through each tile code
    for tile in tile_codes:
        # Select the grid cell for the given tile name
        cell = mgrs_grid[mgrs_grid[tile_name_col] == tile]
        if cell.empty:
            result[tile] = []
            continue

        # Merge multiple parts into a single geometry if needed
        cell_geom = unary_union(cell.geometry)

        # Find intersecting countries
        hits = countries[countries.geometry.intersects(cell_geom)]

        short_output = True
        countries_list: List[str] = []
        for _, row in hits.iterrows():
            # Attempt to read standard fields for name and ISO code
            name = row.get('ADMIN') or row.get('name')
            iso = row.get('ISO_A2_EH') or row.get('ISO3166-1:alpha2')
            if short_output:
                if iso:
                    countries_list.append(f"{iso}")
            else:
                if name and iso:
                    countries_list.append(f"{name} ({iso})")
                elif name:
                    countries_list.append(name)

        # Deduplicate and sort country list
        result[tile] = sorted(set(countries_list))

    return result


def get_geometry(tilename):
    """
    Return a GeoDataFrame with one row per requested MGRS tile name.

    Accepts:
      - a single tile name string, e.g. "47PPR"
      - an iterable/list/tuple of tile name strings, e.g. ["47PPR", "12UUA"]

    Returns:
      - geopandas.GeoDataFrame with columns:
          - "Name": requested tile name (keeps order of the input)
          - "geometry": merged shapely geometry for the tile, or None if not found
      - The returned GeoDataFrame uses the same CRS as the MGRS grid.

    Examples:
      get_geometry("47PPR")
      get_geometry(["47PPR", "12UUA"])
    """
    tile_name_col = "Name"
    mgrs_geojson = pkg_resources.files(data) / "s2mgrs_s2_index.geojson"
    mgrs_grid = gpd.read_file(mgrs_geojson)

    # Normalize input to a list preserving order
    if isinstance(tilename, str):
        tiles = [tilename]
    elif isinstance(tilename, (list, tuple, np.ndarray)):
        tiles = list(tilename)
    else:
        raise TypeError("tilename must be a str or a list/tuple/ndarray of str")

    rows = []
    for t in tiles:
        cell = mgrs_grid[mgrs_grid[tile_name_col] == t]
        if cell.empty:
            rows.append({"Name": t, "geometry": None})
            continue

        # Merge multiple parts into a single geometry if needed
        cell_geom = unary_union(cell.geometry)
        rows.append({"Name": t, "geometry": cell_geom})

    gdf = gpd.GeoDataFrame(rows, geometry="geometry", crs=mgrs_grid.crs)
    return gdf


def s2tile(*args):
    """
    Unified entry point for tile lookup:
    - s2tile(aoi_gdf) -> list of tiles intersecting a GeoDataFrame
    - s2tile((w, s, e, n)) -> list of tiles intersecting bbox tuple
    - s2tile(lat, lon) -> list of tiles covering a point
    """
    # Case 1: GeoDataFrame AOI mode
    if len(args) == 1 and isinstance(args[0], gpd.GeoDataFrame):
        return s2tile_features(args[0])

    # Case 2: bbox tuple/list mode
    if (
        len(args) == 1
        and isinstance(args[0], (tuple, list))
        and len(args[0]) == 4
        and all(isinstance(c, (int, float)) for c in args[0])
    ):
        w, s, e, n = args[0]
        bbox_gdf = gpd.GeoDataFrame(
            {"geometry": [box(w, s, e, n)]},
            crs="EPSG:4326"
        )
        return s2tile_features(bbox_gdf)

    # Case 3: point mode
    if len(args) == 2 and all(isinstance(a, (int, float)) for a in args):
        return s2tiles_from_point(lat=args[0], lon=args[1])

    # Invalid usage
    raise TypeError(
        "s2tile() accepts either:\n"
        "  • one GeoDataFrame, or\n"
        "  • one 4-tuple/list of (w, s, e, n), or\n"
        "  • two numeric args (lat, lon)."
    )
