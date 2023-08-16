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

def s2mgrs(lat, lng):
    points_path = os.path.join(os.path.dirname(__file__), 'points.csv')    
    nodes = np.loadtxt(points_path, delimiter=',')
    return get_tilename(closest_node([lng,lat], nodes))
    