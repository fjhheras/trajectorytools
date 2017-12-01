#import joblib #import Parallel, delayed
import scipy.spatial
from sklearn.neighbors import NearestNeighbors
import numpy as np

def _neighbours_indices_in_frame(positions, num_neighbours):
    nbrs = NearestNeighbors(n_neighbors=num_neighbours+1, algorithm='ball_tree').fit(positions)
    return nbrs.kneighbors(positions, return_distance = False)

def _in_convex_hull(positions):
    hull = scipy.spatial.ConvexHull(positions)
    convex_hull = np.zeros(positions.shape[0], dtype = np.bool)
    convex_hull[hull.vertices] = True
    return convex_hull

def radius_of_circumcentre(pa,pb,pc):
    # Sides of triangles
    a = np.linalg.norm(pb-pc, axis=-1) 
    b = np.linalg.norm(pc-pa, axis=-1) 
    c = np.linalg.norm(pa-pb, axis=-1) 
    # Area using Heron formula
    s = (a + b + c)/2 # Semiperimeter
    area = np.sqrt(s*(s-a)*(s-b)*(s-c))
    return a*b*c/(4*area)

def _in_alpha_shape(positions, alpha=10.0):
    num_individuals, _ = positions.shape
    delaunay = scipy.spatial.Delaunay(positions)
    pa, pb, pc = [], [], []
    for triangle in delaunay.simplices:
        pa.append(triangle[0])
        pb.append(triangle[1])
        pc.append(triangle[2])
    R = radius_of_circumcentre(np.stack(a), np.stack(b), np.stack(c))
    edges = np.zeros((num_individuals, num_individuals), dtype=np.int)
    #### TODO: Unfinished


def in_convex_hull(positions):
    convex_hull_list = [_in_convex_hull(positions_in_frame) for positions_in_frame in positions]
    return np.stack(convex_hull_list, axis = 0)

def give_indices(positions, num_neighbours):
    total_time_steps = positions.shape[0]
    individuals = positions.shape[1]
    next_neighbours = np.empty([total_time_steps, individuals, num_neighbours+1], dtype = np.int)
    for frame in range(total_time_steps):
        next_neighbours[frame,...] = _neighbours_indices_in_frame(positions[frame], num_neighbours) 
    return next_neighbours

def restrict(data, indices, individual=None):
    num_restricted = indices.shape[-1]
    total_time_steps = data.shape[0]
    num_individuals = data.shape[1]
    coordinates = data.shape[-1]
    
    if individual is None:
        output_data = np.empty([total_time_steps, num_individuals, num_restricted, coordinates])
        for frame in range(total_time_steps):
            output_data[frame,...] = data[frame, indices[frame,:],:]
    else:
        output_data = np.empty([total_time_steps, num_restricted, coordinates])
        for frame in range(total_time_steps):
            output_data[frame,...] = data[frame, indices[frame,individual],:]
 
    return output_data




