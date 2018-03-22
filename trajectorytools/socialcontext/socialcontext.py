#import joblib #import Parallel, delayed
import math
import scipy.spatial
from sklearn.neighbors import NearestNeighbors
import numpy as np

def _in_convex_hull(positions):
    hull = scipy.spatial.ConvexHull(positions)
    convex_hull = np.zeros(positions.shape[0], dtype = np.bool)
    convex_hull[hull.vertices] = True
    return convex_hull

def in_convex_hull(positions):
    convex_hull_list = [_in_convex_hull(positions_in_frame) for positions_in_frame in positions]
    return np.stack(convex_hull_list, axis = 0)

def circumradius(points):
    '''Radius of the circumcentre
    defined by three points.
    '''
    # Sides of triangles
    side_vectors = points - np.roll(points,1,axis=-2)
    sides = np.sqrt(side_vectors[...,0]**2 + side_vectors[...,1]**2)
    a = sides[...,0]
    b = sides[...,1]
    c = sides[...,2]
    s = (a+b+c)/2
    area = np.sqrt((s-a)*(s-b)*(s-c)*s)
    return a*b*c/(4*area)

def _in_alpha_border(positions, alpha=5):
    '''Calculate vertices in border of alpha-shape
    by pruning a Delaunay triangulation.

    Border points are either:
    1. In convex hull
    2. In rejected triangles
    '''
    def radius_too_large(triangles):
        return circumradius(triangles) > 1/alpha
    num_individuals, _ = positions.shape
    delaunay = scipy.spatial.Delaunay(positions)
    triangles = np.array([positions[triangle] for triangle in delaunay.simplices])
    in_rejected_triangles = radius_too_large(triangles)
    points_in_rejected_triangles = [p for triangle in delaunay.simplices[in_rejected_triangles] for p in triangle]
    in_border = np.zeros(num_individuals, np.bool)
    in_border[points_in_rejected_triangles] = True
    in_border[delaunay.convex_hull] = True
    return in_border

def in_alpha_border(positions):
    alpha_border_list = [_in_alpha_border(positions_in_frame) for positions_in_frame in positions]
    return np.stack(alpha_border_list, axis = 0)

### LOCAL NEIGHBOURS

def _neighbours_indices_in_frame(positions, num_neighbours):
    nbrs = NearestNeighbors(n_neighbors=num_neighbours+1, algorithm='ball_tree').fit(positions)
    return nbrs.kneighbors(positions, return_distance = False)

def give_indices(positions, num_neighbours):
    total_time_steps = positions.shape[0]
    individuals = positions.shape[1]
    next_neighbours = np.empty([total_time_steps, individuals, num_neighbours+1], dtype = np.int)
    for frame in range(total_time_steps):
        next_neighbours[frame,...] = _neighbours_indices_in_frame(positions[frame], num_neighbours)
    return next_neighbours

def restrict(data, indices, individual=None, delay = 0):
    num_restricted = indices.shape[-1]
    total_time_steps = data.shape[0]
    num_individuals = data.shape[1]
    coordinates = data.shape[-1]

    if individual is None:
        output_data = np.empty([total_time_steps, num_individuals, num_restricted, coordinates])
        for frame in range(total_time_steps):
            output_data[frame,...] = data[frame - delay, indices[frame,:],:]
    else:
        output_data = np.empty([total_time_steps, num_restricted, coordinates])
        for frame in range(total_time_steps):
            output_data[frame,...] = data[frame - delay, indices[frame,individual],:]

    return output_data
