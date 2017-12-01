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

def _quick_distance(x,y):
    return math.sqrt(math.pow(x[0]-y[0],2) + math.pow(x[1]-y[1], 2) )

def circumradius(pa,pb,pc):
    '''Radius of the circumcentre
    defined by three points
    '''
    # Sides of triangles
    a = _quick_distance(pb, pc)
    b = _quick_distance(pc, pa)
    c = _quick_distance(pa, pb)
    # Area using Heron formula
    s = (a + b + c)/2 # Semiperimeter
    area = math.sqrt(s*(s-a)*(s-b)*(s-c))
    return a*b*c/(4*area)

def _in_alpha_border(positions, alpha=5):
    '''Calculate vertices in border of alpha-shape
    by pruning a Delaunay triangulation

    Based on:
    https://sgillies.net/2012/10/13/the-fading-shape-of-alpha.html
    '''
    num_individuals, _ = positions.shape
    delaunay = scipy.spatial.Delaunay(positions)
    edges = np.zeros((num_individuals, num_individuals), dtype=np.int)
    edges_in_delaunay = np.zeros((num_individuals, num_individuals), dtype=np.bool)
    for triangle in delaunay.simplices:
        i,j,k = triangle
        edges_in_delaunay[i,j] = True
        edges_in_delaunay[j,i] = True
        edges_in_delaunay[i,k] = True
        edges_in_delaunay[k,i] = True
        edges_in_delaunay[k,j] = True
        edges_in_delaunay[j,k] = True
        if circumradius(*positions[triangle]) < 1/alpha:
            #Accept triangle
            edges[i,j] += 1
            edges[j,i] += 1
            edges[i,k] += 1
            edges[k,i] += 1
            edges[k,j] += 1
            edges[j,k] += 1
    # Now edges is
    # 0 iff edge is exterior or not in delaunay
    # 1 iff edge is in border
    # 2 iff edge is in interior
    return np.any(np.logical_and(edges<2, edges_in_delaunay), axis=-1)#.astype(np.float)

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




