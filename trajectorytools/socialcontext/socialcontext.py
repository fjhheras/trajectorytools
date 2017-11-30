#import joblib #import Parallel, delayed
from sklearn.neighbors import NearestNeighbors
import numpy as np

def _neighbours_indices_in_frame(positions, num_neighbours):
    nbrs = NearestNeighbors(n_neighbors=num_neighbours+1, algorithm='ball_tree').fit(positions)
    return nbrs.kneighbors(positions, return_distance = False)
 
def give_indices(positions, num_neighbours):
    total_time_steps = positions.shape[0]
    individuals = positions.shape[1]
    next_neighbours = np.empty([total_time_steps, individuals, num_neighbours], dtype = np.int)
    for frame in range(total_time_steps):
        next_neighbours[frame,...] = _neighbours_indices_in_frame(positions[frame], num_neighbours)[:,1:] 
    return next_neighbours

def restrict(data, indices, individual):
    num_restricted = indices.shape[-1]
    total_time_steps = data.shape[0]
    coordinates = data.shape[-1]
    
    output_data = np.empty([total_time_steps, num_restricted, coordinates])
    for frame in range(total_time_steps):
        output_data[frame,...] = data[frame, indices[frame,individual],:]
    return output_data




