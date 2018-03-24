import numpy as np
import trajectorytools as tt
from .socialcontext import restrict

def restrict_with_delay(data, indices, individual=None, delay=0):
    ''' This function works as ttsocial.restrict, but
    if first applies a delay in data, and cuts down the
    indices accordingly 
    '''
    delayed_data = data[delay:]
    if delay == 0:
        restricted_indices = indices
    elif delay > 0:
        restricted_indices = indices[:-delay]
    else:
        raise NotImplementedError
    return restrict(delayed_data, restricted_indices, individual=individual)

def sweep_delays(data, indices, max_delay):
    ''' This function sweeps delays of whole data
    and outputs an array with an extra dimension 
    '''
    num_restricted = indices.shape[-1]
    total_time_steps = data.shape[0] - max_delay
    num_individuals = data.shape[1]
    coordinates = data.shape[-1]
    output = np.empty([max_delay, total_time_steps, num_individuals, num_restricted, coordinates]) 
    for delay in range(max_delay):
        delayed_restricted = restrict_with_delay(data, indices, delay=delay)
        output[delay,...] = delayed_restricted[:total_time_steps]
    return output

def sweep_delayed_orientation_with_neighbours(orientation, indices, max_delay):
    # Orientation: time x num_individuals x 2
    # Indices: assumed to be exclusive of own
    # i.e. if they come from ttsocial.give_indices
    # then the first row must be removed
    total_time_steps = orientation.shape[0] - max_delay
    sweep_delay_e = sweep_delays(orientation, indices, max_delay)
    # dimensions: delay x time x num_individuals x num nb x 2
    sweep_delay_P = tt.collective.polarization(sweep_delay_e)
    # dimensions: delay x time x num_individuals x 2
    restricted_orientation = orientation[:total_time_steps]
    projected_orientation = np.einsum('...j,i...j->i...',restricted_orientation, sweep_delay_P) 
    # dimensions: delay x time x num_individuals
    return projected_orientation
    


