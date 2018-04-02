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

def sweep_delays(data, indices, max_delay, individual=None):
    ''' This function sweeps delays of whole data
    and outputs an array with an extra dimension
    '''
    num_restricted = indices.shape[-1]
    total_time_steps = data.shape[0] - max_delay
    num_individuals = data.shape[1]
    coordinates = data.shape[-1]
    if individual is None:
        output = np.empty([max_delay, total_time_steps, num_individuals, num_restricted, coordinates])
    else:
        output = np.empty([max_delay, total_time_steps, num_restricted, coordinates])
    for delay in range(max_delay):
        delayed_restricted = restrict_with_delay(data, indices, delay=delay, individual=individual)
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
    return projected_orientation, sweep_delay_e

def fleshout_with_delay_slow_(data, indices, sweeped_delays, frame, inplace = None):
    num_restricted = indices.shape[-1]
    num_individuals = data.shape[1]
    max_delay = sweeped_delays.shape[0]
    if inplace is None:
        inplace = np.zeros([max_delay, num_individuals, num_individuals], dtype=data.dtype)
    for i in range(num_individuals):
        for ij,j in enumerate(indices[frame, i]):
            sweep_delays_r = sweeped_delays[:,frame,i,ij]
            orientation = data[frame,i]
            inplace[:,i,j] += tt.dot(sweep_delays_r, orientation)
    return inplace

def fleshout_with_delay_(data, indices, sweeped_delays, frame, inplace = None):
    num_restricted = indices.shape[-1]
    num_individuals = data.shape[1]
    max_delay = sweeped_delays.shape[0]
    if inplace is None:
        inplace = np.zeros([max_delay, num_individuals, num_individuals], dtype=data.dtype)
    for i in range(num_individuals):
        sweeped_delays_r = sweeped_delays[:,frame,i,:]
        orientation_r = data[frame,i]
        inplace[:,i,indices[frame,i]] += np.einsum('ijk,k->ij',sweeped_delays_r, orientation_r)
    return inplace

def give_connection_matrix(indices_in_frame, inplace = None):
    num_individuals = indices_in_frame.shape[0]
    if inplace is None:
        connection_matrix = np.zeros([num_individuals, num_individuals])  
    else:
        connection_matrix = inplace
    for i in range(num_individuals):
        connection_matrix[i, indices_in_frame[i,:]] += 1.0
    return connection_matrix

def fleshout_with_delay(data, indices, sweep_delayed_e, frames):
    inplace = fleshout_with_delay_(data, indices, sweep_delayed_e, frames[0])
    for frame in frames[1:]:
        inplace = fleshout_with_delay_(data, indices, sweep_delayed_e, frame, inplace=inplace)
    return inplace/len(frames)

def sliding_average_fleshout_with_delay(data, indices, sweep_delayed_e, start_frame, end_frame, num_frames_to_average = 50):
    frames = range(start_frame, end_frame+num_frames_to_average)
    print(frames)
    print(data.shape, indices.shape, sweep_delayed_e.shape)
    fleshout_list = [fleshout_with_delay_(data, indices, sweep_delayed_e, frame) for frame in frames]
    return [sum(fleshout_list[i:(i+num_frames_to_average)])/num_frames_to_average for i in range(end_frame-start_frame)]


### Here be dragons (do not look below this line)

def fleshout_with_delay_soft_window(data, indices, sweep_delayed_e, frames):
    # Assumes ordered frames
    # Super time/memory inefficient!!
    assert(len(frames)%2 == 0) #trivial to extend to the odd case if needed 
    ## Triangular weights
    weights = list(range(1,len(frames)//2+1))
    weights += weights[::-1]
    ##
    fleshout_list = [fleshout_with_delay_(data, indices, sweep_delayed_e, frame)*w for w,frame in zip(weights, frames)]
    return sum(fleshout_list)/sum(weights)


def fleshout_with_delay_average_across_nb(data, indices, sweep_delayed_e, frames):
    max_delay = sweep_delayed_e.shape[0]
    inplace = fleshout_with_delay_(data, indices, sweep_delayed_e, frames[0])
    connection_matrix = give_connection_matrix(indices[frames[0]])
    for frame in frames[1:]:
        inplace = fleshout_with_delay_(data, indices, sweep_delayed_e, frame, inplace=inplace)
        connection_matrix = give_connection_matrix(indices[frame], inplace = connection_matrix)
    print(inplace.shape, connection_matrix.shape)
    for t in range(max_delay):
        inplace[t,(connection_matrix > 0)]/=connection_matrix
    return inplace



