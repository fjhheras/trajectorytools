import numpy as np

import trajectorytools as tt

from .socialcontext import restrict


def restrict_with_delay(data, indices, individual=None, delay=0):
    """Restricts data point to neighbours with a delay

    :param data: np.array with dimensions time x individuals x coordinates
    :param indices: np.array with dimensions time x individuals x subset_of_individuals
    :param individual: label of individual to calculate restriction 
    with delay. If None, calculating for all individuals
    :param delay:

    This function works as ttsocial.restrict, but
    it first applies a delay in data, and cuts down the
    indices accordingly
    """
    delayed_data = data[delay:]
    if delay == 0:
        restricted_indices = indices
    elif delay > 0:
        restricted_indices = indices[:-delay]
    else:
        raise NotImplementedError
    return restrict(delayed_data, restricted_indices, individual=individual)


def sweep_delays(data, indices, max_delay, individual=None):
    """ This function sweeps delays of whole data
    and outputs an array with an extra dimension
    """
    num_restricted = indices.shape[-1]
    total_time_steps = data.shape[0] - max_delay
    num_individuals = data.shape[1]
    coordinates = data.shape[-1]
    if individual is None:
        output = np.empty(
            [
                max_delay,
                total_time_steps,
                num_individuals,
                num_restricted,
                coordinates,
            ],
            dtype=data.dtype,
        )
    else:
        output = np.empty(
            [max_delay, total_time_steps, num_restricted, coordinates],
            dtype=data.dtype,
        )
    for delay in range(max_delay):
        delayed_restricted = restrict_with_delay(
            data, indices, delay=delay, individual=individual
        )
        output[delay, ...] = delayed_restricted[:total_time_steps]
    return output


def sweep_delayed_orientation_with_neighbours(orientation, indices, max_delay):
    # Orientation: time x num_individuals x 2
    # Indices: assumed to be exclusive of own
    # i.e. if they come from ttsocial.neighbour_indices
    # then the first row must be removed
    total_time_steps = orientation.shape[0] - max_delay
    sweep_delay_e = sweep_delays(orientation, indices, max_delay)
    # dimensions: delay x time x num_individuals x num nb x 2
    sweep_delay_P = tt.collective.polarization(sweep_delay_e)
    # dimensions: delay x time x num_individuals x 2
    restricted_orientation = orientation[:total_time_steps]
    projected_orientation = np.einsum(
        "...j,i...j->i...", restricted_orientation, sweep_delay_P
    )
    # dimensions: delay x time x num_individuals
    return projected_orientation, sweep_delay_e


def dot_product_per_frame_with_delays(
    data, indices, sweeped_delays, frame, inplace=None
):
    """

    :param data: array of orientations 
    (total_frames, num_individuals, 2)
    :param indices: indices of neighbours 
    (total_frames, num_individuals, num_individuals-1)
    :param sweeped_delays: arrays of sweeped orientations with lag 
    (num_delays, total_frames-num_delays, num_individuals, num_individuals-1, 2)
    :param frame: frame to compute
    :param inplace: (num_delays, num_individuals, num_individual)
    :return: (num_delays, num_individuals, num_individual)
    """

    num_individuals = data.shape[1]
    max_delay = sweeped_delays.shape[0]
    if inplace is None:
        inplace = np.zeros(
            [max_delay, num_individuals, num_individuals], dtype=data.dtype
        )
    for i in range(num_individuals):
        sweeped_delays_r = sweeped_delays[:, frame, i, :]
        orientation_r = data[frame, i]
        inplace[:, i, indices[frame, i]] += np.einsum(
            "ijk,k->ij", sweeped_delays_r, orientation_r
        )
    return inplace


def dot_product_with_delays(data, indices, sweep_delayed_e, frames):
    inplace = dot_product_per_frame_with_delays(
        data, indices, sweep_delayed_e, frames[0]
    )
    for frame in frames[1:]:
        inplace = dot_product_per_frame_with_delays(
            data, indices, sweep_delayed_e, frame, inplace=inplace
        )
    return inplace / len(frames)


def sliding_average_dot_product_with_delays(
    data,
    indices,
    sweep_delayed_e,
    start_frame=0,
    end_frame=None,
    window_size=50,
):
    """

    :param data: array of orientations with shape
    (total_frames, num_individuals, 2)
    :param indices: indices of neighbours with shape
    (total_frames, num_individuals, num_individuals-1)
    :param sweep_delayed_e: arrays of sweeped orientations with lag 
    (num_delays, total_frames-num_delays, num_individuals, num_individuals-1, 2)
    :param start_frame: 0
    :param end_frame:
    :param window_size:
    :return:
    """
    assert end_frame + window_size < data.shape[0]
    frames = range(start_frame, end_frame + window_size)
    fleshout_list = [
        dot_product_per_frame_with_delays(
            data, indices, sweep_delayed_e, frame
        )
        for frame in frames
    ]
    return [
        sum(fleshout_list[i : (i + window_size)]) / window_size
        for i in range(end_frame - start_frame)
    ]


def sliding_average_dot_product_with_delays2(
    data,
    indices,
    sweep_delayed_e,
    start_frame=0,
    end_frame=None,
    num_frames_to_average=50,
):
    max_delay = sweep_delayed_e.shape[0]
    frames = range(start_frame, end_frame + num_frames_to_average)
    # The 2-by-2 xcorrelation in sparse matrix:
    # max delay x num_individuals x num_individuals
    fleshout_list = [
        dot_product_per_frame_with_delays(
            data, indices, sweep_delayed_e, frame
        )
        for frame in frames
    ]
    # A binary matrix (num_individuals x num_individuals)
    # telling us whether individuals are neighbours in a given frame
    connection_matrix_list = [
        give_connection_matrix(indices[frame]) for frame in frames
    ]
    output = []
    for i in range(end_frame - start_frame):
        sum_fleshout = sum(fleshout_list[i : (i + num_frames_to_average)])
        sum_connections = sum(
            connection_matrix_list[i : (i + num_frames_to_average)]
        )
        for t in range(max_delay):
            sum_fleshout[t][np.where(sum_connections > 0)] /= sum_connections[
                np.where(sum_connections > 0)
            ]
        output.append(sum_fleshout)
    return output


def give_connection_matrix(indices_in_frame, inplace=None):
    num_individuals = indices_in_frame.shape[0]
    if inplace is None:
        connection_matrix = np.zeros([num_individuals, num_individuals])
    else:
        connection_matrix = inplace
    for i in range(num_individuals):
        connection_matrix[i, indices_in_frame[i, :]] += 1.0
    return connection_matrix


# For debugging


def dot_product_with_delays_slow(
    data, indices, sweeped_delays, frame, inplace=None
):
    # Only for debugging
    indices.shape[-1]
    num_individuals = data.shape[1]
    max_delay = sweeped_delays.shape[0]
    if inplace is None:
        inplace = np.zeros(
            [max_delay, num_individuals, num_individuals], dtype=data.dtype
        )
    for i in range(num_individuals):
        for ij, j in enumerate(indices[frame, i]):
            sweep_delays_r = sweeped_delays[:, frame, i, ij]
            orientation = data[frame, i]
            inplace[:, i, j] += tt.dot(sweep_delays_r, orientation)
    return inplace
