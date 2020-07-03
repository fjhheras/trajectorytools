import numpy as np
import pytest
import random

import trajectorytools as tt
import trajectorytools.constants as cons
from trajectorytools.socialcontext import (
    in_alpha_border,
    in_convex_hull,
    neighbour_indices,
    adjacency_matrix,
    neighbour_indices_in_frame,
    adjacency_matrix_in_frame,
)


def test_convex_hull_vs_alpha_border():
    t = np.load(cons.test_raw_trajectories_path, allow_pickle=True)
    tt.interpolate_nans(t)

    convex_hull = in_convex_hull(t)
    alpha_border = in_alpha_border(t)
    in_alpha_border_not_in_convex_hull = np.logical_and(
        np.logical_not(alpha_border), convex_hull
    )
    assert not np.any(in_alpha_border_not_in_convex_hull)


@pytest.mark.parametrize("num_neighbours", [1, 3, 15])
def test_neighbour_indices_vs_adjacency_matrix(num_neighbours):
    t = np.load(cons.test_raw_trajectories_path, allow_pickle=True)
    tt.interpolate_nans(t)

    nb_indices = neighbour_indices(t, num_neighbours=num_neighbours)
    assert nb_indices.shape == tuple(
        [t.shape[0], t.shape[1], num_neighbours + 1]
    )
    adj_matrix = adjacency_matrix(t, num_neighbours=num_neighbours)
    assert adj_matrix.shape == tuple([t.shape[i] for i in [0, 1, 1]])

    # When there is an index in neighbour_indices output, the
    # corresponding elment in the adjacency_matrix must be True
    for _ in range(5):
        frame = random.randrange(0, t.shape[0])
        individual = random.randrange(0, t.shape[1])

        indices_neighbours = nb_indices[frame, individual, :]
        indices_no_neighbours = [
            i for i in range(t.shape[1]) if i not in indices_neighbours
        ]
        assert np.all(adj_matrix[frame, individual, indices_neighbours])
        assert not np.any(adj_matrix[frame, individual, indices_no_neighbours])


@pytest.mark.parametrize("num_neighbours", [1, 3, 15])
def test_neighbour_indices_vs_adjacency_matrix_in_frame(num_neighbours):
    t = np.load(cons.test_raw_trajectories_path, allow_pickle=True)
    tt.interpolate_nans(t)
    frame = t[random.randrange(0, t.shape[0])]

    nb_indices = neighbour_indices_in_frame(
        frame, num_neighbours=num_neighbours
    )
    assert nb_indices.shape == tuple([frame.shape[0], num_neighbours + 1])
    adj_matrix = adjacency_matrix_in_frame(
        frame, num_neighbours=num_neighbours
    )
    assert adj_matrix.shape == tuple([frame.shape[i] for i in [0, 0]])

    # When there is an index in neighbour_indices output, the
    # corresponding elment in the adjacency_matrix must be True
    for _ in range(5):
        individual = random.randrange(0, t.shape[1])

        indices_neighbours = nb_indices[individual, :]
        indices_no_neighbours = [
            i for i in range(t.shape[1]) if i not in indices_neighbours
        ]
        assert np.all(adj_matrix[individual, indices_neighbours])
        assert not np.any(adj_matrix[individual, indices_no_neighbours])
