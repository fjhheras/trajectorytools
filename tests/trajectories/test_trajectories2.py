import numpy as np
import pytest

from trajectorytools.trajectories.trajectories import (
    _radius_and_center_from_traj_dict,
)


def circular_trajectory(radius=1, center=(0, 0)):
    length = 1000
    max_rad = 5  # Almost a complete circle
    omega = max_rad / length
    # Calculation
    offset = max_rad * np.random.rand()
    angles = np.arange(length) * omega + offset
    return np.stack(
        [
            radius * np.sin(angles) + center[0],
            radius * np.cos(angles) + center[1],
        ],
        axis=-1,
    )


def circular_trajectories(number_of_individuals, radius=1, center=(0, 0)):
    return np.stack(
        [
            circular_trajectory(radius=radius, center=center)
            for _ in range(number_of_individuals)
        ],
        axis=1,
    )


def traj_dict(border=None, arena_radius=None, arena_center=None):
    if border is not None:
        d = dict(setup_points=dict(border=border))
    else:
        d = {}
    if arena_radius is not None:
        d["arena_radius"] = arena_radius
    if arena_center is not None:
        d["arena_center"] = arena_center
    return d


param_list = [
    # If border: use it
    (
        circular_trajectories(6, radius=0.5),
        traj_dict(border=circular_trajectory()),
        dict(radius=1, center_a=(0, 0)),
    ),
    # No info: get from tajectories
    (
        circular_trajectories(6, radius=0.5),
        {},
        dict(radius=0.5, center_a=(0, 0)),
    ),
    # Partial info: use it and get the rest from traj
    (
        circular_trajectories(6, radius=0.5, center=(0.1, 0.2)),
        traj_dict(arena_radius=2, arena_center=(0.1, 0.1)),
        dict(radius=2, center_a=(0.1, 0.1)),
    ),
    (
        circular_trajectories(6, radius=0.5, center=(0.1, 0.2)),
        traj_dict(arena_radius=1),
        dict(radius=1, center_a=(0.1, 0.2)),
    ),
    (
        circular_trajectories(6, radius=0.5, center=(0.1, 0.2)),
        traj_dict(arena_center=(0.0, -0.1)),
        dict(radius=0.5, center_a=(0.0, -0.1)),
    ),
]


@pytest.mark.parametrize("locations,traj_dict,expected", param_list)
@pytest.mark.xfail
def test_radius_and_center_from_traj_dict(locations, traj_dict, expected):
    radius, center_a = _radius_and_center_from_traj_dict(locations, traj_dict)
    np.testing.assert_allclose(
        radius, expected["radius"], atol=1e-3, rtol=1e-3
    )
    np.testing.assert_allclose(
        center_a, expected["center_a"], atol=1e-3, rtol=1e-3
    )
