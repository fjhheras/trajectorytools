from trajectorytools.trajectories import import_idtrackerai_dict
from .trajectories import Trajectories
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from typing import List

# Utils


def _best_ids(xa: np.ndarray, xb: np.ndarray) -> np.ndarray:

    # Input: two arrays of locations of the same shape
    number_of_individuals = xa.shape[0]
    assert xa.shape == (number_of_individuals, 2)
    assert xb.shape == (number_of_individuals, 2)
    assert not np.isnan(xa).any()
    assert not np.isnan(xb).any()

    # We calculate the matrix of all distances between
    # all points in a and all points in b, and then find
    # the assignment that minimises the sum of distances
    distances = cdist(xa, xb)
    _, col_ind = linear_sum_assignment(distances)

    return col_ind


def _concatenate_two_np(ta: np.ndarray, tb: np.ndarray):
    # Shape of ta, tb: (individuals, frames, 2)
    best_ids = _best_ids(ta[:, -1], tb[:, 0])
    return np.concatenate([ta, tb[best_ids]], axis=1)


def _concatenate_np(t_list: List[np.ndarray]) -> np.ndarray:
    if len(t_list) == 1:
        return t_list[0]
    return _concatenate_two_np(t_list[0], _concatenate_np(t_list[1:]))


# Obtain trajectories from concatenation


def from_several_positions(t_list: List[np.ndarray], **kwargs) -> Trajectories:
    """Obtains a single trajectory object from a concatenation
    of several arrays representing locations
    """
    t_concatenated = _concatenate_np(t_list)
    return Trajectories.from_positions(t_concatenated, **kwargs)


def _concatenate_idtrackerai_dicts(traj_dicts):
    """Concatenates several idtrackerai dictionaries.

    The output contains:
    - a concatenation of the trajectories
    - the values of the first diccionary for all other keys
    """
    traj_dict_cat = traj_dicts[0].copy()
    traj_cat = _concatenate_np(
        [traj_dict["trajectories"] for traj_dict in traj_dicts]
    )
    traj_dict_cat["trajectories"] = traj_cat
    return traj_dict_cat


def from_several_idtracker_files(trajectories_paths, **kwargs):
    traj_dicts = []
    for trajectories_path in trajectories_paths:
        traj_dict = np.load(
            trajectories_path, encoding="latin1", allow_pickle=True
        ).item()
        traj_dicts.append(traj_dict)
    traj_dict = _concatenate_idtrackerai_dicts(traj_dicts)
    tr = import_idtrackerai_dict(traj_dict, **kwargs)
    tr.params["path"] = trajectories_paths
    tr.params["construct_method"] = "from_several_idtracker_files"
