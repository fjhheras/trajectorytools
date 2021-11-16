import numpy as np
import numpy.testing as np_testing
from .test_trajectories2 import circular_trajectories
from trajectorytools.trajectories.concatenate import _best_ids, _concatenate_np
import pytest

rng = np.random.default_rng(0)


@pytest.mark.parametrize("num_splits", [2, 7])
@pytest.mark.parametrize("number_of_individuals", [2, 5, 9])
def test_concatenate_np(num_splits, number_of_individuals):

    # generate a dataset
    tr_np = circular_trajectories(number_of_individuals)
    # make num_splits splits, so the trajectory is splitted in num_splits virtual chunks
    # each virtual chunk or split should have the same number of individuals but only 1/num_splits of the frames
    # the frame index is on the first axis (axis=0), so we split along axis 0
    tr_split = np.array_split(tr_np, num_splits, axis=0)
    # concatenate the virtual splits, which should yield the same original dataset
    tr_np_concat = _concatenate_np(tr_split)
    # assert that!
    np_testing.assert_equal(tr_np, tr_np_concat)


@pytest.mark.parametrize("number_of_individuals", [3, 4, 5])
def test_best_ids(number_of_individuals):
    # xb is xa with a permutation of identity
    xa = rng.normal(size=(number_of_individuals, 2))
    permutation = rng.permutation(number_of_individuals)
    xb = xa[permutation]

    # _best_ids must output the inverse of the permutation used
    inverse_permutation = np.argsort(permutation)
    inferred_inverse_permutation = _best_ids(xa, xb)
    np_testing.assert_equal(inverse_permutation, inferred_inverse_permutation)
