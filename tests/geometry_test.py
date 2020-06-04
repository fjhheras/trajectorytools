import pytest
import numpy as np

import trajectorytools.constants as cons
from trajectorytools.geometry import straightness, distance_travelled
import trajectorytools as tt


class TestGeometry():
    def setup_class(self, max_length=11):
        t = np.load(cons.test_raw_trajectories_path,
                    allow_pickle=True)[:max_length]
        tt.interpolate_nans(t)

        # Modifying the first individual for super-straight movement
        for i in range(t.shape[0]):
            t[i, 0, :] = i
        self.t = t

    def test_distance_travelled(self):
        distance_t = distance_travelled(self.t)
        assert distance_t.ndim == 2
        assert distance_t.shape[0] == self.t.shape[0]
        assert distance_t.shape[1] == self.t.shape[1]
        np.testing.assert_almost_equal(distance_t[-1, 0],
                                       (self.t.shape[0] - 1) * np.sqrt(2))

    def test_straightness(self):
        straight = straightness(self.t)
        assert straight.ndim == 1
        assert straight.shape[0] == self.t.shape[1]
        assert np.all(straight >= 0) and np.all(straight <= 1)
        np.testing.assert_almost_equal(straight[0], 1.)
