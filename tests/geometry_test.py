import numpy as np
import pytest

import trajectorytools as tt
import trajectorytools.constants as cons
from trajectorytools.geometry import (angle_between_vectors,
                                      distance_travelled,
                                      signed_angle_between_vectors,
                                      straightness)


class TestWithStraightTrajectory():
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


class TestExtraDimensions():
    def setup_class(self):
        self.u = np.random.rand(10, 12, 2)
        self.u_2 = self.u.reshape((10, 3, 4, 2))

    @pytest.mark.parametrize("func", [straightness, distance_travelled])
    def test_extra_dimensions(self, func):
        normal = func(self.u)
        reshaped = np.reshape(func(self.u_2), normal.shape)
        print(normal, reshaped)
        np.testing.assert_almost_equal(normal, reshaped)


class TestAngleBetweenVectors():
    def setup_class(self):
        self.u = np.random.rand(10, 4, 2)
        self.v = np.random.rand(10, 4, 2)

    def test_signed_vs_normal(self):
        normal = angle_between_vectors(self.u, self.v)
        signed = signed_angle_between_vectors(self.u, self.v)
        np.testing.assert_almost_equal(normal, np.abs(signed))

    def test_symmetry(self):
        signed = signed_angle_between_vectors(self.u, self.v)
        signed_flipped = signed_angle_between_vectors(self.v, self.u)
        np.testing.assert_almost_equal(signed, -signed_flipped)
