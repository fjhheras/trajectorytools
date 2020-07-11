import numpy as np
import pytest

import trajectorytools as tt
import trajectorytools.constants as cons
from trajectorytools.collective import angular_momentum, polarization
from trajectorytools.geometry import center_in_trajectory


class TestExtremeCases:
    def setup_class(self):
        v = np.random.randn(50, 2)
        self.v_same = np.stack([v] * 10, axis=1)
        self.v_antiparallel = self.v_same.copy()
        self.v_antiparallel[:, ::2, :] *= -1

    def test_parallel(self):
        pol = tt.norm(polarization(self.v_same))
        assert pytest.approx(pol, 1e-16) == 1

    def test_pol_antiparallel(self):
        pol = tt.norm(polarization(self.v_antiparallel))
        assert pytest.approx(pol, 1e-16) == 0

    @pytest.mark.parametrize("num_dims_point", [2, 1])
    def test_ang_mom_antiparallel(self, num_dims_point):
        # Random point to measure angular momentum
        if num_dims_point == 2:
            point = np.random.randn(2)
        elif num_dims_point == 1:
            point = np.random.randn(50, 2)
        # Random locations (but all in the same point)
        locations = np.stack([np.random.randn(50, 2)] * 10, axis=1)
        ang_momentum = tt.norm(
            angular_momentum(self.v_antiparallel, locations, center=point)
        )
        assert pytest.approx(ang_momentum, 1e-16) == 0

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        v_same = self.v_same
        v_antiparallel = self.v_antiparallel
        yield
        # Check there are no side effects
        assert np.all(v_same == self.v_same)
        assert np.all(v_antiparallel == self.v_antiparallel)


class TestAngularMomentum:
    def setup_class(self):
        self.v = np.random.randn(50, 10, 2)
        self.s = np.random.randn(50, 10, 2)
        self.p = np.random.rand(2)

    def test_theorem_center_of_mass(self):
        # Velocity and location center of mass
        num_individuals = 10
        V = np.mean(self.v, axis=-2)
        S = np.mean(self.s, axis=-2)
        # velocity and location relative to center of mass
        v_ = self.v - V[:, np.newaxis, :]
        s_ = center_in_trajectory(self.s, S)

        # Angular momentum
        L = angular_momentum(self.v, self.s, self.p)

        L_1 = angular_momentum(v_, s_)  # center 0,0
        L_1_b = angular_momentum(v_, self.s, S)
        # Both options for L_1 the same:
        np.testing.assert_almost_equal(L_1, L_1_b)

        L_2 = num_individuals * angular_momentum(
            V[:, np.newaxis, :], S[:, np.newaxis, :], self.p
        )

        # angular momentum around a point is the sum of the angular
        # momentum around the center of mass and the angular momentum
        # of the center of mass around the point
        np.testing.assert_almost_equal(L, L_1 + L_2)
