import numpy as np
import pytest

import trajectorytools as tt
import trajectorytools.constants as cons
from trajectorytools.geometry import (
    angle_between_vectors,
    comoving_to_fixed,
    distance_travelled,
    fixed_to_comoving,
    norm,
    normalise,
    signed_angle_between_vectors,
    straightness,
)


class TestWithStraightTrajectory:
    def setup_class(self, max_length=11):
        t = np.load(cons.test_raw_trajectories_path, allow_pickle=True)[
            :max_length
        ]
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
        np.testing.assert_almost_equal(
            distance_t[-1, 0], (self.t.shape[0] - 1) * np.sqrt(2)
        )

    def test_straightness(self):
        straight = straightness(self.t)
        assert straight.ndim == 1
        assert straight.shape[0] == self.t.shape[1]
        assert np.all(straight >= 0) and np.all(straight <= 1)
        np.testing.assert_almost_equal(straight[0], 1.0)

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        t = self.t.copy()
        yield
        # Check there are no side effects
        assert np.all(t == self.t)


class TestExtraDimensions:
    def setup_class(self):
        self.u = np.random.rand(10, 12, 2)
        self.u_2 = self.u.reshape((10, 3, 4, 2))

    @pytest.mark.parametrize(
        "func", [straightness, distance_travelled, norm, normalise]
    )
    def test_extra_dimensions(self, func):
        normal = func(self.u)
        reshaped = np.reshape(func(self.u_2), normal.shape)
        np.testing.assert_almost_equal(normal, reshaped)

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        u = self.u.copy()
        u_2 = self.u_2.copy()
        yield
        # Check there are no side effects
        assert np.all(u == self.u)
        assert np.all(u_2 == self.u_2)


class TestAngleBetweenVectors:
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


class TestRotate:
    def setup_class(self):
        self.t = np.random.rand(10, 4, 2)
        self.t2 = np.random.rand(10, 4, 2)
        self.v = np.random.rand(10, 4, 2)
        self.e = normalise(self.v)

    def test_own_v(self):
        for v in [self.v, self.t]:
            v_rot = fixed_to_comoving(v, v)
            e_y_like_v = np.zeros_like(v)
            e_y_like_v[..., 1] = 1
            np.testing.assert_almost_equal(
                v_rot, norm(v, keepdims=True) * e_y_like_v
            )

    def test_own_t(self):
        t_rot = fixed_to_comoving(self.t, self.t)
        e_y_like_t = np.zeros_like(self.t)
        e_y_like_t[..., 1] = 1
        np.testing.assert_almost_equal(
            t_rot, norm(self.t, keepdims=True) * e_y_like_t
        )

    def test_fixed_to_comoving_unnormalised(self):
        t_rot1 = fixed_to_comoving(self.t, self.v)
        t_rot2 = fixed_to_comoving(self.t, self.e)
        np.testing.assert_almost_equal(t_rot1, t_rot2)
        np.testing.assert_almost_equal(norm(t_rot1), norm(self.t))

    def test_conmute_with_normalise(self):
        t_rot1 = normalise(fixed_to_comoving(self.t, self.v))
        t_rot2 = fixed_to_comoving(normalise(self.t), self.v)
        np.testing.assert_almost_equal(t_rot1, t_rot2)

    def test_angles_unchanged(self):
        t_rot = fixed_to_comoving(self.t, self.v)
        t2_rot = fixed_to_comoving(self.t2, self.v)
        angles_before = signed_angle_between_vectors(self.t, self.t2)
        angles_after = signed_angle_between_vectors(t_rot, t2_rot)
        np.testing.assert_almost_equal(angles_before, angles_after)

    def test_angles_with_rot(self):
        t_rot = fixed_to_comoving(self.t, self.v)
        e_y = np.zeros_like(self.v)
        e_y[..., 1] = 1
        angles1 = signed_angle_between_vectors(self.t, t_rot)
        angles2 = signed_angle_between_vectors(self.v, e_y)
        np.testing.assert_almost_equal(angles1, angles2)

    def test_comoving_to_fixed(self):
        t_rot = fixed_to_comoving(self.t, self.v)
        t_back = comoving_to_fixed(t_rot, self.v)
        np.testing.assert_almost_equal(self.t, t_back)

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        v = self.v.copy()
        t = self.t.copy()
        yield
        # Check there are no side effects
        assert np.all(v == self.v)
        assert np.all(t == self.t)


class TestRotateWithNeighbours(TestRotate):
    def setup_class(self):
        self.t = np.random.rand(10, 4, 3, 2)  # With neighbours
        self.t2 = np.random.rand(10, 4, 3, 2)

    def test_angles_with_rot(self):
        t_rot = fixed_to_comoving(self.t, self.v)
        e_y = np.zeros_like(self.v)
        e_y[..., 1] = 1
        angles1 = signed_angle_between_vectors(self.t, t_rot)
        angles2 = signed_angle_between_vectors(self.v, e_y)
        angles2_b = np.broadcast_to(angles2[:, :, None], angles1.shape)
        np.testing.assert_almost_equal(angles1, angles2_b)


class TestRotateOneVectorPerFrame(TestRotate):
    def setup_class(self):
        self.v = np.random.rand(10, 2)  # One vector per frame
        self.e = normalise(self.v)

    def test_angles_with_rot(self):
        t_rot = fixed_to_comoving(self.t, self.v)
        e_y = np.zeros_like(self.v)
        e_y[..., 1] = 1
        angles1 = signed_angle_between_vectors(self.t, t_rot)
        angles2 = signed_angle_between_vectors(self.v, e_y)
        angles2_b = np.broadcast_to(angles2[:, None], angles1.shape)
        np.testing.assert_almost_equal(angles1, angles2_b)
