import tempfile
import unittest

import numpy as np
import numpy.testing as nptest

import trajectorytools as tt
import trajectorytools.constants as cons
import trajectorytools.socialcontext as ttsocial
from trajectorytools import Trajectories, TrajectoriesWithPoints


class TrajectoriesTestCase(unittest.TestCase):
    def setUp(self):
        self.t = Trajectories.from_idtrackerai(cons.test_trajectories_path)

    def test_len(self):
        assert len(self.t) == self.t._a.shape[0]
        assert len(self.t) == self.t._v.shape[0]
        assert len(self.t) == self.t._s.shape[0]

    def test_center_of_mass(self):
        assert self.t.params is self.t.center_of_mass.params  # Same object
        nptest.assert_allclose(self.t.s.mean(axis=1), self.t.center_of_mass.s)
        nptest.assert_allclose(self.t.v.mean(axis=1), self.t.center_of_mass.v)
        nptest.assert_allclose(self.t.a.mean(axis=1), self.t.center_of_mass.a)

    def test_slice(self):
        new_t = self.t[50:100]
        assert isinstance(new_t, Trajectories)
        self.assertEqual(
            new_t.number_of_individuals, self.t.number_of_individuals
        )
        nptest.assert_equal(new_t.s, self.t.s[50:100])
        nptest.assert_equal(new_t.v, self.t.v[50:100])
        nptest.assert_equal(new_t.a, self.t.a[50:100])

    def test_restrict_individuals(self):
        n_individuals = 3
        individuals = np.random.permutation(
            np.arange(self.t.number_of_individuals)
        )[:n_individuals]
        new_t = self.t.restrict_individuals(individuals)
        nptest.assert_equal(new_t.s, self.t.s[:, individuals])
        nptest.assert_equal(new_t.v, self.t.v[:, individuals])
        nptest.assert_equal(new_t.a, self.t.a[:, individuals])

    def test_to_px(self):
        nptest.assert_allclose(self.t.point_to_px(self.t.s), self.t._s)
        nptest.assert_allclose(self.t.vector_to_px(self.t.v), self.t._v)
        nptest.assert_allclose(self.t.vector_to_px(self.t.a), self.t._a)

    def test_check_unit_change(self, new_length_unit=10, new_time_unit=5):
        length_unit = self.t.params["length_unit"]
        time_unit = self.t.params["time_unit"]
        factor_length = new_length_unit / length_unit
        factor_time = new_time_unit / time_unit

        s, v, a = self.t.s, self.t.v, self.t.a

        # We check a time unit change first
        self.t.new_time_unit(new_time_unit)
        nptest.assert_allclose(self.t.s, s)
        nptest.assert_allclose(self.t.v, v * factor_time)
        nptest.assert_allclose(self.t.a, a * factor_time ** 2)

        # We check a length unit change
        self.t.new_length_unit(new_length_unit)
        nptest.assert_allclose(self.t.s, s / factor_length)
        nptest.assert_allclose(self.t.v, v / factor_length * factor_time)
        nptest.assert_allclose(self.t.a, a / factor_length * factor_time ** 2)

    def test_straightness(self):
        straight = self.t.straightness
        assert np.all(straight <= 1)
        assert np.all(straight >= 0)
        assert straight.ndim == 1
        assert straight.shape[0] == self.t.number_of_individuals

    def test_acceleration(self):
        nptest.assert_allclose(
            self.t.acceleration,
            np.sqrt(
                self.t.normal_acceleration ** 2 + self.t.tg_acceleration ** 2
            ),
        )


class TrajectoriesTestCaseUnitChange(TrajectoriesTestCase):
    def setUp(self):
        self.t = Trajectories.from_idtrackerai(cons.test_trajectories_path)
        self.t_unchanged = Trajectories.from_idtrackerai(
            cons.test_trajectories_path
        )
        self.new_length_unit = 3
        self.t.new_length_unit(self.new_length_unit)

    def test_check_only_length_change(self):
        length_unit = self.t_unchanged.params["length_unit"]
        factor_length = self.new_length_unit / length_unit

        s, v, a = self.t_unchanged.s, self.t_unchanged.v, self.t_unchanged.a
        nptest.assert_allclose(self.t.s, s / factor_length)
        nptest.assert_allclose(self.t.v, v / factor_length)
        nptest.assert_allclose(self.t.a, a / factor_length)

    def test_check_time_change_after_length_change(self):
        length_unit = self.t_unchanged.params["length_unit"]
        time_unit = self.t_unchanged.params["time_unit"]
        new_time_unit = 3
        factor_length = self.new_length_unit / length_unit
        factor_time = new_time_unit / time_unit

        self.t.new_time_unit(new_time_unit)

        s, v, a = self.t_unchanged.s, self.t_unchanged.v, self.t_unchanged.a
        nptest.assert_allclose(self.t.s, s / factor_length)
        nptest.assert_allclose(self.t.v, v / factor_length * factor_time)
        nptest.assert_allclose(self.t.a, a / factor_length * factor_time ** 2)

    def test_estimation_enclosing_circle(self):
        center, r = self.t.estimate_center_and_radius_from_locations()
        center_px, r_px = self.t.estimate_center_and_radius_from_locations(
            in_px=True
        )
        nptest.assert_allclose(self.t.point_to_px(center), center_px)
        nptest.assert_allclose(self.t.point_from_px(center_px), center)
        nptest.assert_allclose(r_px / r, self.t.params["length_unit"])

    def test_invariance_straightness(self):
        nptest.assert_allclose(
            self.t.straightness, self.t_unchanged.straightness
        )

    def test_scaling_distance_travelled(self):
        nptest.assert_allclose(
            self.t.distance_travelled * self.new_length_unit,
            self.t_unchanged.distance_travelled,
        )


class TrajectoriesTestCaseSaveLoad(TrajectoriesTestCase):
    def setUp(self):
        t = Trajectories.from_idtrackerai(cons.test_trajectories_path)
        temporary_file = tempfile.mkstemp(
            ".npy", "trajectorytoolstest", "/tmp"
        )[1]
        t.save(temporary_file)
        self.t = Trajectories.load(temporary_file)
        self.t1 = Trajectories.from_idtrackerai(cons.test_trajectories_path)

    def test_save_and_load_equal(self):
        nptest.assert_equal(self.t1.s, self.t.s)
        nptest.assert_equal(self.t1.v, self.t.v)
        nptest.assert_equal(self.t1.a, self.t.a)
        for key in self.t.params:
            try:
                assert self.t.params[key] == self.t1.params[key]
            except ValueError:
                nptest.assert_equal(self.t.params[key], self.t1.params[key])
        for key in self.t1.params:
            try:
                assert self.t.params[key] == self.t1.params[key]
            except ValueError:
                nptest.assert_equal(self.t.params[key], self.t1.params[key])


class TrajectoriesTestCase2(TrajectoriesTestCase):
    def test_interindividual_distances(self):
        d1 = self.t.interindividual_distances
        d2 = ttsocial.adjacency_matrix(self.t.s, mode="distance")
        d3 = ttsocial.adjacency_matrix(
            self.t.s, mode="distance", use_pdist_if_all_nb=False
        )
        nptest.assert_equal(d1, d2)
        nptest.assert_equal(d2, d3)

    def test_mean_interindividual_distances(self):
        self.t.mean_interindividual_distances


class TrajectoriesTestCase3(TrajectoriesTestCase):
    def setUp(self):
        self.t1 = Trajectories.from_idtrackerai(cons.test_trajectories_path)
        self.t2 = Trajectories.from_idtrackerai(cons.test_trajectories_path)
        self.t = self.t1[50:100]

    def test_slice_and_unit_change(self, new_length_unit=10, new_time_unit=3):
        self.t.new_length_unit(new_length_unit)
        self.t.new_time_unit(new_time_unit)
        # Test that original did not change
        nptest.assert_allclose(self.t1.s, self.t2.s)

        # Test that slice and change conmute
        self.t2.new_length_unit(new_length_unit)
        self.t2.new_time_unit(new_time_unit)
        t2_sliced = self.t2[50:100]
        nptest.assert_allclose(t2_sliced.s, self.t.s)


class TrajectoriesTestCase4(TrajectoriesTestCase):
    def setUp(self):
        super().setUp()
        self.index_cut = np.random.randint(self.t.number_of_frames - 4) + 2

    def test_shape(self):
        assert self.t.distance_travelled.shape[0] == self.t.s.shape[0]
        assert self.t.distance_travelled.shape[1] == self.t.s.shape[1]

    def test_first_frame_zero(self):
        assert np.all(self.t.distance_travelled[0] == 0)

    def test_split_consistency1(self):
        shift = self.t.distance_travelled[30:] - self.t[30:].distance_travelled
        nptest.assert_allclose(shift[0], shift[1])

    def test_split_consistency2(self):
        d1 = self.t[30:].distance_travelled
        d = self.t.distance_travelled[30:]
        nptest.assert_allclose(d1[1], d[1] - d[0])

    def test_split_consistency3(self):
        f = self.t[:31].distance_travelled
        s = self.t[30:].distance_travelled
        all = self.t.distance_travelled
        nptest.assert_allclose(all[30:], s + f[-1])

    def test_split_consistency(self):
        from_first = self.t[: self.index_cut].distance_travelled
        from_second = self.t[(self.index_cut - 1) :].distance_travelled
        from_original = self.t.distance_travelled
        # Distance travelled in the first part should be the same
        nptest.assert_allclose(from_original[: self.index_cut], from_first)
        # Distance travelled in second is offset by total travelled in first
        total_travelled_in_first = from_first[-1]
        nptest.assert_allclose(
            from_original[(self.index_cut - 1) :],
            from_second + total_travelled_in_first,
        )


class TrajectoriesWithPointsTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t = TrajectoriesWithPoints.from_idtrackerai(
            cons.test_trajectories_with_points_path, center=True
        )

    def test_correct_class(self):
        assert isinstance(self.t, TrajectoriesWithPoints)


class TrajectoriesWithPointsTestCaseSaveLoad(TrajectoriesWithPointsTestCase):
    def setUp(self):
        t = TrajectoriesWithPoints.from_idtrackerai(
            cons.test_trajectories_with_points_path
        )
        temporary_file = tempfile.mkstemp(
            ".npy", "trajectorytoolstest", "/tmp"
        )[1]
        t.save(temporary_file)
        self.t = TrajectoriesWithPoints.load(temporary_file)
        self.t1 = TrajectoriesWithPoints.from_idtrackerai(
            cons.test_trajectories_with_points_path
        )

    def test_save_and_load_equal(self):
        nptest.assert_equal(self.t1.s, self.t.s)
        nptest.assert_equal(self.t1.v, self.t.v)
        nptest.assert_equal(self.t1.a, self.t.a)
        for key in self.t.points:
            try:
                assert self.t.points[key] == self.t1.points[key]
            except ValueError:
                nptest.assert_equal(self.t.points[key], self.t1.points[key])
        for key in self.t1.points:
            try:
                assert self.t.points[key] == self.t1.points[key]
            except ValueError:
                nptest.assert_equal(self.t.points[key], self.t1.points[key])


class TrajectoriesWithPointsTestCaseCenter(TrajectoriesWithPointsTestCase):
    def setUp(self):
        self.t = TrajectoriesWithPoints.from_idtrackerai(
            cons.test_trajectories_with_points_path, center=False
        )
        self.t_center = TrajectoriesWithPoints.from_idtrackerai(
            cons.test_trajectories_with_points_path, center=True
        )

    def test_recenter(self):
        self.t_center.origin_to(np.zeros(2))
        nptest.assert_allclose(self.t_center._s, self.t._s)
        for key in self.t_center.points:
            nptest.assert_allclose(
                self.t_center.points[key], self.t.points[key]
            )


class TrajectoriesWithPointsTestCaseChangeLengthUnit(
    TrajectoriesWithPointsTestCase
):
    def setUp(self):
        self.t = TrajectoriesWithPoints.from_idtrackerai(
            cons.test_trajectories_with_points_path, center=True
        )
        self.t2 = TrajectoriesWithPoints.from_idtrackerai(
            cons.test_trajectories_with_points_path, center=True
        )
        self.new_length_unit = 10
        # Scaling trajectory self.t by 10
        self.factor = self.t.new_length_unit(self.new_length_unit)

    def test_check_unit_length_change_in_points(self):
        length_unit = self.t2.params["length_unit"]
        factor_length = length_unit / self.new_length_unit
        nptest.assert_allclose(self.factor, factor_length)

        for point in self.t.points:
            nptest.assert_allclose(
                self.t.points[point], self.t2.points[point] * factor_length
            )

    def test_check_unit_length_change_in_points_twice(self):
        length_unit = self.t2.params["length_unit"]
        factor_length = (
            length_unit / self.new_length_unit
        )  # Factor of first change

        # Factor 1 in the second change
        factor2 = self.t.new_length_unit(self.new_length_unit)
        nptest.assert_allclose(factor2, 1)

        for point in self.t.points:
            nptest.assert_allclose(
                self.t.points[point], self.t2.points[point] * factor_length
            )

    def test_check_unit_length_change_in_points_and_back(self):
        factor2 = self.t.new_length_unit(1)
        nptest.assert_allclose(factor2, 1 / self.factor)

        for point in self.t.points:
            nptest.assert_allclose(self.t.points[point], self.t2.points[point])

    def test_check_unit_length_change_in_points_and_back_twice(self):
        factor2 = self.t.new_length_unit(1)
        nptest.assert_allclose(factor2, 1 / self.factor)
        factor3 = self.t.new_length_unit(1)
        nptest.assert_allclose(factor3, 1)

        for point in self.t.points:
            nptest.assert_allclose(self.t.points[point], self.t2.points[point])


class TrajectoriesWithPointsSlicedTestCaseChangeLengthUnit(
    TrajectoriesWithPointsTestCase
):
    def setUp(self):
        self.t = TrajectoriesWithPoints.from_idtrackerai(
            cons.test_trajectories_with_points_path, center=True
        )
        self.t2 = TrajectoriesWithPoints.from_idtrackerai(
            cons.test_trajectories_with_points_path, center=True
        )

    def test_check_unit_length_change_in_points(self, new_length_unit=10):
        length_unit = self.t.params["length_unit"]

        factor_length = length_unit / new_length_unit

        factor = self.t.new_length_unit(new_length_unit)
        nptest.assert_allclose(factor, factor_length)

        sliced_t = self.t[50:100]

        for point in self.t.points:
            nptest.assert_allclose(
                sliced_t.points[point], self.t2.points[point] * factor_length
            )

    def test_check_unit_length_change_in_points_twice(
        self, new_length_unit=10
    ):
        length_unit = self.t.params["length_unit"]

        factor_length = length_unit / new_length_unit

        factor = self.t.new_length_unit(new_length_unit)
        nptest.assert_allclose(factor, factor_length)

        sliced_t = self.t[50:100]
        factor = sliced_t.new_length_unit(new_length_unit)
        nptest.assert_allclose(factor, 1)

        for point in self.t.points:
            nptest.assert_allclose(
                sliced_t.points[point], self.t2.points[point] * factor_length
            )

    def test_check_unit_length_change_in_points_and_back(
        self, new_length_unit=10
    ):
        length_unit = self.t.params["length_unit"]

        factor_length = length_unit / new_length_unit

        factor = self.t.new_length_unit(new_length_unit)
        nptest.assert_allclose(factor, factor_length)

        sliced_t = self.t[50:100]
        factor = sliced_t.new_length_unit(1)
        nptest.assert_allclose(factor, 1 / factor_length)

        for point in self.t.points:
            nptest.assert_allclose(
                sliced_t.points[point], self.t2.points[point]
            )

    def test_check_unit_length_change_in_points_and_back_twice(
        self, new_length_unit=10
    ):
        length_unit = self.t.params["length_unit"]

        factor_length = length_unit / new_length_unit

        factor = self.t.new_length_unit(new_length_unit)
        nptest.assert_allclose(factor, factor_length)

        sliced_t1 = self.t[50:100]
        factor = sliced_t1.new_length_unit(1)
        nptest.assert_allclose(factor, 1 / factor_length)

        sliced_t2 = self.t[150:200]
        factor = sliced_t2.new_length_unit(1)
        nptest.assert_allclose(factor, 1 / factor_length)

        for point in self.t.points:
            nptest.assert_allclose(
                sliced_t1.points[point], self.t2.points[point]
            )
            nptest.assert_allclose(
                sliced_t2.points[point], self.t2.points[point]
            )


class RawTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        t = np.load(cons.test_raw_trajectories_path, allow_pickle=True)
        self.t = Trajectories.from_positions(t, smooth_params={"sigma": 1})


class ArenaRadiusCenterFromBorder(TrajectoriesTestCase):
    def setUp(self):
        self.t = Trajectories.from_idtrackerai(
            cons.test_trajectories_path_border
        )

    def test_arena_radius_and_center_from_border(self):
        # The width and height of the frame are 1160 and 938 pixels
        # respectively. In the trajectories dictionary there
        # is a key named 'setup_points'
        nptest.assert_allclose(
            self.t.params["_center"], (580, 469), rtol=0.1, atol=1.0
        )
        nptest.assert_allclose(
            self.t.params["radius_px"], 400, rtol=0.1, atol=1.0
        )


class CenterTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t_nocenter = Trajectories.from_idtrackerai(
            cons.test_trajectories_path
        )
        self.t = Trajectories.from_idtrackerai(
            cons.test_trajectories_path, center=True
        )

    def test_recenter(self):
        self.t.origin_to(np.zeros(2))
        nptest.assert_allclose(self.t_nocenter._s, self.t._s)

    def test_recenter2(self):
        self.t_nocenter.origin_to(self.t.params["_center"])
        nptest.assert_allclose(self.t_nocenter._s, self.t._s)


class SmoothTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t = Trajectories.from_idtrackerai(
            cons.test_trajectories_path, smooth_params={"sigma": 1}
        )


def assert_global_allclose(a, b, rel_error):
    # For things that are around 0
    abs_error = rel_error * min(a.std(), b.std())
    nptest.assert_allclose(a, b, rtol=0, atol=abs_error)


class CenterScaleTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t_nocenter = Trajectories.from_idtrackerai(
            cons.test_trajectories_path
        ).normalise_by("radius")
        self.t = Trajectories.from_idtrackerai(
            cons.test_trajectories_path, center=True
        ).normalise_by("radius")
        self.rel_error = [1e-14] * 2

    def test_recenter(self):
        self.t.origin_to(np.zeros(2))
        nptest.assert_allclose(self.t_nocenter._s, self.t._s)
        nptest.assert_allclose(self.t_nocenter._v, self.t._v)

    def test_recenter2(self):
        self.t_nocenter.origin_to(self.t.params["_center"])
        nptest.assert_allclose(self.t_nocenter._s, self.t._s)
        nptest.assert_allclose(self.t_nocenter._v, self.t._v)


class DoubleTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t = Trajectories.from_idtrackerai(
            cons.test_trajectories_path, smooth_params={"sigma": 2}
        )
        self.t2 = Trajectories.from_idtrackerai(
            cons.test_trajectories_path, smooth_params={"sigma": 2}
        )
        self.rel_error = [1e-14] * 3

    def test_close_to_original(self):
        assert_global_allclose(self.t.s, self.t2.s, self.rel_error[0])
        assert_global_allclose(self.t.v, self.t2.v, self.rel_error[1])
        assert_global_allclose(self.t.a, self.t2.a, self.rel_error[2])


class TrivialResample(DoubleTrajectoriesTestCase):
    def setUp(self):
        super().setUp()
        self.t.resample(self.t.params["frame_rate"])


class UpDownResample(DoubleTrajectoriesTestCase):
    def setUp(self):
        super().setUp()
        factor = 2.0
        frame_rate = self.t.params["frame_rate"]
        self.t.resample(frame_rate * factor)
        self.t.resample(frame_rate)
        self.rel_error = [2e-3, 5e-2, 0.3]


class DownUpResample(DoubleTrajectoriesTestCase):
    def setUp(self):
        super().setUp()
        factor = 0.5
        frame_rate = self.t.params["frame_rate"]
        self.t.resample(frame_rate * factor)
        self.t.resample(frame_rate)
        # Poor in acceleration (changes fast)
        self.rel_error = [5e-2, 0.5, 1.0]


class ScaleRadiusTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t = Trajectories.from_idtracker(
            cons.test_trajectories_path, center=True
        ).normalise_by("radius")
        self.t_normal = Trajectories.from_idtracker(
            cons.test_trajectories_path
        )

    def test_scale(self):
        corrected_s = self.t_normal.s
        corrected_s[..., 0] -= self.t_normal.params["_center"][0]
        corrected_s[..., 1] -= self.t_normal.params["_center"][1]
        corrected_s /= self.t.params["radius_px"]
        corrected_v = self.t_normal.v / self.t.params["radius_px"]
        corrected_a = self.t_normal.a / self.t.params["radius_px"]
        # nptest.assert_allclose(self.t.s, corrected_s)
        nptest.assert_allclose(self.t.v, corrected_v)
        nptest.assert_allclose(self.t.a, corrected_a, atol=1e-15)

    def test_transform_center(self):
        center_px = self.t_normal.params["_center"]
        center_transformed = self.t.point_from_px(center_px)
        nptest.assert_allclose(
            center_transformed, np.zeros_like(center_transformed)
        )

    def test_transform_back_center(self):
        center_transformed = np.zeros(2)
        center_px = self.t.point_to_px(center_transformed)
        nptest.assert_allclose(center_px, self.t_normal.params["_center"])


class TrajectoriesRadiusTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t_normal = Trajectories.from_idtracker(
            cons.test_trajectories_path, smooth_params={"sigma": 1}
        )
        self.t = Trajectories.from_idtracker(
            cons.test_trajectories_path, smooth_params={"sigma": 1}
        ).normalise_by("radius")

    def test_scaling(self):
        self.assertEqual(self.t.params["radius"], 1.0)
        nptest.assert_allclose(
            self.t.v, self.t_normal.v / self.t.params["radius_px"]
        )
        nptest.assert_allclose(
            self.t.a, self.t_normal.a / self.t.params["radius_px"], atol=1e-15
        )


if __name__ == "__main__":
    unittest.main()
