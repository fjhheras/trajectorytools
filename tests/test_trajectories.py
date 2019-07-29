import pathlib
import unittest

import numpy as np
import numpy.testing as nptest
from trajectorytools import Trajectories
from trajectorytools.constants import dir_of_data

trajectories_path = pathlib.Path(dir_of_data) / 'test_trajectories_idtrackerai.npy'
raw_trajectories_path = pathlib.Path(dir_of_data) / 'test_trajectories.npy'

class TrajectoriesTestCase(unittest.TestCase):
    def setUp(self):
        self.t = Trajectories.from_idtracker(trajectories_path)

    def test_center_of_mass(self):
        assert(self.t.params is self.t.center_of_mass.params) #Same object
        nptest.assert_allclose(self.t.s.mean(axis=1), self.t.center_of_mass.s)
        nptest.assert_allclose(self.t.v.mean(axis=1), self.t.center_of_mass.v)
        nptest.assert_allclose(self.t.a.mean(axis=1), self.t.center_of_mass.a)

    def test_check_unit_change(self, new_length_unit=10, new_time_unit=3):
        length_unit = self.t.params['length_unit']
        time_unit = self.t.params['time_unit']

        s, v, a = self.t.s, self.t.v, self.t.a

        factor_length = new_length_unit / length_unit
        factor_time = new_time_unit / time_unit

        self.t.new_length_unit(new_length_unit)
        nptest.assert_allclose(self.t.s, s/factor_length)
        nptest.assert_allclose(self.t.v, v/factor_length)
        nptest.assert_allclose(self.t.a, a/factor_length)

        self.t.new_time_unit(new_time_unit)
        nptest.assert_allclose(self.t.s, s/factor_length)
        nptest.assert_allclose(self.t.v, v/factor_length * factor_time)
        nptest.assert_allclose(self.t.a, a/factor_length * factor_time**2)

    def test_slice(self):
        new_t = self.t[50:100]
        assert(isinstance(new_t, Trajectories))
        self.assertEqual(new_t.number_of_individuals,
                         self.t.number_of_individuals)
        nptest.assert_equal(new_t.s, self.t.s[50:100])
        nptest.assert_equal(new_t.v, self.t.v[50:100])
        nptest.assert_equal(new_t.a, self.t.a[50:100])

class RawTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        t = np.load(raw_trajectories_path, allow_pickle=True)
        self.t = Trajectories.from_positions(t)

class CenterTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t = Trajectories.from_idtracker(trajectories_path, center=True)

class SmoothTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t = Trajectories.from_idtracker(trajectories_path,
                                             smooth_params={'sigma':1})

def assert_global_allclose(a, b, rel_error):
    # For things that are around 0
    abs_error = rel_error*min(a.std(), b.std())
    nptest.assert_allclose(a, b, rtol=0, atol=abs_error)

class DoubleTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t = Trajectories.from_idtracker(trajectories_path,
                                             smooth_params={'sigma':2})
        self.t2 = Trajectories.from_idtracker(trajectories_path,
                                              smooth_params={'sigma':2})
        self.rel_error = [1e-14]*3
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
        self.t.resample(frame_rate*factor)
        self.t.resample(frame_rate)
        self.rel_error = [2e-3, 5e-2, 0.3]

class DownUpResample(DoubleTrajectoriesTestCase):
    def setUp(self):
        super().setUp()
        factor = 0.5
        frame_rate = self.t.params["frame_rate"]
        self.t.resample(frame_rate*factor)
        self.t.resample(frame_rate)
        self.rel_error = [5e-2, 0.5, 1.0] #Poor in acceleration (changes fast)

class ScaleRadiusTrajectoriesTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t = Trajectories.from_idtracker(trajectories_path, center=True
                                             ).normalise_by('radius')
        self.t_normal = Trajectories.from_idtracker(trajectories_path)

    def test_scale(self):
        corrected_s = self.t_normal.s
        corrected_s[..., 0] -= self.t_normal.params['center'][0]
        corrected_s[..., 1] -= self.t_normal.params['center'][1]
        corrected_s /= self.t.params['radius_px']
        corrected_v = self.t_normal.v / self.t.params['radius_px']
        corrected_a = self.t_normal.a / self.t.params['radius_px']
        #nptest.assert_allclose(self.t.s, corrected_s)
        nptest.assert_allclose(self.t.v, corrected_v)
        nptest.assert_allclose(self.t.a, corrected_a, atol=1e-15)

class TrajectoriesRadiusTestCase(TrajectoriesTestCase):
    def setUp(self):
        self.t_normal = Trajectories.from_idtracker(trajectories_path,
                                                    smooth_params={'sigma': 1})
        self.t = Trajectories.from_idtracker(trajectories_path,
                                             smooth_params={'sigma': 1}
                                             ).normalise_by('radius')
    def test_scaling(self):
        self.assertEqual(self.t.params['radius'], 1.0)
        nptest.assert_allclose(self.t.v,
                               self.t_normal.v / self.t.params['radius_px'])
        nptest.assert_allclose(self.t.a,
                               self.t_normal.a / self.t.params['radius_px'],
                               atol=1e-15)


if __name__ == '__main__':
    unittest.main()
