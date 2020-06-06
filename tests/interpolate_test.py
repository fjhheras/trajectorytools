import pytest
import numpy as np

import trajectorytools.constants as cons
from trajectorytools.interpolate import (find_enclosing_circle,
                                         find_enclosing_circle_simple)
import trajectorytools as tt


class TestCircle():
    def setup_class(self):
        angles = np.random.rand(50, 10)*2*np.pi
        self.center = np.random.randn(2)
        self.R = 1 + np.random.rand(1)
        self.t = np.stack([self.R * np.cos(angles) + self.center[0],
                           self.R * np.sin(angles) + self.center[1]], axis=-1)

    @pytest.mark.parametrize("func_find_circle", [find_enclosing_circle,
                                                  find_enclosing_circle_simple])
    def test_consistency(self, func_find_circle):
        center_x, center_y, radius = func_find_circle(self.t)
        # It is possible but highly improbable that this test fails by chance
        assert pytest.approx(center_x, 1e-2) == self.center[0]
        assert pytest.approx(center_y, 1e-2) == self.center[1]
        assert pytest.approx(radius, 1e-2) == self.R

    @pytest.mark.parametrize("func_find_circle", [find_enclosing_circle,
                                                  find_enclosing_circle_simple])
    def test_consistency_some_nans(self, func_find_circle, num_nans=2):
        t = self.t.copy()
        for i in range(num_nans):
            t[np.random.randint(0, 50),
              np.random.randint(0, 10), :] = np.nan
        center_x, center_y, radius = func_find_circle(t)
        # It is possible but highly improbable that this test fails by chance
        assert pytest.approx(center_x, 1e-2) == self.center[0]
        assert pytest.approx(center_y, 1e-2) == self.center[1]
        assert pytest.approx(radius, 1e-2) == self.R

    @pytest.fixture(autouse=True)
    def run_around_tests(self):
        t = self.t
        yield
        # Check there are no side effects
        assert np.all(t == self.t)
