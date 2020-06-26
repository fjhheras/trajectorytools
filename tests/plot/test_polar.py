import pytest
import numpy as np

from trajectorytools.plot import binned_statistic_polar, polar_histogram


class TestPolar:
    def setup_method(self):
        self.r = np.arange(10)
        self.theta = np.zeros(10)
        self.values = np.zeros(10) / 2
        self.args = [self.r, self.theta, self.values]
        self.args_hist = [self.r, self.theta]

    @pytest.mark.parametrize("entry", range(3))
    @pytest.mark.xfail(raises=ValueError)
    def test_binned_statistic_value_error(self, entry):
        self.args[entry] = self.args[entry][:5]
        binned_statistic_polar(*self.args)

    @pytest.mark.parametrize("entry", range(2))
    @pytest.mark.xfail(raises=ValueError)
    def test_polar_histogram_value_error(self, entry):
        self.args_hist[entry] = self.args[entry][:5]
        polar_histogram(*self.args_hist)

    def test_binned_statistic_expected(self):
        result = binned_statistic_polar(*self.args, bins=3, range_r=5)
        expected_result = np.array([[np.nan, 0, np.nan]] * 3)
        np.testing.assert_equal(result.statistic, expected_result)

    def test_polar_histogram_expected(self):
        hist, _, _ = polar_histogram(*self.args_hist, bins=3, range_r=5)
        expected_result = np.array([[0, 2, 0]] * 3)  # 0,1 | 2,3 | 4,5
        np.testing.assert_equal(hist, expected_result)

    @pytest.mark.parametrize(
        "bins_r,bins_theta", [(1, 5), (5, 2), (3, 3), (12, 1)]
    )
    def test_same_limits(self, bins_r, bins_theta):
        range_r = np.random.randint(1, 10)
        bins = (bins_r, bins_theta)
        _, r_edges, theta_edges = polar_histogram(
            *self.args_hist, bins=bins, range_r=range_r
        )
        result = binned_statistic_polar(*self.args, bins=bins, range_r=range_r)

        np.testing.assert_equal(r_edges, result.r_edge)
        np.testing.assert_equal(theta_edges, result.theta_edge)
