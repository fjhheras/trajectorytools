import numpy as np
import pytest

import trajectorytools as tt
import trajectorytools.constants as cons
from trajectorytools.socialcontext import in_alpha_border, in_convex_hull


def test_convex_hull_vs_alpha_border():
    t = np.load(cons.test_raw_trajectories_path, allow_pickle=True)
    tt.interpolate_nans(t)

    convex_hull = in_convex_hull(t)
    alpha_border = in_alpha_border(t)
    in_alpha_border_not_in_convex_hull = np.logical_and(
        np.logical_not(alpha_border), convex_hull
    )
    assert not np.any(in_alpha_border_not_in_convex_hull)
