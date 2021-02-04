import os

import matplotlib as mpl
import numpy as np

import trajectorytools as tt
import trajectorytools.animation as ttanimation
from trajectorytools.constants import dir_of_data

if __name__ == "__main__":
    test_trajectories_file = os.path.join(dir_of_data, "test_trajectories.npy")
    t = np.load(test_trajectories_file)
    tt.center_trajectories_and_normalise(t)
    tt.interpolate_nans(t)

    t = tt.smooth(t, sigma=0.5)
    s_, v_, a_ = tt.velocity_acceleration(t)

    speed = tt.norm(v_)
    colornorm = mpl.colors.Normalize(
        vmin=speed.min(), vmax=speed.max(), clip=True
    )
    mapper = mpl.cm.ScalarMappable(norm=colornorm, cmap=mpl.cm.hot)
    color = mapper.to_rgba(speed)
    anim1 = ttanimation.scatter_vectors(s_, velocities=v_, k=10)
    anim2 = ttanimation.scatter_ellipses_color(s_, v_, color)

    anim = anim1 + anim2
    anim.prepare()
    anim.show()
