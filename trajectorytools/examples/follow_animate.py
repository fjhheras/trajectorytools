import os

import numpy as np

import trajectorytools as tt
import trajectorytools.animation as ttanimation
import trajectorytools.socialcontext as ttsocial
from trajectorytools.constants import dir_of_data


def animate_and_follow(individual=0, num_neighbours=15):
    # Using the low level API
    test_trajectories_file = os.path.join(dir_of_data, "test_trajectories.npy")
    t = np.load(test_trajectories_file, allow_pickle=True)
    tt.interpolate_nans(t)
    tt.center_trajectories_and_normalise(t)
    s_ = tt.smooth(t)
    v_ = tt.smooth_velocity(t)

    e_ = tt.normalise(v_[:, individual, :])
    s = tt.center_in_individual(s_, individual)
    s = tt.fixed_to_comoving(s, e_)
    v = tt.fixed_to_comoving(v_, e_)
    center = tt.fixed_to_comoving(-s_[:, [individual], :], e_)

    indices = ttsocial.neighbour_indices(s, num_neighbours)
    sn = ttsocial.restrict(s, indices, individual)
    vn = ttsocial.restrict(v, indices, individual)

    sf = s[:, [individual], :]
    vf = v[:, [individual], :]

    anim = ttanimation.scatter_vectors(s, velocities=v, k=10)
    anim += ttanimation.scatter_ellipses(s, velocities=v, color="c")
    anim += ttanimation.scatter_ellipses(sn, velocities=vn, color="b")
    anim += ttanimation.scatter_ellipses(sf, velocities=vf, color="r")
    anim += ttanimation.scatter_circle(center)
    anim.prepare()
    anim.show()


if __name__ == "__main__":
    animate_and_follow()
