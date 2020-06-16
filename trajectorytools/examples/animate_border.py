import matplotlib as mpl
import numpy as np

import trajectorytools as tt
import trajectorytools.animation as ttanimation
import trajectorytools.socialcontext as ttsocial
from trajectorytools.constants import test_raw_trajectories_path

if __name__ == "__main__":
    # Loading test trajectories as a numpy array of locations
    test_trajectories = np.load(test_raw_trajectories_path, allow_pickle=True)

    # We will process the numpy array, interpolate nans and smooth it.
    # To do this, we will use the Trajectories API
    smooth_params = {"sigma": 1}
    traj = tt.Trajectories.from_positions(
        test_trajectories, smooth_params=smooth_params
    )

    # We assume a circular arena and populate center and radius keys
    center, radius = traj.estimate_center_and_radius_from_locations()

    # We center trajectories around the estimated center
    traj.origin_to(center)

    # We will normalise the location by the radius:
    traj.new_length_unit(radius)

    # We will change the time units to seconds. The video was recorded at 32
    # fps, so we do:
    traj.new_time_unit(32, "second")

    # Now we can find the smoothed trajectories, velocities and accelerations
    # in traj.s, traj.v and traj.a
    # We can use, for instance, the positions in traj.s and find the border of
    # the group:
    in_border = ttsocial.in_alpha_border(traj.s, alpha=5)

    # Animation showing the fish on the border
    colornorm = mpl.colors.Normalize(vmin=0, vmax=3, clip=True)
    mapper = mpl.cm.ScalarMappable(norm=colornorm, cmap=mpl.cm.RdBu)
    color = mapper.to_rgba(in_border)

    anim1 = ttanimation.scatter_vectors(traj.s, velocities=traj.v, k=0.3)
    anim2 = ttanimation.scatter_ellipses_color(traj.s, traj.v, color)
    anim = anim1 + anim2

    anim.prepare()
    anim.show()
