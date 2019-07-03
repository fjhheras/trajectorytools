from pathlib import Path
import numpy as np
import matplotlib as mpl

import trajectorytools as tt
import trajectorytools.animation as ttanimation
import trajectorytools.socialcontext as ttsocial
from trajectorytools.constants import dir_of_data

if __name__ == '__main__':
    # Loading test trajectories as a numpy array of locations
    test_trajectories_file = Path(dir_of_data) / 'test_trajectories.npy'
    test_trajectories = np.load(test_trajectories_file, allow_pickle=True)

    # We will process the numpy array, interpolate nans and smooth it.
    # To do this, we will use the Trajectories class
    smooth_params = {'sigma': 1}
    traj = tt.Trajectories.from_positions(test_trajectories,
                                          smooth_params = smooth_params,
                                          center=True)
    # We will normalise the location by the radius:
    traj.normalise_by('radius')

    # We will change the time units to seconds. The video was recorded at 32
    # fps, so we do:
    traj.new_time_unit(32, 'second')

    # Now we can find the smoothed trajectories, velocities and accelerations
    # in traj.s, traj.v and traj.a
    # We can use, for instance, the positions in traj.s and find the border of
    # the group:
    in_border = ttsocial.in_alpha_border(traj.s, alpha=5)


    # Animation showing the fish on the border
    colornorm = mpl.colors.Normalize(vmin = 0,
                                     vmax = 3,
                                     clip = True)
    mapper = mpl.cm.ScalarMappable(norm=colornorm, cmap=mpl.cm.RdBu)
    color = mapper.to_rgba(in_border)

    anim1 = ttanimation.scatter_vectors(traj.s, velocities = traj.v, k = 0.3)
    anim2 = ttanimation.scatter_ellipses_color(traj.s, traj.v, color)
    anim = anim1 + anim2

    anim.prepare()
    anim.show()




