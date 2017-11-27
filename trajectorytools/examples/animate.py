import os
import numpy as np
import trajectorytools as tt 
import trajectorytools.plot as ttplot
import trajectorytools.animation as ttanimation
import trajectorytools.socialcontext as ttsocial
from trajectorytools.constants import dir_of_data

if __name__ == '__main__':
    test_trajectories_file = os.path.join(dir_of_data, 'test_trajectories.npy')
    t = np.load(test_trajectories_file)
    tt.normalise_trajectories(t)
    s_ = tt.smooth(t, interpolate = True)
    v_ = tt.smooth_velocity(t, interpolate = True)
    #a_ = tt.smooth_acceleration(t, interpolate = True)
    anim1 = ttanimation.scatter_vectors(s_, velocities = v_, k = 10)
    anim2 = ttanimation.scatter_ellipses(s_, velocities = v_)

    anim = anim1 + anim2
    anim.prepare()
    anim.show()




