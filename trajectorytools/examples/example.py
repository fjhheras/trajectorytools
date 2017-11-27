import os
import numpy as np
import matplotlib.pyplot as plt
from trajectorytools.constants import dir_of_data
import trajectorytools as tt 
import trajectorytools.plot as ttplot

if __name__ == '__main__':
    test_trajectories_file = os.path.join(dir_of_data, 'test_trajectories.npy')
    t = np.load(test_trajectories_file)
    tt.interpolate_nans(t)
    [s_, v_, a_] = tt.smooth_several(t, derivatives = [0,1,2])
    
    n = t.shape[1]
    print("Number of fish: ", n)
    
    v = tt.norm(v_) 
    a = tt.norm(a_) 
    
    fig, ax_hist = plt.subplots(5)
    for i in range(5):
        ttplot.histogram(v[:,i], ax_hist[i])
 
    e_ = tt.normalise(v_)
    fig, ax = plt.subplots(2)
    for i in range(n):
        ax[0].plot(v[:,i])
        ax[1].plot(a[:,i])
    
    fig, ax_trajectories = plt.subplots()
    for i in range(n):
        ax_trajectories.plot(s_[:,i,0], s_[:,i,1])
 
    plt.show()
    

