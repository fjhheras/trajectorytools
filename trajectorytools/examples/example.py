import os

import matplotlib.pyplot as plt
import numpy as np

import trajectorytools as tt
from trajectorytools.constants import dir_of_data

if __name__ == "__main__":

    # Loading a npy file and using trajectorytools normal API
    test_trajectories_file = os.path.join(dir_of_data, "test_trajectories.npy")
    t = np.load(test_trajectories_file, allow_pickle=True)
    tt.interpolate_nans(t)
    t = tt.smooth(t, sigma=0.5)
    s_, v_, a_ = tt.velocity_acceleration(t)

    n = t.shape[1]
    print("Number of fish: ", n)
    n_to_plot = 4

    v = tt.norm(v_)
    a = tt.norm(a_)

    fig, ax_hist = plt.subplots(5)
    for i in range(n_to_plot):
        ax_hist[i].hist(v[:, i])

    e_ = tt.normalise(v_)
    fig, ax = plt.subplots(2)
    for i in range(n_to_plot):
        ax[0].plot(v[:, i])
        ax[1].plot(a[:, i])

    fig, ax_trajectories = plt.subplots()
    for i in range(n_to_plot):
        ax_trajectories.plot(s_[:, i, 0], s_[:, i, 1])

    plt.show()
