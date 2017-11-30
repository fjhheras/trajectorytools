import numpy as np

def histogram(v, ax):
    valid = np.logical_not(np.isnan(v[:]))
    vv = v[valid]
    ax.hist(vv)

def position_histogram(trajectories):
    nbinsX = 100
    nbinsY = 100
    histogram2d = np.zeros((nbinsX, nbinsY))
    for i in range(trajectories.shape[1]):
        H, xedges, yedges = np.histogram2d(trajectories[:,i,0], trajectories[:,i,1], bins=(nbinsX, nbinsY))
        histogram2d += H

    average_histogram_2d = histogram2d / trajectories.shape[1]
    return average_histogram_2d


