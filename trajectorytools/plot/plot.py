import numpy as np

def histogram(v, ax):
    valid = np.logical_not(np.isnan(v[:]))
    vv = v[valid]
    ax.hist(vv)


