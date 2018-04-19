import numpy as np
import trajectorytools as tt


def average_across_individuals(s):
    """ Averages along the penultimate dimension
    """
    return np.mean(s, axis=-2)


def sum_across_individuals(s):
    """ Sums along the penultimate dimension
    """
    return np.sum(s, axis=-2)


def polarization(v):
    """ Calculates (the normalised) polarization of the vectors
    """
    norm_v = tt.norm(v, keepdims=True)
    return sum_across_individuals(v) / sum_across_individuals(norm_v)


def angular_momentum(v, s, center=np.array([0, 0])):
    if len(center.shape) == 1:  # A point
        delta_s = s - center[np.newaxis, np.newaxis, ...]
    if len(center.shape) == 2:  # A trajectory
        delta_s = s - np.expand_dims(center, 1)
    return np.sum(tt.cross2D(v, delta_s), axis=-1)
