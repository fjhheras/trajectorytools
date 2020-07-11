import warnings

import numpy as np

import trajectorytools as tt


def average_across_individuals(s):
    """ Averages along the penultimate dimension
    """
    warnings.warn(Warning("To be deprecated"))
    return np.mean(s, axis=-2)


def sum_across_individuals(s):
    """ Sums along the penultimate dimension
    """
    warnings.warn(Warning("To be deprecated"))
    return np.sum(s, axis=-2)


def polarization(v):
    """ Calculates (the normalised) polarization vector

    Reduction is performed on the penultimate dimension, which 
    for normal trajectories (3 dims) means individuals, so a
    single polarisation is calculated per frame.

    For trajectories with neighbours (4 dims), means neighbours,
    so a polarisation is calculated per focal.
    """
    norm_v = tt.norm(v, keepdims=True)
    return np.sum(v, axis=-2) / np.sum(norm_v, axis=-2)


def angular_momentum(v, s, center=np.array([0, 0])):
    """ Calculates angular momentum around a point

    Reduction is performed on the penultimate dimension, which 
    for normal trajectories (3 dims) means individuals, so a
    single angular momentum is calculated per frame.

    For trajectories with neighbours (4 dims), means neighbours,
    so angular momentum is calculated per focal.
    """

    if len(center.shape) == 1:  # A point
        delta_s = s - center[np.newaxis, np.newaxis, ...]
    if len(center.shape) == 2:  # A trajectory
        delta_s = s - np.expand_dims(center, 1)
    return np.sum(tt.cross2D(v, delta_s), axis=-1)
