import traceback
import logging
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d, convolve1d


def interpolate_nans(t):
    """Interpolates nans linearly in a trajectory

    :param t: trajectory
    :returns: interpolated trajectory
    """
    shape_t = t.shape
    reshaped_t = t.reshape((shape_t[0], -1))
    for timeseries in range(reshaped_t.shape[-1]):
        y = reshaped_t[:, timeseries]
        nans, x = _nan_helper(y)
        y[nans] = np.interp(x(nans), x(~nans), y[~nans])

    # Ugly slow hack, as reshape seems not to return a view always
    back_t = reshaped_t.reshape(shape_t)
    t[...] = back_t


def _nan_helper(y):
    """Helper to handle indices and logical indices of NaNs.
    https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array
    Input:
        - y, 1d numpy array with possible NaNs
    Output:
        - nans, logical indices of NaNs
        - index, a function, with signature indices= index(logical_indices),
          to convert logical indices of NaNs to 'equivalent' indices
    Example:
        >>> # linear interpolation of NaNs
        >>> nans, x= nan_helper(y)
        >>> y[nans]= np.interp(x(nans), x(~nans), y[~nans])
    """

    return np.isnan(y), lambda z: z.nonzero()[0]


def find_enclosing_circle_simple(t):
    center_x = (np.nanmax(t[..., 0]) + np.nanmin(t[..., 0]))/2
    center_y = (np.nanmax(t[..., 1]) + np.nanmin(t[..., 1]))/2
    r_x = (np.nanmax(t[..., 0]) - np.nanmin(t[..., 0]))/2
    r_y = (np.nanmax(t[..., 1]) - np.nanmin(t[..., 1]))/2
    radius = max(r_x, r_y)
    return center_x, center_y, radius


def find_enclosing_circle(t):
    """Find center of trajectories

    It first tries to use external library miniball. If it fails, it resorts
    to a simpler algorithm.

    :param t: trajectory
    """
    try:
        import miniball
        assert not np.isnan(t).any()
        flat_t = t.reshape((-1, 2))
        P = [(x[0], x[1]) for x in flat_t]
        mb = miniball.Miniball(P)
        center_x, center_y = mb.center()
        radius = np.sqrt(mb.squared_radius())
    except ImportError:
        logging.warning("Miniball was not used for centre detection")
        logging.warning("Please, install https://github.com/weddige/miniball")
        center_x, center_y, radius = find_enclosing_circle_simple(t)
    except Exception:
        logging.error(traceback.format_exc())
        logging.error("Miniball was not used for centre detection")
        center_x, center_y, radius = find_enclosing_circle_simple(t)
    return center_x, center_y, radius


def normalise_trajectories(t, body_length=None):
    """Normalise trajectories in place so their values are between -1 and 1

    :param t: trajectory, to be modified in place. Last dimension is (x,y)
    :returns: scale and shift to recover unnormalised trajectory
    """
    center_x, center_y, radius = find_enclosing_circle(t)
    t[..., 0] -= center_x
    t[..., 1] -= center_y
    if body_length is not None:
        np.divide(t, body_length, t)
        return radius/body_length, center_x/body_length, center_y/body_length, body_length
    else:
        np.divide(t, radius, t)
        return 1, center_x/radius, center_y/radius, radius


def smooth_several(t, sigma=2, truncate=5, derivatives=[0]):
    return [smooth(t, sigma=sigma, truncate=truncate,
                   derivative=derivative) for derivative in derivatives]


def smooth(t, sigma=2, truncate=5, derivative=0, only_past=False):
    if only_past:
        assert derivative == 0 # Not implemented for more
        kernel_radius = 2 #TODO: change dynamically with input
        kernel_size = kernel_radius*2 + 1.0
        kernel = np.exp(-np.arange(0.0,kernel_size)**2/2/sigma**2)
        kernel /= kernel.sum()
        smoothed = convolve1d(t, kernel, axis=0, origin=-kernel_radius)
    else:
        smoothed = gaussian_filter1d(t, sigma=sigma, axis=0,
                                 truncate=truncate, order=derivative)
    return smoothed


def smooth_velocity(t, **kwargs):
    kwargs['derivative'] = 1
    return smooth(t, **kwargs)


def smooth_acceleration(t, **kwargs):
    kwargs['derivative'] = 2
    return smooth(t, **kwargs)


def velocity_acceleration_backwards(t, k_v_history=0.0):
    v = (1-k_v_history)*(t[2:] - t[1:-1]) + k_v_history*(t[1:-1] - t[:-2])
    a = t[2:] - 2*t[1:-1] + t[:-2]
    return t[2:], v, a


def velocity_acceleration(t):
    v = (t[2:] - t[:-2])/2
    a = t[2:] - 2*t[1:-1] + t[:-2]
    return t[1:-1], v, a
