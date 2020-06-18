import logging
import traceback
import warnings

import numpy as np
from scipy import signal
from scipy.ndimage.filters import convolve1d, gaussian_filter1d


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


def resample(x, up, down, params={}):
    # Temporary
    # I submitted this to SciPy as part of the PR#10543
    axis = 0
    n_in = x.shape[axis]

    # Substracting background
    background_line = [
        x.take(0, axis),
        (x.take(-1, axis) - x.take(0, axis)) * n_in / (n_in - 1),
    ]
    rel_len = np.linspace(0.0, 1.0, n_in, endpoint=False)
    background_in = np.stack(
        [background_line[0] + background_line[1] * l for l in rel_len],
        axis=axis,
    )
    x = x - background_in.astype(x.dtype)

    resampled_x = signal.resample_poly(x, up, down, axis=0, **params)

    # Adding background back
    n_out = resampled_x.shape[axis]
    rel_len = np.linspace(0.0, 1.0, n_out, endpoint=False)
    background_out = np.stack(
        [background_line[0] + background_line[1] * l for l in rel_len],
        axis=axis,
    )
    resampled_x += background_out.astype(x.dtype)
    return resampled_x


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
    center_x = (np.nanmax(t[..., 0]) + np.nanmin(t[..., 0])) / 2
    center_y = (np.nanmax(t[..., 1]) + np.nanmin(t[..., 1])) / 2
    r_x = (np.nanmax(t[..., 0]) - np.nanmin(t[..., 0])) / 2
    r_y = (np.nanmax(t[..., 1]) - np.nanmin(t[..., 1])) / 2
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

        if not np.isnan(t).any():
            flat_t = t.reshape((-1, 2))
        else:
            flat_t_with_nans = t.reshape((-1, 2))
            no_nans = np.logical_not(
                np.any(np.isnan(flat_t_with_nans), axis=1)
            )
            flat_t = flat_t_with_nans[np.where(no_nans)]
            logging.warning(
                "Some nans found and removed before aplying miniball"
            )
        P = [(x[0], x[1]) for x in flat_t]
        mb = miniball.Miniball(P)
        center_x, center_y = mb.center()
        radius = np.sqrt(mb.squared_radius())
    except ImportError:
        logging.warning("Miniball was not used for centre detection")
        logging.warning("Please, install https://github.com/weddige/miniball")
        center_x, center_y, radius = find_enclosing_circle_simple(t)
    except Exception:
        # logging.error(traceback.format_exc())
        # print(sys.exc_info()[0])
        traceback.print_stack()
        logging.error(
            "Miniball was not used for centre detection. Reason unknown"
        )
        center_x, center_y, radius = find_enclosing_circle_simple(t)
    return center_x, center_y, radius


def center_trajectories_and_obtain_radius(t, forced_radius=None):
    warnings.warn(Warning("To be deprecated"))
    center_x, center_y, radius = find_enclosing_circle(t)
    radius = radius if forced_radius is None else forced_radius
    t[..., 0] -= center_x
    t[..., 1] -= center_y
    return center_x, center_y, radius


def center_trajectories_and_normalise(t, unit_length=None, forced_radius=None):
    warnings.warn(Warning("To be deprecated"))
    center_x, center_y, radius = center_trajectories_and_obtain_radius(
        t, forced_radius=forced_radius
    )
    if unit_length is None:
        unit_length = radius
    np.divide(t, unit_length, t)

    return (
        radius / unit_length,
        center_x / unit_length,
        center_y / unit_length,
        unit_length,
    )


def smooth_several(t, sigma=2, truncate=5, derivatives=[0]):
    warnings.warn(Warning("To be deprecated"))
    # No longer recommended to use, particularly for small sigma
    return [
        smooth(t, sigma=sigma, truncate=truncate, derivative=derivative)
        for derivative in derivatives
    ]


def smooth(t, sigma=2, truncate=5, derivative=0, only_past=False):
    if only_past:
        assert derivative == 0  # Not implemented for more
        kernel_radius = 2  # TODO: change dynamically with input
        kernel_size = kernel_radius * 2 + 1.0
        kernel = np.exp(-np.arange(0.0, kernel_size) ** 2 / 2 / sigma ** 2)
        kernel /= kernel.sum()
        smoothed = convolve1d(t, kernel, axis=0, origin=-kernel_radius)
    else:
        smoothed = gaussian_filter1d(
            t, sigma=sigma, axis=0, truncate=truncate, order=derivative
        )
    return smoothed


def smooth_velocity(t, **kwargs):
    kwargs["derivative"] = 1
    return smooth(t, **kwargs)


def smooth_acceleration(t, **kwargs):
    kwargs["derivative"] = 2
    return smooth(t, **kwargs)


def velocity_acceleration_backwards(t, k_v_history=0.0):
    v = (1 - k_v_history) * (t[2:] - t[1:-1]) + k_v_history * (
        t[2:] - t[:-2]
    ) / 2
    a = t[2:] - 2 * t[1:-1] + t[:-2]
    return t[2:], v, a


def velocity_acceleration(t):
    v = (t[2:] - t[:-2]) / 2
    a = t[2:] - 2 * t[1:-1] + t[:-2]
    return t[1:-1], v, a
