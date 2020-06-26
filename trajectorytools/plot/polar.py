import logging

import numpy as np
import scipy

from collections import namedtuple

__all__ = ["binned_statistic_polar", "polar_histogram", "plot_polar_histogram"]


def remove_nans_from_args(f_unwrapped):
    """ Decorator that removes nans from input numpy arrays
    """

    def wrapped(*args, **kwargs):
        for arg in args:
            if not isinstance(arg, np.ndarray):
                raise TypeError("Input needs to be np array")

            # Maybe move this to functions?
            if arg.ndim != 1 or len(arg) != len(args[0]):
                raise ValueError("Input arrays must be 1d with same length")

        # Remove nans, if any
        nan_values = np.logical_or(*[np.isnan(arg) for arg in args])
        if np.any(nan_values):
            logging.info(f"Removing nans from {f_unwrapped} input")
            args = [arg[~nan_values] for arg in args]
        return f_unwrapped(*args, **kwargs)

    return wrapped


BinnedStatisticPolarResult = namedtuple(
    "BinnedStatisticPolarResult",
    ("statistic", "r_edge", "theta_edge", "binnumber"),
)


@remove_nans_from_args
def binned_statistic_polar(
    r, theta, values, statistic="mean", bins=(7, 12), range_r=5
):
    """Version of scipy.binned_statistic_2d for polar plots

    :param r: 1d np.array 
    :param theta: 1d np.array
    :param values: np.array The data on which the statistic will be 
    computed. This must be the same shape as r and theta
    :param statistic: The statistic to compute (default is 'mean')
    Check scipy.stats.binned_statistic_2d for possible options.
    :param bins: The bin specification.
    :param range_r: The leftmost and rightmost edges for radious bins.
    Leftmost and rightmost edges of theta are always -pi and pi
    """
    if not isinstance(r, (list, tuple)):
        range_r = (0, range_r)

    results = scipy.stats.binned_statistic_2d(
        r,
        theta,
        values,
        statistic=statistic,
        bins=bins,
        range=[range_r, [-np.pi, np.pi]],
    )
    return BinnedStatisticPolarResult(
        results.statistic, results.x_edge, results.y_edge, results.binnumber
    )


@remove_nans_from_args
def polar_histogram(r, theta, range_r=5, bins=(7, 12), density=None):
    """ Version of np.histogram2d for polar plots
    :param r: 1d np.array
    :param theta: 1d np.array
    :param range_r: The leftmost and rightmost edges for radious bins.
    Leftmost and rightmost edges of theta are always -pi and pi
    :param bins: The bin specification.
    :param density: If False, the default, returns the number of 
    samples in each bin. If True, returns the probability density 
    function at the bin: `bin_count / sample_count / bin_area`
    """

    if not isinstance(r, (list, tuple)):
        range_r = (0, range_r)

    num_samples = len(r)

    H, r_edges, theta_edges = np.histogram2d(
        r, theta, bins=bins, range=[range_r, [-np.pi, np.pi]], density=False,
    )

    if density:
        # Calculating bin area
        dr = np.pi * (r_edges[1:] ** 2 - r_edges[0:-1] ** 2)
        dtheta = (theta_edges[1] - theta_edges[0]) / (2 * np.pi)
        area = np.repeat(
            dtheta * dr[:, np.newaxis], theta_edges.shape[0] - 1, 1
        )
        H = H / num_samples / area

    return H, r_edges, theta_edges


## Plotting functions


def interpolate_polarmap_angles(values, theta_edges, r_edges, factor=1):
    # interpolate_polarmap_angles has a cosmetic purpose:
    # It smoothes the corners in plot_polar_histogram/pcolormesh
    # so it looks more like a circle and not a polygon
    repeated_values = np.repeat(values, factor, axis=1)
    theta_edges = np.linspace(
        -np.pi, np.pi, (theta_edges.shape[0] - 1) * factor + 1
    )
    return repeated_values, theta_edges, r_edges


def plot_polar_histogram(
    values,
    r_edges,
    theta_edges,
    ax=None,
    cmap=None,
    symmetric_color_limits=False,
    interpolation_factor=5,
    angle_convention="clock",
):
    """ Plot color polar plot

    :param values: np.array
    :param r_edges: np.array
    :param theta_edges: np.array
    :param ax: matplotlib.axes.Axes. Needs to be `polar=True`
    :param cmap: string with matplotlib colormap
    :param symmetric_color_limits: bool
    :param interpolation_factor: None or int
    :param angle_convention: If "clock", angles increase clockwise
    and are 0 for positive y axis. If "math", angles increase
    counterclockwise and are 0 for positive x axis
    """

    if interpolation_factor is not None:
        values, theta_edges, r_edges = interpolate_polarmap_angles(
            values, theta_edges, r_edges, factor=interpolation_factor
        )
    theta, r = np.meshgrid(theta_edges, r_edges)

    # Select color limits:
    if symmetric_color_limits:
        vmax = np.max(np.abs(values))
        vmin = -vmax
    else:
        vmax = np.max(values)
        vmin = 0

    # Plot histogram/map
    fig = ax.get_figure()
    im = ax.pcolormesh(theta, r, values, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, cmap=cmap)
    cb.ax.tick_params(labelsize=24)

    if angle_convention == "clock":
        # Adjusting axis and sense of rotation to make it compatible
        # with [2]: Direction of movement is y axis, angles increase clockwise
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
