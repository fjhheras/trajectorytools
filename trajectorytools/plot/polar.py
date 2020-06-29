import logging
from collections import namedtuple
from typing import Callable, Optional, Union, Tuple

import numpy as np
import scipy
import matplotlib as mpl

__all__ = ["binned_statistic_polar", "polar_histogram", "plot_polar_histogram"]


def remove_not_finite_from_args(f_unwrapped: Callable) -> Callable:
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
        finite_values = np.logical_and.reduce(
            [np.isfinite(arg) for arg in args]
        )
        if not np.all(finite_values):
            print(f"Keeping only finite values in {f_unwrapped} input")
            logging.info(f"Keeping only finite values in {f_unwrapped} input")
            args = [arg[finite_values] for arg in args]
        return f_unwrapped(*args, **kwargs)

    return wrapped


BinnedStatisticPolarResult = namedtuple(
    "BinnedStatisticPolarResult",
    ("statistic", "r_edge", "theta_edge", "binnumber"),
)


@remove_not_finite_from_args
def binned_statistic_polar(
    r: np.ndarray,
    theta: np.ndarray,
    values: np.ndarray,
    statistic: Union[Callable[[np.ndarray], float], str] = "mean",
    bins: Union[Tuple[int], int] = (7, 12),
    range_r: Union[Tuple[float], float] = 5,
) -> BinnedStatisticPolarResult:
    """Version of scipy.binned_statistic_2d for polar plots

    :param r: 1d np.array
    :param theta: 1d np.array
    :param values: The data on which the statistic will be
    computed. This must be the same shape as r and theta
    :param statistic: The statistic to compute (default is 'mean')
    Check scipy.stats.binned_statistic_2d for possible options.
    :param bins: The bin specification.
    :param range_r: The leftmost and rightmost edges for radious bins.
    Leftmost and rightmost edges of theta are always -pi and pi
    """
    if not isinstance(range_r, (list, tuple)):
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


@remove_not_finite_from_args
def polar_histogram(
    r: np.ndarray,
    theta: np.ndarray,
    bins: Union[Tuple[int], int] = (7, 12),
    range_r: Union[Tuple[float], float] = 5,
    density: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

    if not isinstance(range_r, (list, tuple)):
        range_r = (0, range_r)

    num_samples = len(r)

    H, r_edges, theta_edges = np.histogram2d(
        r, theta, bins=bins, range=[range_r, [-np.pi, np.pi]], density=False,
    )

    if density:
        # Calculating bin area
        n_sectors = len(theta_edges) - 1
        area = np.pi * (r_edges[1:] ** 2 - r_edges[:-1] ** 2) / n_sectors
        # normalising to obtain density
        H = H / num_samples / area[:, np.newaxis]

    return H, r_edges, theta_edges


## Plotting functions


def interpolate_polarmap_angles(
    values: np.ndarray,
    theta_edges: np.ndarray,
    r_edges: np.ndarray,
    factor: float = 1,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    # interpolate_polarmap_angles has a cosmetic purpose:
    # It smoothes the corners in plot_polar_histogram/pcolormesh
    # so it looks more like a circle and not a polygon
    repeated_values = np.repeat(values, factor, axis=1)
    theta_edges = np.linspace(
        -np.pi, np.pi, (theta_edges.shape[0] - 1) * factor + 1
    )
    return repeated_values, theta_edges, r_edges


def plot_polar_histogram(
    values: np.ndarray,
    r_edges: np.ndarray,
    theta_edges: np.ndarray,
    ax: mpl.axes.Axes = None,
    cmap: str = None,
    limits: Optional[Tuple[float]] = None,
    symmetric_color_limits: bool = False,
    interpolation_factor: float = 5,
    angle_convention: str = "clock",
):
    """ Plot color polar plot

    :param values: np.array
    :param r_edges: np.array
    :param theta_edges: np.array
    :param ax: matplotlib.axes.Axes. Needs to be `polar=True`
    :param cmap: matplotlib colormap
    :param limits: a tuple with the lower and upper limits of the
    color bar. If None, they are obtained from data.
    :param symmetric_color_limits: How to obtain lower and upper 
    limits of colormap from data. If False, limits are (min, max). If
    True, limits are (-max(abs), max(abs))
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
    if limits is not None:
        vmin, vmax = limits
    elif symmetric_color_limits:
        vmax = np.nanmax(np.abs(values))
        vmin = -vmax
    else:
        vmax = np.nanmax(values)
        vmin = np.nanmin(values)

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
