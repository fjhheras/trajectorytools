import scipy
import numpy as np


def interpolate_polarmap_angles(histogram, theta_edges, r_edges, factor=1):
    # interpolate_polarmap_angles has merely a cosmetic purpose:
    # It smoothes the corners in plot_polar_histogram/pcolormesh
    histogram_interpolated = np.zeros(
        (histogram.shape[0], histogram.shape[1] * factor)
    )
    for k in range(factor):
        histogram_interpolated[:, k::factor] = histogram
    theta_edges = np.linspace(
        -np.pi, np.pi, (theta_edges.shape[0] - 1) * factor + 1
    )
    Theta, R = np.meshgrid(theta_edges, r_edges)
    return histogram_interpolated, Theta, R


def binned_statistic_polar(
    x, y, values, statistic="mean", bins=(7, 12), range_r=5
):
    """ Version of scipy.binned_statistic_2d for polar plots"""
    assert x.ndim == y.ndim == 1
    assert len(x) == len(y)

    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(x, y)

    # Remove nans, if any
    # TODO: Throw a warning
    nan_values = np.logical_or(np.isnan(r), np.isnan(theta), np.isnan(values))
    value_nonans = values[~nan_values]
    r_nonans = r[~nan_values]
    theta_nonans = theta[~nan_values]

    return scipy.stats.binned_statistic_2d(
        r_nonans,
        theta_nonans,
        value_nonans,
        statistic=statistic,
        bins=bins,
        range=[[0, range_r], [-np.pi, np.pi]],
    )


def polar_histogram(x, y, range_r=5, bins=(7, 12), density=None):
    """ Version of np.histogram2d for polar plots"""
    assert x.ndim == y.ndim == 1
    assert len(x) == len(y)
    num_samples = len(x)
    r = np.sqrt(x ** 2 + y ** 2)
    theta = np.arctan2(x, y)

    H, r_edges, theta_edges = np.histogram2d(
        r,
        theta,
        bins=bins,
        range=[[0, range_r], [-np.pi, np.pi]],
        density=False,
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


def plot_polar_histogram(
    values, r_edges, theta_edges, label=None, ax=None, cmap=None, sym=False,
):

    Theta, R = np.meshgrid(theta_edges, r_edges)
    mp, Theta, R = interpolate_polarmap_angles(
        values, theta_edges, r_edges, factor=5
    )

    # Select color limits:
    if sym:
        vmax = np.max(np.abs(values))
        vmin = -vmax
    else:
        vmax = np.max(values)
        vmin = 0

    # Plot histogram/map
    fig = ax.get_figure()
    im = ax.pcolormesh(Theta, R, mp, cmap=cmap, vmin=vmin, vmax=vmax)
    cb = fig.colorbar(im, ax=ax, cmap=cmap)
    cb.ax.tick_params(labelsize=24)
    ax.set_title(label, fontsize=36)

    # Adjusting axis and sense of rotation to make it compatible with [2]:
    # Direction of movement along vertical axis, angles increase in clockwise
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
