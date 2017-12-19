import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# TODO: Ideally, we need to separate functions calculating and plotting histograms

def get_spaced_colors(n, cmap = 'jet'):
    RGB_tuples = matplotlib.cm.get_cmap(cmap)
    return [RGB_tuples(i / n) for i in range(n)]

def subplots_row_and_colums(number_of_individuals):
    number_of_columns = int(np.sqrt(number_of_individuals))
    number_of_rows = int(np.ceil(number_of_individuals / number_of_columns))
    return number_of_rows, number_of_columns

def no_ticks_no_labels(ax):
    ax.set_yticklabels([]), ax.set_yticks([])
    ax.set_xticklabels([]), ax.set_xticks([])

def with_ordering(old_func):
    def new_func(scalar_or_vector, order_by = None, indices = None, **kwargs):
        number_of_individuals = scalar_or_vector.shape[1]
        if order_by is None:
            if indices is None:
                indices = range(number_of_individuals)
        else:
            if indices is None:
                indices = np.argsort(order_by)
            else:
                print("Warning! Giving both indices and order_by. Taking indices")
        return old_func(scalar_or_vector, indices, **kwargs)
    return new_func

@with_ordering
def plot_individual_distribution(variable, indices, nbins = 25, ticks = False):
    number_of_individuals = len(indices)
    number_of_rows, number_of_columns = subplots_row_and_colums(number_of_individuals)
    fig, ax_arr = plt.subplots(number_of_rows, number_of_columns, sharex = True, sharey = True)
    min_variable = np.min(variable)
    max_variable = np.percentile(variable, 99)
    bins = np.linspace(min_variable, max_variable, nbins)
    for i, identity in enumerate(indices):
        ax = ax_arr[int(i/number_of_columns), i%number_of_columns]
        n, bins_edges = np.histogram(variable[:,identity], bins = bins)
        ax.plot(bins_edges[:-1] + np.diff(bins_edges)[0], n)
        if ticks is False:
            no_ticks_no_labels(ax)
        sns.despine(fig, ax, top = True, left = True)
    return fig

@with_ordering
def plot_individual_distribution_of_vector(vector, indices, nbins = 10, ticks = False):
    number_of_individuals = len(indices)
    number_of_rows, number_of_columns = subplots_row_and_colums(number_of_individuals)
    fig, ax_arr = plt.subplots(number_of_rows, number_of_columns)
    v_max = 0
    min_x, max_x = np.percentile(vector[:,:,0],1), np.percentile(vector[:,:,0],99)
    min_y, max_y = np.percentile(vector[:,:,1],1), np.percentile(vector[:,:,1],99)
    print("X from {} to {}".format(min_x, max_x))
    print("Y from {} to {}".format(min_y, max_y))
    binsX = np.linspace(min_x, max_x, nbins)
    binsY = np.linspace(min_y, max_y, nbins)
    ax = []
    H = []
    for i, identity in enumerate(indices):
        ax.append(ax_arr[int(i/number_of_columns), i%number_of_columns])
        H.append(np.histogram2d(vector[:,identity,0].flatten(), vector[:,identity,1].flatten(), bins=(binsX, binsY))[0])
        v_max = max(v_max, H[i].max())
    for i, H_i in enumerate(H):
        ax[i].imshow(H_i, vmin = 0, vmax = v_max)
        if ticks is False:
            no_ticks_no_labels(ax[i])
    return fig

## Old scripts for removal below
import warnings

def histogram(v, ax):
    valid = np.logical_not(np.isnan(v[:]))
    vv = v[valid]
    ax.hist(vv)
    warnings.warn("Fraction to be removed!", DeprecationWarning)

def position_histogram(trajectories):
    warnings.warn("Fuction to be removed!", DeprecationWarning)
    nbinsX = 100
    nbinsY = 100
    histogram2d = np.zeros((nbinsX, nbinsY))
    for i in range(trajectories.shape[1]):
        H, xedges, yedges = np.histogram2d(trajectories[:,i,0], trajectories[:,i,1], bins=(nbinsX, nbinsY))
        histogram2d += H

    average_histogram_2d = histogram2d / trajectories.shape[1]
    return average_histogram_2d



