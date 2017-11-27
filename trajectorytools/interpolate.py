import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

def interpolate_nans(t):
    """Interpolates nans linearly in a trajectory

    :param t: trajectory
    :returns: interpolated trajectory
    """
    shape_t = t.shape
    reshaped_t = t.reshape((shape_t[0],-1))
    for timeseries in range(reshaped_t.shape[-1]):
        y = reshaped_t[:,timeseries]
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

def normalise_trajectories(t):
    """Normalise trajectories in place so their values are between -1 and 1

    :param t: trajectory, to be modified in place. Last dimension is (x,y)
    :returns: multiplicative factor and shifts to recover unnormalised trajectory 
    """
    norm_const_x = (np.nanmax(t[...,0]) - np.nanmin(t[...,0]))/2
    norm_const_y = (np.nanmax(t[...,1]) - np.nanmin(t[...,1]))/2
    norm_const = max(norm_const_x, norm_const_y)
    
    center_x = (np.nanmax(t[...,0]) + np.nanmin(t[...,0]))/2
    center_y = (np.nanmax(t[...,1]) + np.nanmin(t[...,1]))/2
    #mean_x = np.nanmean(t[:,:,0])
    #mean_y = np.nanmean(t[:,:,1])
    
    t[...,0] -= center_x
    t[...,1] -= center_y
    np.divide(t, norm_const, t)
    return norm_const, center_x, center_y

def smooth_several(t, sigma = 2, truncate = 5, derivatives = [0], interpolate = False, remove_extremes = True):
    return [smooth(t, sigma=sigma, truncate = truncate, derivative = derivative, remove_extremes = remove_extremes) for derivative in derivatives]

def smooth(t, sigma = 2, truncate = 5, derivative = 0, interpolate = False, remove_extremes = True):
    if interpolate:
        interpolate_nans(t)
    smooth = gaussian_filter1d(t, sigma=sigma, axis=0, truncate = truncate, order = derivative)
    if remove_extremes:
        cut = np.int(np.ceil(sigma))
        return smooth[cut:-cut]
    else:
        return smooth

def smooth_velocity(t, **kwargs):
    kwargs['derivative'] = 1
    return smooth(t, **kwargs)

def smooth_acceleration(t, **kwargs):
    kwargs['derivative'] = 2
    return smooth(t, **kwargs)

