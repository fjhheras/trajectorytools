import trajectorytools as tt
import numpy as np
import scipy.signal
from typing import Any, Dict
import warnings


def _find_interlaced_peaks(
    signal: np.ndarray, find_dict: Dict[str, Any] = None
):
    """Finds peaks and minima-between-peaks of a signal

    :param signal: array of shape (num timepoints, )
    :param find_dict: kwargs for scipy.signal.find_peaks

    returns
    :peaks: Array of shape (number_of_peaks, ) of indices of peaks in the
    signal
    :minima: Array of shape (number_of_peaks - 1, ) of minima between the
    peaks of the signal
    """
    peaks = scipy.signal.find_peaks(signal, **find_dict)[0]
    minima = []
    for start, end in zip(peaks[:-1], peaks[1:]):
        local_min = signal[start:end].argmin() + start
        minima.append(local_min)
    return np.array(peaks), np.array(minima)


def find_bouts_individual(
    speed: np.ndarray,
    find_min_dict: Dict[str, Any] = None,
    find_max_dict: Dict[str, Any] = None,
) -> np.ndarray:
    """Obtain frames of bouts from the speed of one individual

    :param speed: array of shape (num frames, )
    :param find_max_dict: kwargs for scipy.signal.find_peaks
    :param find_min_dict: kwargs for scipy.signal.find_peaks

    returns
    :all_bouts: Array of shape (number_of_bouts, 3). Col-1 is the
                starting frame of the bout, col-2 is the frame
                where the bout peaks, col-3 is the beginning of
                the next bout
    """

    number_of_frames = speed.shape[0]

    # Find local minima and maxima
    if (find_max_dict is None) and (find_min_dict is None):
        find_min_dict = {}  # default to  finding minima with scipy

    if find_max_dict is None:
        min_frames_, max_frames_ = _find_interlaced_peaks(
            signal=-speed, find_dict=find_min_dict
        )
    elif find_min_dict is None:
        max_frames_, min_frames_ = _find_interlaced_peaks(
            signal=speed, find_dict=find_max_dict
        )
    else:
        warnings.warn(
            "Using both find_min_dict and find_max_dict is deprecated, as it "
            "can miss maxima and minima of bouts, yielding erroneous results. "
            "In future, specify only one such argument.",
            DeprecationWarning,
        )
        min_frames_ = scipy.signal.find_peaks(-speed, **find_min_dict)[0]
        max_frames_ = scipy.signal.find_peaks(speed, **find_max_dict)[0]

    # Filter out NaNs
    min_frames = [f for f in min_frames_ if not np.isnan(speed[f])]
    max_frames = [f for f in max_frames_ if not np.isnan(speed[f])]

    # Obtain couples of consecutive minima and maxima
    frames = min_frames + max_frames
    frameismax = [False] * len(min_frames) + [True] * len(max_frames)
    ordered_frames, ordered_ismax = zip(*sorted(zip(frames, frameismax)))
    bouts = [
        ordered_frames[i : i + 2]
        for i in range(len(ordered_frames) - 1)
        if not ordered_ismax[i] and ordered_ismax[i + 1]
    ]

    # Ordering, and adding next_bout
    starting_bouts, bout_peaks = zip(*bouts)
    next_bout_start = list(starting_bouts[1:]) + [number_of_frames - 1]
    return np.array(list(zip(starting_bouts, bout_peaks, next_bout_start)))


def bout_statistics():
    def latency(tr, bout, focal):
        return bout[2] - bout[0]

    def acceleration_time(tr, bout, focal):
        return bout[1] - bout[0]

    def gliding_time(tr, bout, focal):
        """
        It can only be interpreted as gliding time if the end of the current
        bout coincides with the beginning of the next bout
        """
        return bout[2] - bout[1]

    def location_start(tr, bout, focal):
        return tr.s[bout[0], focal]

    def location_end(tr, bout, focal):
        return tr.s[bout[1], focal]

    def location_end_gliding(tr, bout, focal):
        return tr.s[bout[2], focal]

    def displacement(tr, bout, focal):
        return tt.norm(tr.s[bout[1], focal] - tr.s[bout[0], focal])

    def displacement_with_gliding(tr, bout, focal):
        return tt.norm(tr.s[bout[2], focal] - tr.s[bout[0], focal])

    def turning_angle(tr, bout, focal):
        return tt.signed_angle_between_vectors(
            tr.v[bout[1], focal], tr.v[bout[0], focal]
        )

    def turning_angle_with_gliding(tr, bout, focal):
        return tt.signed_angle_between_vectors(
            tr.v[bout[2], focal], tr.v[bout[0], focal]
        )

    def turning_angle_old(tr, bout, focal):
        return tt.angle_between_vectors(
            tr.v[bout[1], focal], tr.v[bout[0], focal]
        )

    return [
        value
        for key, value in locals().items()
        if callable(value) and not key.startswith("__")
    ]


def compute_bouts_parameters(tr, bouts, focal):
    variables = bout_statistics()
    bouts_dict = {
        var.__name__: [var(tr, bout, focal) for bout in bouts]
        for var in variables
    }
    bouts_dict["bouts"] = bouts
    return bouts_dict


def get_bouts_parameters(tr, find_max_dict=None, find_min_dict=None):
    all_bouts = tr.get_bouts(find_max_dict, find_min_dict)
    indiv_bouts = [
        compute_bouts_parameters(tr, all_bouts[focal], focal)
        for focal in range(tr.number_of_individuals)
    ]
    return indiv_bouts
