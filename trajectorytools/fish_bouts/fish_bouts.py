import numpy as np
from scipy import signal

def get_bouts(tr, **kwargs):
    """Obtain bouts start and peak for all individuals

    :param tr: trajectorytools Trajectories instance
    :param **kwargs: named arguments passed to scipy.signal.find_peaks
    """
    all_starting_bouts = []
    all_bout_peaks = []
    for focal in range(tr.number_of_individuals):
        #print("Computing bouts for individual {:.0f}".format(focal))
        # Find local minima and maxima
        min_frames_ = signal.find_peaks(-tr.speed[:, focal], **kwargs)[0]
        max_frames_ = signal.find_peaks(tr.speed[:, focal], **kwargs)[0]
        min_frames = [f for f in min_frames_ if not np.isnan(tr.s[f, focal, 0])]
        max_frames = [f for f in max_frames_ if not np.isnan(tr.s[f, focal, 0])]
        frames = min_frames + max_frames
        frameismax = [False]*len(min_frames) + [True]*len(max_frames)
        ordered_frames, ordered_ismax = zip(*sorted(zip(frames, frameismax)))
        bouts = [ordered_frames[i:i+2] for i in range(len(ordered_frames)-1)
                 if not ordered_ismax[i] and ordered_ismax[i+1]]
        starting_bouts, bout_peaks = zip(*bouts)
        all_starting_bouts.append(np.asarray(starting_bouts))
        all_bout_peaks.append(np.asarray(bout_peaks))
    return all_starting_bouts, all_bout_peaks
