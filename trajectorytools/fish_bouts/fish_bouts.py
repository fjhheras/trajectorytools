import numpy as np
from scipy import signal


def get_bouts(tr, prominence, distance):
    all_starting_bouts = []
    all_bout_peaks = []
    for focal in range(tr.number_of_individuals):
        print("Computing bouts for individual {:.0f}".format(focal))
        # Find local minima an maxima
        min_frames = signal.find_peaks(-tr.speed[:, focal],
                                       prominence=prominence,
                                       distance=distance)[0]
        max_frames = signal.find_peaks(tr.speed[:, focal],
                                       prominence=prominence,
                                       distance=distance)[0]
        # Compute bouts by pairs of minimums (start) and maximums peaks
        bouts = []
        bout = []
        for frame in range(tr.number_of_frames):
            if not np.isnan(tr.s[frame, focal, 0]):
                if frame in min_frames:
                    # The starting of the bout is the minimum closer to the
                    # next maximum
                    if len(bout) == 0:
                        bout.append(frame)
                    elif len(bout) == 1:
                        bout = []
                        bout.append(frame)
                elif frame in max_frames:
                    # The peak of the bout is the first maximum after the last
                    # minimum
                    if len(bout) == 1:
                        bout.append(frame)
                        bouts.append(bout)
                        bout = []
            else:
                bout = []
        starting_bouts, bout_peaks = zip(*bouts)
        starting_bouts = np.asarray(starting_bouts)
        bout_peaks = np.asarray(bout_peaks)
        all_starting_bouts.append(starting_bouts)
        all_bout_peaks.append(bout_peaks)
    return all_starting_bouts, all_bout_peaks
