import numpy as np
from scipy import signal
import trajectorytools as tt


def get_bouts_1(tr, prominence, distance):
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
                print('pass')
                bout = []
        starting_bouts, bout_peaks = zip(*bouts)
        starting_bouts = np.asarray(starting_bouts)
        bout_peaks = np.asarray(bout_peaks)
        all_starting_bouts.append(starting_bouts)
        all_bout_peaks.append(bout_peaks)
    return all_starting_bouts, all_bout_peaks

if __name__ == '__main__':
    import os
    from matplotlib import pyplot as plt
    plt.ion()
    from trajectorytools.constants import dir_of_data

    def plot_bouts(ax, all_starting_bouts, all_bout_peaks,
                   starting_frame, focal):
        time_range = (starting_frame, starting_frame + 290)
        frame_range = range(time_range[0], time_range[1], 1)
        starting_bouts = all_starting_bouts[focal][
            np.where((all_starting_bouts[focal] > frame_range[0]) &
                     ((all_starting_bouts[focal] < frame_range[-1])))]
        bout_peaks = all_bout_peaks[focal][
            np.where((all_bout_peaks[focal] > frame_range[0]) &
                     ((all_bout_peaks[focal] < frame_range[-1])))]
        ax.plot(np.asarray(frame_range), tr.speed[frame_range, focal], c='b')
        for starting_bout in starting_bouts:
            ax.axvline(x=starting_bout, c='g')
        for bout_peak in bout_peaks:
            ax.axvline(x=bout_peak, c='r')

    test_trajectories_file = os.path.join(dir_of_data, 'test_trajectories.npy')
    positions = np.load(test_trajectories_file, encoding='latin1')
    tr = tt.Trajectories.from_positions(positions,
                                        smooth_sigma=0,
                                        interpolate_nans=True)

    all_starting_bouts, all_bout_peaks = get_bouts_1(tr,
                                                     prominence=(0.002, None),
                                                     distance=3)

    number_of_individuals_to_show = 20
    fig, ax = plt.subplots(number_of_individuals_to_show,
                           figsize=(20, 20), sharex=True, sharey=True)
    for i in range(number_of_individuals_to_show):
        plot_bouts(ax[i], all_starting_bouts, all_bout_peaks, 0, i)
        if i == number_of_individuals_to_show - 1:
            ax[i].set_xlabel('frame number', fontsize=14)
            ax[i].set_ylabel('speed', fontsize=14)

    plt.subplots_adjust(hspace=1.)
    plt.show()
