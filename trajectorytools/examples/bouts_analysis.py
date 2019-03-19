import os
import numpy as np
from matplotlib import pyplot as plt
import trajectorytools as tt
from trajectorytools.constants import dir_of_data


def plot_bouts(ax, starting_frame, focal):
    time_range = (starting_frame, starting_frame + 290)
    frame_range = range(time_range[0], time_range[1], 1)
    starting_bouts = np.squeeze(all_bouts[focal][
        np.where((all_bouts[focal][:,0] > frame_range[0]) &
                 ((all_bouts[focal][:,0] < frame_range[-1]))),0])
    bout_peaks = np.squeeze(all_bouts[focal][
        np.where((all_bouts[focal][:,1] > frame_range[0]) &
                 ((all_bouts[focal][:,1] < frame_range[-1]))),1])
    ax.plot(np.asarray(frame_range), tr.speed[frame_range, focal], c='b')
    for starting_bout in starting_bouts:
        ax.axvline(x=starting_bout, c='g')
    for bout_peak in bout_peaks:
        ax.axvline(x=bout_peak, c='r')


if __name__ == '__main__':
    plt.ion()

    test_trajectories_file = os.path.join(dir_of_data, 'test_trajectories.npy')
    positions = np.load(test_trajectories_file, encoding='latin1')
    tr = tt.FishTrajectories.from_positions(positions,
                                            smooth_sigma=.5,
                                            interpolate_nans=True)

    all_bouts = tr.get_bouts(prominence=(0.002, None), distance=3)

    fig, ax = plt.subplots(10, figsize=(20, 20), sharex=True, sharey=True)
    for i in range(10):
        plot_bouts(ax[i], 0, i)
        if i == 9:
            ax[i].set_xlabel('frame number', fontsize=14)
            ax[i].set_ylabel('speed', fontsize=14)

    plt.subplots_adjust(hspace=1.)
    #plt.ioff()
    plt.show()
