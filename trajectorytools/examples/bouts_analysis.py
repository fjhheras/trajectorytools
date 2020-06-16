import numpy as np
from matplotlib import pyplot as plt

import trajectorytools as tt
from trajectorytools.constants import test_trajectories_path


def plot_bouts(ax, starting_frame, focal, num_frames=220):
    time_range = (starting_frame, starting_frame + num_frames)
    frame_range = range(time_range[0], time_range[1], 1)
    starting_bouts = np.squeeze(
        all_bouts[focal][
            np.where(
                (all_bouts[focal][:, 0] > frame_range[0])
                & ((all_bouts[focal][:, 0] < frame_range[-1]))
            ),
            0,
        ]
    )
    bout_peaks = np.squeeze(
        all_bouts[focal][
            np.where(
                (all_bouts[focal][:, 1] > frame_range[0])
                & ((all_bouts[focal][:, 1] < frame_range[-1]))
            ),
            1,
        ]
    )
    ax.plot(np.asarray(frame_range), tr.speed[frame_range, focal], c="b")
    for starting_bout in starting_bouts:
        ax.axvline(x=starting_bout, c="g")
    for bout_peak in bout_peaks:
        ax.axvline(x=bout_peak, c="r")


if __name__ == "__main__":

    # Loading a trajectory file produced by idtracker.ai
    tr = tt.FishTrajectories.from_idtrackerai(
        test_trajectories_path,
        smooth_params={"sigma": 0.5},
        interpolate_nans=True,
    )

    find_max_dict = {"prominence": (0.2 * tr.speed.std(), None), "distance": 3}
    find_min_dict = {
        "prominence": (0.01 * tr.speed.std(), None),
        "distance": 3,
    }
    all_bouts = tr.get_bouts(find_max_dict, find_min_dict)

    fig, ax = plt.subplots(
        tr.number_of_individuals, figsize=(20, 20), sharex=True, sharey=True
    )
    for i in range(tr.number_of_individuals):
        plot_bouts(ax[i], 0, i)
        if i == tr.number_of_individuals - 1:
            ax[i].set_xlabel("frame number", fontsize=14)
            ax[i].set_ylabel("speed", fontsize=14)

    plt.subplots_adjust(hspace=1.0)
    plt.show()
