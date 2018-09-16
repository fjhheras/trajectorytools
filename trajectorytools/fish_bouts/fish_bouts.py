import numpy as np
from scipy import signal
import trajectorytools as tt


# def get_bouts_0(tr, smooth_sigma, order):
#     tr_s = tt.Trajectories.from_positions(tr.raw,
#                                           interpolate_nans=True,
#                                           smooth_sigma=smooth_sigma,
#                                           only_past=True,
#                                           body_length=None,
#                                           frame_rate=None)
#     all_starting_bouts = []
#     all_bout_peaks = []
#     for focal in range(tr.number_of_individuals):
#         print("Computing bouts for individual {:.0f}".format(focal))
#         # Find local minima an maxima
#         min_frames = signal.argrelextrema(tr_s.speed[:, focal],
#                                           np.less, order=order)[0]
#         max_frames = signal.argrelextrema(tr_s.speed[:, focal],
#                                           np.greater, order=order)[0]
#         # Compute bouts by pairs of minimums (start) and maximums peaks
#         bouts = []
#         bout = []
#         for frame in range(tr.number_of_frames):
#             if not np.isnan(tr.s[frame, focal, 0]):
#                 if frame in min_frames:
#                     # The starting of the bout is the minimum closer to the
#                     # next maximum
#                     if len(bout) == 0:
#                         bout.append(frame)
#                     elif len(bout) == 1:
#                         bout = []
#                         bout.append(frame)
#                 elif frame in max_frames:
#                     # The peak of the bout is the first maximum after the last
#                     # minimum
#                     if len(bout) == 1:
#                         bout.append(frame)
#                         bouts.append(bout)
#                         bout = []
#             else:
#                 bout = []
#         starting_bouts, bout_peaks = zip(*bouts)
#         starting_bouts = np.asarray(starting_bouts)
#         bout_peaks = np.asarray(bout_peaks)
#         all_starting_bouts.append(starting_bouts)
#         all_bout_peaks.append(bout_peaks)
#     return all_starting_bouts, all_bout_peaks, tr_s
#
#
# def correct_bouts_frames(tr, all_starting_bouts, all_bout_peaks):
#     corrected_all_starting_bouts = []
#     corrected_all_bout_peaks = []
#     for focal, (starting_bouts, bout_peaks) in enumerate(zip(all_starting_bouts, all_bout_peaks)):
#         indices_to_delete = []
#         for i, (starting_bout, bout_peak) in enumerate(zip(starting_bouts, bout_peaks)):
#             next_starting_bout = starting_bout + 1
#             while (next_starting_bout != tr.number_of_frames) and\
#                 (tr.speed[next_starting_bout, focal] < tr.speed[starting_bouts[i],focal]):
#                 starting_bouts[i] = next_starting_bout
#                 next_starting_bout += 1
#             previous_bout_peak = bout_peak - 1
#             while tr.speed[previous_bout_peak,focal] > tr.speed[bout_peaks[i],focal]:
#                 bout_peaks[i] = previous_bout_peak
#                 previous_bout_peak -= 1
#             # Correct frames for which the starting frame is bigger than the peak frame
#             if starting_bouts[i] >= bout_peaks[i]:
#                 starting_bouts[i] = starting_bout
#                 bout_peaks[i] = bout_peak
#             if (tr.speed[bout_peaks[i], focal] - tr.speed[starting_bouts[i], focal])< 0:
#                 indices_to_delete.append(i)
#         print(indices_to_delete)
#         # Delete bad bouts
#         starting_bouts = np.delete(starting_bouts, indices_to_delete)
#         bout_peaks = np.delete(bout_peaks, indices_to_delete)
#         corrected_all_starting_bouts.append(starting_bouts)
#         corrected_all_bout_peaks.append(bout_peaks)
#
#     return corrected_all_starting_bouts, corrected_all_bout_peaks


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

    test_trajectories_file = os.path.join(dir_of_data, 'test_trajectories.npy')
    positions = np.load(test_trajectories_file, encoding='latin1')
    tr = tt.Trajectories.from_positions(positions,
                                        smooth_sigma=.5,
                                        interpolate_nans=True)

    ## Method using argrelextrema and np.less and np.greater and then cleaning
    # all_starting_bouts, all_bout_peaks, tr_s = get_bouts_0(tr,
    #                                                      smooth_sigma=2,
    #                                                      order=3)
    # all_starting_bouts, all_bout_peaks = correct_bouts_frames(tr,
    #                                                           all_starting_bouts,
    #                                                           all_bout_peaks)
    ## Method using find_peaks
    all_starting_bouts, all_bout_peaks = get_bouts_1(tr,
                                                     prominence=(0.002,None),
                                                     distance=3)


    def plot_bouts(ax, starting_frame, focal):
        time_range = (starting_frame, starting_frame + 290)  # in frames
        frame_range = range(time_range[0], time_range[1], 1)  # Select only the first 60 seconds for plotting
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

    fig, ax = plt.subplots(10, figsize=(20, 20), sharex=True, sharey=True)
    for i in range(10):
        plot_bouts(ax[i], 0, i)
        if i == 9:
            ax[i].set_xlabel('frame number', fontsize=14)
            ax[i].set_ylabel('speed', fontsize=14)

    plt.subplots_adjust(hspace=1.)
    plt.show()
