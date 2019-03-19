from argparse import Namespace
from scipy import signal
import numpy as np
import trajectorytools as tt


def calculate_center_of_mass(trajectories):
    center_of_mass = Namespace()
    center_of_mass_dict = vars(center_of_mass)
    trajectories_dict = vars(trajectories)
    center_of_mass_dict.update(
        {k: np.nanmean(v, axis=1) for k, v in trajectories_dict.items()})
    return center_of_mass


class Trajectories():
    def __init__(self, trajectories):
        self.trajectories = trajectories
        self.center_of_mass = calculate_center_of_mass(trajectories)
        self.__dict__.update(vars(self.trajectories))

    def __getitem__(self, val):
        view_trajectories = Namespace()
        vars(view_trajectories).update(
            {k: v[val] for k, v in vars(self.trajectories).items()})
        return Trajectories(view_trajectories)

    def view(self, start=None, end=None):
        return self[slice(start, end)]

    @classmethod
    def from_idtracker(cls, trajectories_path,
                       interpolate_nans=True, normalise_by=1, center=False,
                       smooth_sigma=0, only_past=False, dtype=np.float64):
        """Create Trajectories from a idtracker.ai trajectories file

        :param trajectories_path: idtracker.ai generated trajectories file
        :param interpolate_nans: whether to interpolate NaNs
        :param normalise_by: If a number is provided, all trajectories are
        scaled using that number. If 'radius', the program tries to obtain
        'arena_radius' from idtracker.ai trajectories. Failing that, it uses
        the smallest circle containing all trajectories. If 'body length', it
        looks for body length information in the idtrakcer.ai trajectory.
        :param center: Whether to center trajectories, using a center estimated
        from the trajectories.
        :param smooth_sigma: Sigma of smoothing (semi-)gaussian.
        :param only_past: Only smooth using data from past frames.
        :param dtype: Desired dtype of trajectories.
        """

        traj_dict = np.load(trajectories_path, encoding='latin1').item()
        t = traj_dict['trajectories'].astype(dtype)

        # If the trajectories contain the arena radius information, use it
        # Otherwise, return 0 for radius to be estimated from trajectories
        arena_radius = traj_dict.get('arena_radius', None)

        if normalise_by == 'body length' or normalise_by == 'body_length':
            unit_length = traj_dict['body_length']
        elif normalise_by == 'radius':
            unit_length = None
        elif normalise_by is None:
            unit_length = 1
        else:
            unit_length = float(normalise_by)
            assert unit_length > 0

        return cls.from_positions(t, interpolate_nans=interpolate_nans,
                                  smooth_sigma=smooth_sigma,
                                  only_past=only_past,
                                  unit_length=unit_length,
                                  frame_rate=traj_dict['frames_per_second'],
                                  arena_radius=arena_radius,
                                  center=center)

    @classmethod
    def from_positions(cls, t, interpolate_nans=True, smooth_sigma=0,
                       only_past=False, unit_length=1, frame_rate=None,
                       arena_radius=None, center=False):
        """Trajectory from positions

        :param t: Positions nd.array.
        :param interpolate_nans: whether to interpolate NaNs
        :param smooth_sigma: Sigma of smoothing (semi-)gaussian
        :param only_past: Smooth data using only past frames
        :param unit_length: Normalisation constant. If None, radius is used.
        :param frame_rate: Declared frame rate (currently not used)
        :param arena_radius: Declared arena radius (overrides radius estimation)
        :param center: Whether to offset trajectories (center to 0)
        """
        trajectories = Namespace()
        trajectories.raw = t.copy()

        # Interpolate trajectories
        if interpolate_nans:
            tt.interpolate_nans(t)

        # Center and scale trajectories
        if center:
            radius, center_x, center_y, unit_length = \
                tt.center_trajectories_and_normalise(t,
                                                     unit_length=unit_length,
                                                     forced_radius=arena_radius)
        else:
            if unit_length is None: #Use radius to normalise
                if arena_radius is None: #This means, calculate radius
                    _, _, unit_length = tt.find_enclosing_circle(t)
                else:
                    unit_length = arena_radius
                radius = 1.0
            else: # Do not bother with radius
                radius = None
            center_x, center_y = 0.0, 0.0
            np.divide(t, unit_length, t)

        if smooth_sigma > 0:
            t_smooth = tt.smooth(t, sigma=smooth_sigma,
                                 only_past=only_past)
        else:
            t_smooth = t

        # Smooth trajectories
        if only_past:
            [trajectories.s, trajectories.v, trajectories.a] = \
                tt.velocity_acceleration_backwards(t_smooth)
        else:
            [trajectories.s, trajectories.v, trajectories.a] = \
                tt.velocity_acceleration(t_smooth)

        trajectories.speed = tt.norm(trajectories.v)
        trajectories.acceleration = tt.norm(trajectories.a)
        trajectories.distance_to_center = tt.norm(trajectories.s) # Distance to the center of coordinates
        trajectories.e = tt.normalise(trajectories.v)
        trajectories.tg_acceleration = tt.dot(trajectories.a, trajectories.e)
        trajectories.curvature = tt.curvature(trajectories.v, trajectories.a)
        trajectories.normal_acceleration = \
            np.square(trajectories.speed)*trajectories.curvature
        traj = cls(trajectories)
        traj.params = {"frame_rate": frame_rate,
                       "center_x": center_x,            # Units: unit_length
                       "center_y": center_y,            # Units: unit length
                       "radius": radius,                # Units: unit length
                       "unit_length": unit_length}      # Units: pixels
        return traj


    @property
    def number_of_frames(self):
        return self.s.shape[0]

    @property
    def number_of_individuals(self):
        return self.s.shape[1]

    @property
    def identity_labels(self):
        # Placeholder, in case in the future labels are explicitly given
        return np.arange(self.number_of_individuals)

    @property
    def identities_array(self):
        ones = np.ones(self.raw.shape[:-1], dtype=np.int)
        return np.einsum('ij,j->ij', ones, self.identity_labels)


class FishTrajectories(Trajectories):
    def get_bouts(self, **kwargs):
        """Obtain bouts start and peak for all individuals

        :param **kwargs: named arguments passed to scipy.signal.find_peaks

        returns
        :all_bouts: list of arrays, one array per individual. The dimensions
        of every array are (number_of_bouts, 3). Col-1 is the starting frame
        of the bout, col-2 is the peak of the bout, col-3 is the beginning of
        the next bout
        """
        all_bouts = []
        for focal in range(self.number_of_individuals):
            # Find local minima and maxima
            min_frames_ = signal.find_peaks(-self.speed[:, focal], **kwargs)[0]
            max_frames_ = signal.find_peaks(self.speed[:, focal], **kwargs)[0]
            # Filter out NaNs
            min_frames = [f for f in min_frames_
                          if not np.isnan(self.s[f, focal, 0])]
            max_frames = [f for f in max_frames_
                          if not np.isnan(self.s[f, focal, 0])]
            # Obtain couples of consecutive minima and maxima
            frames = min_frames + max_frames
            frameismax = [False]*len(min_frames) + [True]*len(max_frames)
            ordered_frames, ordered_ismax = zip(*sorted(zip(frames, frameismax)))
            bouts = [ordered_frames[i:i+2] for i in range(len(ordered_frames)-1)
                     if not ordered_ismax[i] and ordered_ismax[i+1]]
            starting_bouts, bout_peaks = zip(*bouts)
            next_bout_start = list(starting_bouts[1:]) + [self.number_of_frames-1]
            bouts = np.asarray(list(zip(starting_bouts, bout_peaks, next_bout_start)))
            all_bouts.append(bouts)
        return all_bouts
