from copy import deepcopy
from scipy import signal
import numpy as np
import trajectorytools as tt


def calculate_center_of_mass(trajectories, params):
    center_of_mass = {k: np.nanmean(trajectories[k], axis=1)
                      for k in Trajectory.keys_to_copy}
    return Trajectory(center_of_mass, params)


class Trajectory:
    keys_to_copy = ['_s', '_v', '_a']

    def __init__(self, trajectories, params):
        for key in self.keys_to_copy:
            setattr(self, key, trajectories[key])
        self.params = deepcopy(params)

    def new_length_unit(self, length_unit, length_unit_name='?'):
        factor = self.params['length_unit'] / length_unit
        self.params['center_x'] *= factor
        self.params['center_y'] *= factor
        self.params['radius'] *= factor
        self._s *= factor
        self._v *= factor
        self._a *= factor
        self.params['length_unit'] = length_unit
        self.params['length_unit_name'] = length_unit_name

    def new_time_unit(self, time_unit, time_unit_name='?'):
        factor = self.params['time_unit'] / time_unit
        self._v /= factor
        self._a /= factor**2
        self.params['time_unit'] = time_unit
        self.params['time_unit_name'] = time_unit_name

    @property
    def number_of_frames(self): return self._s.shape[0]

    @property
    def s(self): return self._s.copy()

    @property
    def v(self): return self._v.copy()

    @property
    def a(self): return self._a.copy()

    @property
    def speed(self): return tt.norm(self._v)

    @property
    def acceleration(self): return tt.norm(self._a)

    @property
    def distance_to_center(self): return tt.norm(self._s)

    @property
    def e(self): return tt.normalise(self._v)

    @property
    def tg_acceleration(self): return tt.dot(self._a, self.e)

    @property
    def curvature(self): return tt.curvature(self._v, self._a)

    @property
    def normal_acceleration(self): return np.square(self.speed)*self.curvature


class Trajectories(Trajectory):
    def __init__(self, trajectories, params):
        super().__init__(trajectories, params)
        self.center_of_mass = calculate_center_of_mass(trajectories, params)

    def __getitem__(self, val):
        view_trajectories = {k: getattr(self, k)[val]
                             for k in self.keys_to_copy}
        return self.__class__(view_trajectories, self.params)

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

        traj_dict = np.load(trajectories_path, encoding='latin1',
                            allow_pickle=True).item()
        t = traj_dict['trajectories'].astype(dtype)

        # If the trajectories contain the arena radius information, use it
        # Otherwise, return 0 for radius to be estimated from trajectories
        arena_radius = traj_dict.get('arena_radius', None)

        if normalise_by == 'body length' or normalise_by == 'body_length':
            length_unit = traj_dict['body_length']
        elif normalise_by == 'radius':
            length_unit = None
        elif normalise_by is None:
            length_unit = 1
        else:
            length_unit = float(normalise_by)
            assert length_unit > 0

        return cls.from_positions(t, interpolate_nans=interpolate_nans,
                                  smooth_sigma=smooth_sigma,
                                  only_past=only_past,
                                  length_unit=length_unit,
                                  frame_rate=traj_dict['frames_per_second'],
                                  arena_radius=arena_radius,
                                  center=center)

    @classmethod
    def from_positions(cls, t, interpolate_nans=True, smooth_sigma=0,
                       only_past=False, length_unit=1, length_unit_name='px',
                       frame_rate=None,
                       arena_radius=None, center=False):
        """Trajectory from positions

        :param t: Positions nd.array.
        :param interpolate_nans: whether to interpolate NaNs
        :param smooth_sigma: Sigma of smoothing (semi-)gaussian
        :param only_past: Smooth data using only past frames
        :param length_unit: Normalisation constant. If None, radius is used.
        :param length_unit_name: Name of normalisation constant
        :param frame_rate: Declared frame rate (currently not used)
        :param arena_radius: Declared arena radius (overrides estimation)
        :param center: Whether to offset trajectories (center to 0)
        """
        trajectories = {'raw': t.copy()}

        # Interpolate trajectories
        if interpolate_nans:
            tt.interpolate_nans(t)

        # Center and scale trajectories
        if center:
            radius, center_x, center_y, length_unit = \
              tt.center_trajectories_and_normalise(t,
                                                   length_unit=length_unit,
                                                   forced_radius=arena_radius)
            length_unit_name = '?'  # Not completely sure about this
        else:
            if length_unit is None:  # Use radius to normalise
                if arena_radius is None:  # This means, calculate radius
                    _, _, length_unit = tt.find_enclosing_circle(t)
                else:
                    length_unit = arena_radius
                radius = 1.0
                length_unit_name = 'arena radius'
            else:  # Do not bother with radius
                radius = np.nan
            center_x, center_y = 0.0, 0.0
            np.divide(t, length_unit, t)

        if smooth_sigma > 0:
            t_smooth = tt.smooth(t, sigma=smooth_sigma,
                                 only_past=only_past)
        else:
            t_smooth = t

        # Smooth trajectories
        if only_past:
            [trajectories['_s'], trajectories['_v'], trajectories['_a']] = \
                tt.velocity_acceleration_backwards(t_smooth)
        else:
            [trajectories['_s'], trajectories['_v'], trajectories['_a']] = \
                tt.velocity_acceleration(t_smooth)

        params = {"frame_rate": frame_rate,        # This is always fps
                  "center_x": center_x,            # Units: length_unit
                  "center_y": center_y,            # Units: unit length
                  "radius": radius,                # Units: unit length
                  "radius_px": radius*length_unit,  # Units: pixels
                  "length_unit": length_unit,      # Units: pixels
                  "length_unit_name": length_unit_name,
                  "time_unit": 1,  # In frames
                  "time_unit_name": 'frames'}

        traj = cls(trajectories, params)
        return traj

    def new_length_unit(self, *args, **kwargs):
        super().new_length_unit(*args, **kwargs)
        self.center_of_mass.new_length_unit(*args, **kwargs)

    def new_time_unit(self, *args, **kwargs):
        super().new_time_unit(*args, **kwargs)
        self.center_of_mass.new_time_unit(*args, **kwargs)

    @property
    def number_of_individuals(self):
        return self._s.shape[1]

    @property
    def identity_labels(self):
        # Placeholder, in case in the future labels are explicitly given
        return np.arange(self.number_of_individuals)

    @property
    def identities_array(self):
        ones = np.ones(self.raw.shape[:-1], dtype=np.int)
        return np.einsum('ij,j->ij', ones, self.identity_labels)


class FishTrajectories(Trajectories):
    def get_bouts(self, find_max_dict=None, find_min_dict=None):
        """Obtain bouts start and peak for all individuals

        :param **kwargs: named arguments passed to scipy.signal.find_peaks

        returns
        :all_bouts: list of arrays, one array per individual. The dimensions
        of every array are (number_of_bouts, 3). Col-1 is the starting frame
        of the bout, col-2 is the peak of the bout, col-3 is the beginning of
        the next bout
        """
        # TODO: Update docstring
        if find_max_dict is None: find_max_dict = {}
        if find_min_dict is None: find_min_dict = {}
        
        all_bouts = []
        for focal in range(self.number_of_individuals):
            # Find local minima and maxima
            min_frames_ = signal.find_peaks(-self.speed[:, focal], **find_min_dict)[0]
            max_frames_ = signal.find_peaks(self.speed[:, focal], **find_max_dict)[0]
            # Filter out NaNs
            min_frames = [f for f in min_frames_
                          if not np.isnan(self._s[f, focal, 0])]
            max_frames = [f for f in max_frames_
                          if not np.isnan(self._s[f, focal, 0])]
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
