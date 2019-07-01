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
        self.params['center'] *= factor
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
    def distance_to_center(self):
        return tt.norm(self._s - self.params['center'])

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
    def from_idtrackerai(cls, trajectories_path, **kwargs):
        return cls.from_idtracker(trajectories_path, **kwargs)

    @classmethod
    def from_idtracker(cls, trajectories_path,
                       interpolate_nans=True, center=False,
                       smooth_params=None, dtype=np.float64):
        """Create Trajectories from a idtracker.ai trajectories file

        :param trajectories_path: idtracker.ai generated trajectories file
        :param interpolate_nans: whether to interpolate NaNs
        :param center: Whether to center trajectories, using a center estimated
        from the trajectories.
        :param smooth_params: Parameters of smoothing
        :param dtype: Desired dtype of trajectories.
        """
        traj_dict = np.load(trajectories_path, encoding='latin1',
                            allow_pickle=True).item()
        t = traj_dict['trajectories'].astype(dtype)

        # If the trajectories contain the arena radius information, use it
        # Otherwise, return None for radius to be estimated from trajectories
        arena_radius = traj_dict.get('arena_radius', None)

        traj = cls.from_positions(t, interpolate_nans=interpolate_nans,
                                  smooth_params=smooth_params,
                                  arena_radius=arena_radius,
                                  center=center)

        traj.params['frame_rate'] = traj_dict.get('frames_per_second', None)
        traj.params['body_length_px'] = traj_dict.get('body_length', None)
        return traj

    @classmethod
    def from_positions(cls, t, interpolate_nans=True, smooth_params=None,
                       arena_radius=None, center=False):
        """Trajectory from positions

        :param t: Positions nd.array.
        :param interpolate_nans: whether to interpolate NaNs
        :param smooth_params: Arguments for smoothing (see tt.smooth)
        :param arena_radius: Declared arena radius (overrides estimation)
        :param center: If True, we offset trajectories (center to 0)
        """
        if smooth_params is None: smooth_params = {'sigma': -1,
                                                   'only_past': False}
        # Interpolate trajectories
        if interpolate_nans:
            tt.interpolate_nans(t)

        # Center and scale trajectories
        center_x, center_y, estimated_radius = tt.find_enclosing_circle(t)
        center_a = np.array([center_x, center_y])
        if arena_radius is None:
            radius = estimated_radius
        else:
            radius = arena_radius

        if center: #TODO: Role of center will change this
            t -= center_a
            displacement = -center_a
            center_a = np.array([0.0, 0.0])
        else:
            displacement = np.array([0.0, 0.0])

        if smooth_params['sigma'] > 0:
            t_smooth = tt.smooth(t, **smooth_params)
        else:
            t_smooth = t

        trajectories = {}
        # Smooth trajectories
        if smooth_params.get('only_past', False):
            [trajectories['_s'], trajectories['_v'], trajectories['_a']] = \
                tt.velocity_acceleration_backwards(t_smooth)
        else:
            [trajectories['_s'], trajectories['_v'], trajectories['_a']] = \
                tt.velocity_acceleration(t_smooth)

        #TODO: Discuss role of center_x / center_y. Maybe add delta_x, delta_y?
        params = {"center": center_a,              # Units: pixels
                  "displacement": displacement,    # Units: pixels
                  "radius": radius,                # Units: unit length
                  "radius_px": radius,             # Units: pixels
                  "length_unit": 1,                # Units: pixels
                  "length_unit_name": 'px',
                  "time_unit": 1,  # In frames
                  "time_unit_name": 'frames'}

        traj = cls(trajectories, params)
        return traj

    def normalise_by(self, normaliser):
        if not isinstance(normaliser, str):
            raise Exception('normalise_by needs a string. To normalise by'
                            'a number, please use new_length_unit')
        if normaliser == 'body length' or normaliser == 'body_length':
            length_unit = self.params['body_length_px']
            length_unit_name = 'BL'
        elif normaliser == 'radius':
            length_unit = self.params['radius_px']
            length_unit_name = 'R'
        else:
            raise Exception('Unknown key')
        self.new_length_unit(length_unit, length_unit_name)
        return self

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
