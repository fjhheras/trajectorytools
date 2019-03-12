from argparse import Namespace
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
                       interpolate_nans=True, normalise_by='body length',
                       smooth_sigma=0, only_past=True, dtype=np.float64):
        traj_dict = np.load(trajectories_path, encoding='latin1').item()
        t = traj_dict['trajectories'].astype(dtype)

        # If the trajectories contain the arena radius information, use it
        # Otherwise, return 0 for radius to be estimated from trajectories
        arena_radius = traj_dict.get('arena_radius', None)

        if normalise_by == 'body length':
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
                                  arena_radius=arena_radius)

    @classmethod
    def from_positions(cls, t, interpolate_nans=True, smooth_sigma=0,
                       only_past=True, unit_length=None, frame_rate=None,
                       arena_radius=None):
        trajectories = Namespace()
        trajectories.raw = t.copy()
        if interpolate_nans:
            tt.interpolate_nans(t)

        radius, center_x, center_y, unit_length = \
            tt.center_trajectories_and_normalise(t, unit_length=unit_length,
                                                 forced_radius=arena_radius)
        if smooth_sigma > 0:
            t_smooth = tt.smooth(t, sigma=smooth_sigma,
                                 only_past=only_past)
        else:
            t_smooth = t

        if only_past:
            [trajectories.s, trajectories.v, trajectories.a] = \
                tt.velocity_acceleration_backwards(t_smooth)
        else:
            [trajectories.s, trajectories.v, trajectories.a] = \
                tt.velocity_acceleration(t_smooth)

        trajectories.speed = tt.norm(trajectories.v)
        trajectories.acceleration = tt.norm(trajectories.a)
        trajectories.distance_to_center = tt.norm(trajectories.s)
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
