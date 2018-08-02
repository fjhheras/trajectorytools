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
                       interpolate_nans=True, normalise=True, body_length=None,
                       smooth_sigma=0, only_past=True, dtype=np.float64):
        traj_dict = np.load(trajectories_path, encoding='latin1').item()
        # Bring here the properties that we need from the dictionary
        t = traj_dict['trajectories'].astype(dtype)
        if body_length == -1: #Force radius
            body_length = None
        else:
            body_length = traj_dict.get('body_length', body_length)
        return cls.from_positions(t, interpolate_nans=interpolate_nans,
                                  smooth_sigma=smooth_sigma,
                                  only_past=only_past,
                                  body_length=body_length)

    @classmethod
    def from_positions(cls, t, interpolate_nans=True, smooth_sigma=0,
                       only_past=True, body_length=None):
        trajectories = Namespace()
        trajectories.raw = t.copy()
        tt.normalise_trajectories(t, body_length)
        if interpolate_nans:
            tt.interpolate_nans(t)
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
        return cls(trajectories)

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
