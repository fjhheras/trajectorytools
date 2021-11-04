from trajectorytools.trajectories.trajectories import Trajectories
from trajectories import estimate_center_and_radius, Trajectories
import numpy as np


def _radius_and_center_from_traj_dict(locations, traj_dict):
    """Obtains radius and center of the arena

    If the trajectories contain the arena radius/center information, use it
    Otherwise, return None to estimate radius/center from trajectories

    :param locations: Numpy array of locations. Last dim must be (x, y)
    :param traj_dict:
    """
    if "setup_points" in traj_dict and "border" in traj_dict["setup_points"]:
        center_a, radius = estimate_center_and_radius(
            traj_dict["setup_points"]["border"]
        )
    else:
        arena_radius = traj_dict.get("arena_radius", None)
        arena_center = traj_dict.get("arena_center", None)

        # Find center and radius. Then override if necessary
        if arena_radius is None or arena_center is None:
            estimated_center, estimated_radius = estimate_center_and_radius(
                locations
            )

        if arena_radius is None:
            radius = estimated_radius
        else:
            radius = arena_radius

        if arena_center is None:
            center_a = estimated_center
        else:
            center_a = arena_center

    return radius, center_a


def import_idtrackerai_file(trajectories_path, **kwargs):
    traj_dict = np.load(
        trajectories_path, encoding="latin1", allow_pickle=True
    ).item()
    tr = import_idtrackerai_dict(traj_dict, **kwargs)
    tr.params["path"] = trajectories_path
    tr.params["construct_method"] = "from_idtrackerai"
    return tr


def import_idtrackerai_dict(
    traj_dict,
    interpolate_nans=True,
    center=False,
    smooth_params=None,
    dtype=np.float64,
):
    """Create Trajectories from a idtracker.ai trajectories dictionary

    :param traj_dict: idtracker.ai generated dictionary
    :param interpolate_nans: whether to interpolate NaNs
    :param center: Whether to center trajectories, using a center estimated
    from the trajectories.
    :param smooth_params: Parameters of smoothing
    :param dtype: Desired dtype of trajectories.
    """

    t = traj_dict["trajectories"].astype(dtype)
    traj = Trajectories.from_positions(
        t, interpolate_nans=interpolate_nans, smooth_params=smooth_params
    )

    radius, center_ = _radius_and_center_from_traj_dict(traj._s, traj_dict)
    traj.params.update(dict(radius=radius, radius_px=radius, _center=center_))
    if center:
        traj.origin_to(traj.params["_center"])

    traj.params["frame_rate"] = traj_dict.get("frames_per_second", None)
    traj.params["body_length_px"] = traj_dict.get("body_length", None)
    traj.params["construct_method"] = "from_idtracker_"
    return traj
