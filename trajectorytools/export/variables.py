import numpy as np
import trajectorytools as tt
import trajectorytools.socialcontext as ttsocial


def get_variable(variable, tr):
    print(f"Using function {variable['func']}")
    kwargs = variable.get("kwargs", {})
    print(f"With kwargs {kwargs}")
    return variable["func"](tr, **kwargs)


# Individual variables (frames, num_indiv) shape
def s_x(tr):
    return tr.s[..., 0]


def s_y(tr):
    return tr.s[..., 1]


def v_x(tr):
    return tr.v[..., 0]


def v_y(tr):
    return tr.v[..., 1]


def a_x(tr):
    return tr.a[..., 0]


def a_y(tr):
    return tr.a[..., 1]


def e_x(tr):
    return tr.e[..., 0]


def e_y(tr):
    return tr.e[..., 1]


def distance_to_origin(tr):
    return tr.distance_to_origin


def speed(tr):
    return tr.speed


def acceleration(tr):
    return tr.acceleration


def normal_acceleration(tr):
    return tr.normal_acceleration


def abs_normal_acceleration(tr):
    return np.abs(tr.normal_acceleration)


def tg_acceleration(tr):
    return tr.tg_acceleration


def distance_to_center_of_group(tr):
    distance_to_group_center = tt.norm(
        tr.center_of_mass.s[:, np.newaxis, :] - tr.s
    )
    return distance_to_group_center


def local_polarization(tr, number_of_neighbours=4):
    indices = ttsocial.neighbour_indices(tr.s, number_of_neighbours)
    en = ttsocial.restrict(tr.e, indices)[..., 1:, :]
    local_polarization = tt.norm(tt.collective.polarization(en))
    return local_polarization


def average_alignment_score(tr, number_of_neighbours=4):
    indices = ttsocial.neighbour_indices(tr.s, number_of_neighbours)
    en = ttsocial.restrict(tr.e, indices)[..., 1:, :]
    alignment = np.nanmean(tt.dot(np.expand_dims(tr.e, 2), en), axis=-1)
    return np.nanmedian(alignment, axis=-1)


def distance_travelled(tr):
    return tr.distance_travelled


def _focal_aligned_accelerations(tr):
    a_rotated = tt.fixed_to_comoving(tr.a, tr.e)
    for focal in range(tr.number_of_individuals):
        a_rotated[:, focal, :] = tt.fixed_to_comoving(
            tr.a[:, focal], tr.e[:, focal]
        )  # Rotate the focal's
    return a_rotated


def focal_turn_accel(tr):
    return _focal_aligned_accelerations(tr)[..., 0]


def focal_fwd_accel(tr):
    return _focal_aligned_accelerations(tr)[..., 1]


INDIVIDUAL_VARIALBES = [
    {"name": "s_x", "func": s_x},
    {"name": "s_y", "func": s_y},
    {"name": "v_x", "func": v_x},
    {"name": "v_y", "func": v_y},
    {"name": "a_x", "func": a_x},
    {"name": "a_y", "func": a_y},
    {"name": "e_x", "func": e_x},
    {"name": "e_y", "func": e_y},
    {"name": "distance_to_origin", "func": distance_to_origin},
    {"name": "speed", "func": speed},
    {"name": "acceleration", "func": acceleration},
    {"name": "normal_acceleration", "func": normal_acceleration},
    {"name": "abs_normal_acceleration", "func": abs_normal_acceleration},
    {"name": "tg_acceleration", "func": tg_acceleration},
    {
        "name": "distance_to_center_of_group",
        "func": distance_to_center_of_group,
    },
    {
        "name": "local_polarization",
        "func": local_polarization,
        "kwargs": {"number_of_neighbours": 4},
    },
    {"name": "distance_travelled", "func": distance_travelled},
    {"name": "focal_turn_accel", "func": focal_turn_accel},
    {"name": "focal_fwd_accel", "func": focal_fwd_accel},
]

# Individual-neighbour varialbes (frames, num_indiv, num_indiv)
def _relative_positions_rotated(tr):
    s_rotated = np.empty(
        (
            tr.number_of_frames,
            tr.number_of_individuals,
            tr.number_of_individuals,
            2,
        )
    )
    for focal in range(tr.number_of_individuals):
        s = tt.center_in_individual(tr.s, focal)
        sr = tt.fixed_to_comoving(s, tr.e[:, focal, :])
        sr[
            :, focal
        ] = (
            np.nan
        )  # Change rotated positions of the focal ittr from '0' to 'NaN'.
        s_rotated[:, focal, :, :] = sr
    return s_rotated


def nb_position_x(tr):
    return _relative_positions_rotated(tr)[..., 0]


def nb_position_y(tr):
    return _relative_positions_rotated(tr)[..., 1]


def nb_angle(tr):
    """
    -pi, pi is on the left
    -0, 0 is on the right
    pi/2 is on the front
    -pi/2 is on the back
    :return:
    """
    return np.arctan2(nb_position_y(tr), nb_position_x(tr))


def nb_cos_angle(tr):
    return np.cos(nb_angle(tr))


def nb_distance(tr):
    return tr.interindividual_distances


INDIVIDUAL_NEIGHBOUR_VARIABLES = [
    {"name": "nb_position_x", "func": nb_position_x},
    {"name": "nb_position_y", "func": nb_position_y},
    {"name": "nb_angle", "func": nb_angle},
    {"name": "nb_cos_angle", "func": nb_cos_angle},
    {"name": "nb_distance", "func": nb_distance},
]

# GROUP VARIABLES (num_frames,)
def mean_distance_to_center_of_group(tr):
    return np.nanmean(distance_to_center_of_group(tr), axis=1)


def average_local_polarization(tr, number_of_neighbours=4):
    return np.nanmean(
        local_polarization(tr, number_of_neighbours=number_of_neighbours),
        axis=-1,
    )


def polarization_order_parameter(tr):
    return tt.norm(tt.collective.polarization(tr.e))


def rotation_order_parameter(tr):
    return (
        tt.collective.angular_momentum(tr.e, tr.s, center=tr.center_of_mass.s)
        / tr.number_of_individuals
    )


GROUP_VARIABLES = [
    {
        "name": "mean_distance_to_center_of_group",
        "func": mean_distance_to_center_of_group,
    },
    {
        "name": "average_local_polarization",
        "func": average_local_polarization,
        "kwargs": {"number_of_neighbours": 4},
    },
    {
        "name": "polarization_order_parameter",
        "func": polarization_order_parameter,
    },
    {
        "name": "rotation_order_parameter",
        "func": polarization_order_parameter,
    },
]
