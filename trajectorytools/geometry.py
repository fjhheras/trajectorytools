import numpy as np

EPSILON = 1e-16

# SIMPLE TOOLS


def dot(x, y, keepdims=False):
    result = np.einsum("...i,...i->...", x, y)
    if keepdims:
        return np.expand_dims(result, -1)
    else:
        return result


def cross2D(v, w, keepdims=False):
    if keepdims:
        return v[..., [0]] * w[..., [1]] - v[..., [1]] * w[..., [0]]
    else:
        return v[..., 0] * w[..., 1] - v[..., 1] * w[..., 0]


def matrix_dot(matrix, data):
    if len(matrix.shape) == 2:
        # Same matrix for everyone
        return np.einsum("ij,...j->...i", matrix, data)
    elif len(matrix.shape) == 3:
        # Each frame has a matrix
        return np.einsum("kij,k...j->k...i", matrix, data)
    elif len(data.shape) == len(matrix.shape) - 1:
        # Each frame, individual (and maybe neighbour) has a matrix
        return np.einsum("...ij,...j->...i", matrix, data)
    elif len(data.shape) == len(matrix.shape) == 4:
        # Each frame and individual has a matrix. There are neighbours
        # but all neighbours use same matrix
        return np.einsum("...ij,...kj->...ki", matrix, data)
    else:
        raise Exception("matrix_dot: wrong input dimension")


def norm(a, keepdims=False):
    return np.linalg.norm(a, axis=-1, keepdims=keepdims)


def normalise(a):
    return a / norm(a)[..., np.newaxis]


# GEOMETRY


def curvature(v, a):
    assert (v.shape[-1]) == 2
    assert np.all(v.shape == a.shape)
    return (v[..., 0] * a[..., 1] - v[..., 1] * a[..., 0]) / np.power(
        norm(v), 3
    )


def distance_travelled(s):
    ds = norm(np.diff(s, axis=0))
    distance = np.zeros(s.shape[:-1])
    distance[1:] = np.cumsum(ds, axis=0)
    return distance


def straightness(s, epsilon=EPSILON):
    norm_displacement = norm(s[-1] - s[0])
    return (norm_displacement + epsilon) / (
        distance_travelled(s)[-1] + epsilon
    )


# TOOLS FOR ROTATING


def fixed_to_comoving(data, e_y):
    return matrix_dot(matrix_rotate_to_vector(e_y), data)


def comoving_to_fixed(data, e_y):
    matrices = matrix_rotate_to_vector(e_y)
    transposed_matrices = np.swapaxes(matrices, -1, -2)
    return matrix_dot(transposed_matrices, data)


def matrix_rotate_to_normalised_vector(e_y):
    e_x = _ey_to_ex(e_y)
    return np.stack((e_x, e_y), axis=-2)


def matrix_rotate_to_vector(v):
    e_y = normalise(v)
    return matrix_rotate_to_normalised_vector(e_y)


def center_in_trajectory(data, trajectory):
    return data - np.expand_dims(trajectory, 1)


def center_in_individual(data, individual):
    return center_in_trajectory(data, data[:, individual, :])


def angle_between_vectors(x, y):
    """Angle between vectors, between 0 and pi radians, no sign
    """
    u, v = normalise(x), normalise(y)
    return np.arccos(np.clip(dot(u, v), -1.0, +1.0))


def signed_angle_between_vectors(x, y):
    """A different algorithms to calculate angle between vectors,
    between -pi and pi radians
    """
    angle = np.arctan2(y[..., 1], y[..., 0]) - np.arctan2(x[..., 1], x[..., 0])
    return (angle + np.pi) % (2 * np.pi) - np.pi


def _ey_to_ex(e_):
    """Takes a vector and rotates it -90 degrees
    it takes a unit vector ey into ex"""
    ex = np.empty_like(e_)
    ex[..., 1] = -e_[..., 0]
    ex[..., 0] = e_[..., 1]
    return ex
