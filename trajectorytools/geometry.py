import numpy as np

def dot(x,y):
    return np.einsum('...i,...i->...', x, y)

def matrix_dot(matrix, data):
    if len(matrix.shape) == 2:
        # Same matrix for everyone
        return np.einsum('ij,...j->...i')
    elif len(matrix.shape) == 3:
        # Each frame has a matrix
        return np.einsum('kij,k...j->k...i', matrix, data)
    elif len(data.shape) == len(matrix.shape) - 1:
        # Each frame and individual has a matrix
        return np.einsum('...ij,...j->...i', matrix, data)
    else:
        raise Exception('matrix_dot: wrong input dimension')

def norm(a):
    return np.linalg.norm(a, axis = -1)

def normalise(a):
    return a / norm(a)[...,np.newaxis]

#def rotate_to_vector(data, v):
#    return fixed_to_comoving(data, normalise(v))

def fixed_to_comoving(data, e_y):
    return matrix_dot(matrix_rotate_to_vector(e_y), data)

def matrix_rotate_to_vector(e_y):
    e_x = _ey_to_ex(e_y)
    return np.stack((e_x,e_y), axis = -2)

def center_in_trajectory(data, trajectory):
    return data - np.expand_dims(trajectory,1)

def center_in_individual(data, individual):
    return center_in_trajectory(data, data[:,individual,:])

def angle_between_vectors(x,y):
    u,v = normalise(x),normalise(y)
    return np.arccos(np.clip(dot(u,v), -1.0, +1.0))

def _ey_to_ex(e_):
    '''Takes a vector and rotates it -90 degrees
    it takes a unit vector ey into ex'''
    ex = np.empty_like(e_)
    ex[...,1] = -e_[...,0]
    ex[...,0] = e_[...,1]
    return ex

