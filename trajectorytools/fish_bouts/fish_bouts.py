import trajectorytools as tt


def bout_latency(tr, bout, focal):
    return bout[2]-bout[0]


def bout_acceleration_time(tr, bout, focal):
    return bout[1]-bout[0]


def bout_gliding_time(tr, bout, focal):
    """
    It can only be interpreted as gliding time if the end of the current bout
    coincides with the beginning of the next bout
    """
    return bout[2]-bout[1]


def bout_displacement(tr, bout, focal):
    return tt.norm(tr.s[bout[2], focal] - tr.s[bout[0], focal])


def bout_turning_angle(tr, bout, focal):
    return tt.signed_angle_between_vectors(tr.v[bout[2]], tr.v[bout[0]])


def compute_bouts_parameters(tr, bouts, focal):
    variables = [bout_latency, bout_acceleration_time, bout_gliding_time,
                 bout_displacement, bout_turning_angle]
    bouts_dict = {var.__name__: [var(tr, bout, focal) for bout in bouts]
                  for var in variables}
    bouts_dict['bouts'] = bouts
    return bouts_dict


def get_bouts_parameters(tr):
    all_bouts = tr.get_bouts(prominence=(0.002, None), distance=3)
    indiv_bouts = [compute_bouts_parameters(tr, all_bouts[focal], focal)
                   for focal in range(tr.number_of_individuals)]
    return indiv_bouts
