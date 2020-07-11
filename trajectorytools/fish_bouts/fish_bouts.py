import trajectorytools as tt


def bout_statistics():
    def latency(tr, bout, focal):
        return bout[2] - bout[0]

    def acceleration_time(tr, bout, focal):
        return bout[1] - bout[0]

    def gliding_time(tr, bout, focal):
        """
        It can only be interpreted as gliding time if the end of the current bout
        coincides with the beginning of the next bout
        """
        return bout[2] - bout[1]

    def location_start(tr, bout, focal):
        return tr.s[bout[0], focal]

    def location_end(tr, bout, focal):
        return tr.s[bout[1], focal]

    def location_end_gliding(tr, bout, focal):
        return tr.s[bout[2], focal]

    def displacement(tr, bout, focal):
        return tt.norm(tr.s[bout[1], focal] - tr.s[bout[0], focal])

    def displacement_with_gliding(tr, bout, focal):
        return tt.norm(tr.s[bout[2], focal] - tr.s[bout[0], focal])

    def turning_angle(tr, bout, focal):
        return tt.signed_angle_between_vectors(
            tr.v[bout[1], focal], tr.v[bout[0], focal]
        )

    def turning_angle_with_gliding(tr, bout, focal):
        return tt.signed_angle_between_vectors(
            tr.v[bout[2], focal], tr.v[bout[0], focal]
        )

    def turning_angle_old(tr, bout, focal):
        return tt.angle_between_vectors(
            tr.v[bout[1], focal], tr.v[bout[0], focal]
        )

    return [
        value
        for key, value in locals().items()
        if callable(value) and not key.startswith("__")
    ]


def compute_bouts_parameters(tr, bouts, focal):
    variables = bout_statistics()
    bouts_dict = {
        var.__name__: [var(tr, bout, focal) for bout in bouts]
        for var in variables
    }
    bouts_dict["bouts"] = bouts
    return bouts_dict


def get_bouts_parameters(tr, find_max_dict=None, find_min_dict=None):
    all_bouts = tr.get_bouts(find_max_dict, find_min_dict)
    indiv_bouts = [
        compute_bouts_parameters(tr, all_bouts[focal], focal)
        for focal in range(tr.number_of_individuals)
    ]
    return indiv_bouts
