import collections
import functools
import itertools
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from variables import get_variable


def melt_df(df: pd.DataFrame, value_name: str, var_name: str):
    melted_df = pd.melt(
        df,
        id_vars=[col for col in df.columns if isinstance(col, str)],
        value_vars=[col for col in df.columns if not isinstance(col, str)],
        var_name=var_name,
        value_name=value_name,
    )
    return melted_df


def generate_var_df(
    x: np.ndarray,
    y: np.ndarray,
    x_name: str,
    y_name: str,
    identities: Optional[Dict[str, List]] = None,
) -> pd.DataFrame:
    assert len(x) == y.shape[0]
    df = pd.DataFrame(
        data=y.T, columns=list(x)
    )  # Transpose so that columns are time steps
    if identities is not None:
        for key in identities:
            df[key] = identities[key]
    return melt_df(df, y_name, x_name)


def get_focal_nb_ids(key: str, identities: List):
    focal_neighbour_ids_conds = list(itertools.product(identities, identities))
    focals, neighbours = list(zip(*focal_neighbour_ids_conds))
    dict_vars = {
        key: focals,
        f"{key}_nb": neighbours,
    }
    return dict_vars


def is_focal_nb_variable(tr, var_array):
    if len(var_array.shape) == 3:
        frames_cond = len(var_array) == tr.number_of_frames
        num_indiv_cond = var_array.shape[1] == tr.number_of_individuals
        focal_nb_cond = var_array.shape[1] == var_array.shape[2]
        return frames_cond and num_indiv_cond and focal_nb_cond
    else:
        return False


def is_individual_variable(tr, var_array):
    if len(var_array.shape) == 2:
        frames_cond = len(var_array) == tr.number_of_frames
        num_indiv_cond = var_array.shape[1] == tr.number_of_individuals
        return frames_cond and num_indiv_cond
    else:
        return False


def is_group_variable(tr, var_array):
    if len(var_array.shape) == 1:
        frames_cond = len(var_array) == tr.number_of_frames
        return frames_cond
    else:
        return False


def tr_variable_to_df(tr, var):
    identities = list(range(tr.number_of_individuals))

    # Get variables
    try:
        y = get_variable(var, tr)
    except ValueError:
        print(f"Cannot extract {var} from {tr}")
        return None

    print(y.shape)
    x = np.arange(len(tr))
    x_name = "frame"
    y_name = var["name"]

    if is_focal_nb_variable(tr, y):  # (frames, num_indiv, num_indiv)
        y = np.reshape(y, (y.shape[0], -1))
        identity_dict = get_focal_nb_ids("identity", identities)
        var_df = generate_var_df(x, y, x_name, y_name, identity_dict)
    elif is_individual_variable(tr, y):  # (num_frames, num_indiv)
        identity_dict = {"identity": identities}
        var_df = generate_var_df(x, y, x_name, y_name, identity_dict)
    elif is_group_variable(tr, y):  # (num_frames,)
        var_df = generate_var_df(x, y[:, np.newaxis], x_name, y_name)
    else:
        print(f"With var {var}")
        print(f"With obj {tr}")
        raise Exception(
            f"Number of dimensions of y array is {y.ndim} not valid"
        )

    return var_df


def tr_variables_to_df(tr, variables: List):
    assert len(variables) > 0
    assert tr is not None

    vars_dfs = []
    for variable in variables:
        vars_df = tr_variable_to_df(tr, variable)
        if vars_df is not None:
            vars_dfs.append(vars_df)

    print([len(df) for df in vars_dfs])
    assert all([len(df) == len(vars_dfs[0]) for df in vars_dfs])

    all_cols = [c for df in vars_dfs for c in df.columns]
    common_cols = [
        col
        for col, count in collections.Counter(all_cols).items()
        if count > 1
    ]
    if common_cols:
        vars_df = functools.reduce(
            lambda x, y: pd.merge(x, y, on=common_cols),
            vars_dfs,
        )
    else:
        vars_df = functools.reduce(
            lambda x, y: pd.merge(x, y, left_index=True, right_index=True),
            vars_dfs,
        )
    if "identity_nb" in vars_df:
        # Focal neighbour variable
        vars_df.set_index(["identity", "identity_nb", "frame"], inplace=True)
    elif "identity" in vars_df:
        # Focal variable
        vars_df.set_index(["identity", "frame"], inplace=True)
    else:
        # Group variable
        vars_df.set_index(["frame"], inplace=True)
    return vars_df


if __name__ == "__main__":
    import trajectorytools.constants as cons
    from trajectorytools.trajectories import Trajectories

    from variables import (
        GROUP_VARIABLES,
        INDIVIDUAL_NEIGHBOUR_VARIABLES,
        INDIVIDUAL_VARIALBES,
    )

    tr = Trajectories.from_idtrackerai(
        cons.test_trajectories_path, interpolate_nans=True
    )
    indiv_df = tr_variables_to_df(tr, INDIVIDUAL_VARIALBES)
    indiv_nb_df = tr_variables_to_df(tr, INDIVIDUAL_NEIGHBOUR_VARIABLES)
    indiv_nb_df = tr_variables_to_df(tr, GROUP_VARIABLES)
