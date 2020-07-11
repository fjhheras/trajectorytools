import os
import pathlib

dir_of_executable = os.path.dirname(__file__)
path_to_project_root = os.path.abspath(dir_of_executable)
dir_of_data = path_to_project_root + "/data/"

test_trajectories_path = (
    pathlib.Path(dir_of_data) / "test_trajectories_idtrackerai.npy"
)
test_trajectories_path_border = (
    pathlib.Path(dir_of_data) / "test_trajectories_idtrackerai_with_border.npy"
)
test_raw_trajectories_path = (
    pathlib.Path(dir_of_data) / "test_trajectories.npy"
)
test_trajectories_with_points_path = (
    pathlib.Path(dir_of_data) / "trajectories_with_points.npy"
)
