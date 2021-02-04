# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](http://keepachangelog.com/en/1.0.0/).


## [0.3.3] - Unreleased

### Added

- Now it is possible to export trajectories to csv

### Removed

- Deprecated unused code (`average_across_individuals`, `sum_across_individuals`)

### Changed

- Removing some deprecation warnings (`center_trajectories_and_obtain_radius`, `center_trajectories_and_normalise`) because an alternative is not available yet.
- Adding deprecation warnings (`Trajectory.orientation_towards`)
- Deprecation warning from `from_idtracker` (use `from_idtrackerai` instead)
- Compute `arena_center` if does not exist in `radius_and_center_from_traj_dict`

### Minor (less important or not affecting user)

- Minor refactoring (`find_bouts_individual`)
- Black version 20.8
- updated README to warn about the two missing frames due to the computation of the velocity and acceleration.
- [created CHANGELOG](https://github.com/fjhheras/trajectorytools/pull/33)

## [0.3.2] - 2020-07-11

### Added

- Added some utilities to perform polar plots to `plot/polar.py`

### Changed

- Changed function names in `social_context`, and made the per-frame versions public (`neighbour_indices_in_frame`, `adjacency_matrix_in_frame`)
