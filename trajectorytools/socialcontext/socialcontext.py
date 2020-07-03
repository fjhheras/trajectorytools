import numpy as np
import scipy.spatial
import scipy.spatial.distance as spdist
from sklearn.neighbors import NearestNeighbors
import warnings


def _in_convex_hull(positions):
    hull = scipy.spatial.ConvexHull(positions)
    convex_hull = np.zeros(positions.shape[0], dtype=np.bool)
    convex_hull[hull.vertices] = True
    return convex_hull


def in_convex_hull(positions):
    convex_hull_list = [
        _in_convex_hull(positions_in_frame) for positions_in_frame in positions
    ]
    return np.stack(convex_hull_list, axis=0)


def circumradius(points):
    """Radius of the circumcentre
    defined by three points.
    """
    # Sides of triangles
    side_vectors = points - np.roll(points, 1, axis=-2)
    sides = np.sqrt(side_vectors[..., 0] ** 2 + side_vectors[..., 1] ** 2)
    a = sides[..., 0]
    b = sides[..., 1]
    c = sides[..., 2]
    s = (a + b + c) / 2
    area = np.sqrt((s - a) * (s - b) * (s - c) * s)
    return a * b * c / (4 * area)


def _in_alpha_border(positions, alpha=5):
    """Calculate vertices in border of alpha-shape
    by pruning a Delaunay triangulation.

    Border points are either:
    1. In convex hull
    2. In rejected triangles
    """

    def radius_too_large(triangles):
        return circumradius(triangles) > 1 / alpha

    num_individuals, _ = positions.shape
    delaunay = scipy.spatial.Delaunay(positions)
    triangles = np.array(
        [positions[triangle] for triangle in delaunay.simplices]
    )
    rejected_triangles = radius_too_large(triangles)
    points_in_rejected_triangles = [
        p
        for triangle in delaunay.simplices[rejected_triangles]
        for p in triangle
    ]
    in_border = np.zeros(num_individuals, np.bool)
    in_border[points_in_rejected_triangles] = True
    in_border[delaunay.convex_hull] = True
    return in_border


def in_alpha_border(positions, alpha=5):
    alpha_border_list = [
        _in_alpha_border(positions_in_frame, alpha=alpha)
        for positions_in_frame in positions
    ]
    return np.stack(alpha_border_list, axis=0)


# LOCAL NEIGHBOURS


def neighbour_indices_in_frame(
    positions: np.ndarray, num_neighbours: int,
) -> np.ndarray:
    """ Calculate the indices of the nearest neighbours

    :param positions: array of locations with dimensions
    (individual x coordinates)
    :param num_neighbours: number of closest neighbours requested
    :return: output dime (individual x num_neighbours + 1)
    """
    nbrs = NearestNeighbors(
        n_neighbors=num_neighbours + 1, algorithm="ball_tree"
    ).fit(positions)
    return nbrs.kneighbors(positions, return_distance=False)


def give_indices(positions, num_neighbours):
    warnings.warn(
        "give_indices to be deprecated. Use neighbour_indices instead"
    )
    return neighbour_indices(positions, num_neighbours)


def neighbour_indices(
    positions: np.ndarray, num_neighbours: int
) -> np.ndarray:
    """ Calculates the indices of the nearest neighbours

    :param positions: array of locations with dimensions
    (time x individual x coordinates)
    :param num_neighbours: number of closest neighbours (does not
    include the focal, e.g. max is individuals - 1)
    :return: array with dims (time x individual x num_neighbours + 1)
    """
    total_time_steps = positions.shape[0]
    individuals = positions.shape[1]
    next_neighbours = np.empty(
        [total_time_steps, individuals, num_neighbours + 1], dtype=np.int
    )
    for frame in range(total_time_steps):
        next_neighbours[frame, ...] = neighbour_indices_in_frame(
            positions[frame], num_neighbours
        )
    return next_neighbours


def adjacency_matrix_in_frame(
    positions: np.ndarray, num_neighbours: int, mode: str = "connectivity",
) -> np.ndarray:
    """
    :param positions: array of locations with dimensions
    (individual x coordinates)
    :param num_neighbours: number of closest neighbours requested
    :param mode: adjacency mode "connectivity" or "distance".     
    :return: output has dimension (individual x individual)
    """
    nbrs = NearestNeighbors(
        n_neighbors=num_neighbours + 1, algorithm="ball_tree"
    ).fit(positions)
    return nbrs.kneighbors_graph(positions, mode=mode).toarray()


def adjacency_matrix(
    positions,
    num_neighbours=None,
    mode="connectivity",
    use_pdist_if_all_nb=True,
):
    total_time_steps = positions.shape[0]
    individuals = positions.shape[1]
    if num_neighbours is None:
        num_neighbours = individuals - 1
    if mode == "connectivity":
        adjacency_m = np.empty(
            [total_time_steps, individuals, individuals], dtype=np.bool
        )
    elif mode == "distance":
        adjacency_m = np.empty(
            [total_time_steps, individuals, individuals], dtype=positions.dtype
        )
    else:
        raise ValueError("mode should be 'connectivity' or 'distance'")

    if (num_neighbours == individuals - 1) and use_pdist_if_all_nb:
        if mode == "connectivity":
            adjacency_m[...] = True
        else:
            for frame in range(total_time_steps):
                adjacency_m[frame, ...] = spdist.squareform(
                    spdist.pdist(positions[frame])
                )
    else:
        for frame in range(total_time_steps):
            adjacency_m[frame, ...] = adjacency_matrix_in_frame(
                positions[frame], num_neighbours, mode=mode
            )
    return adjacency_m


def interindividual_distances(positions):
    return adjacency_matrix(positions, mode="distance")


def restrict(data, indices, individual=None):
    """restrict_with_delay

    :param data: np.array with shape time x individuals x coordinates
    :param indices: np.array with shape 
    time x individuals x subset_of_individuals
    :param individual: label of individual to calculate restriction 
    with delay. If None, calculating for all individuals
    """
    num_restricted = indices.shape[-1]
    total_time_steps = data.shape[0]
    num_individuals = data.shape[1]
    coordinates = data.shape[-1]
    if individual is None:
        output_data = np.empty(
            [total_time_steps, num_individuals, num_restricted, coordinates],
            dtype=data.dtype,
        )
        for frame in range(total_time_steps):
            output_data[frame, ...] = data[frame, indices[frame, :], :]
    else:
        output_data = np.empty(
            [total_time_steps, num_restricted, coordinates], dtype=data.dtype
        )
        for frame in range(total_time_steps):
            output_data[frame, ...] = data[
                frame, indices[frame, individual], :
            ]

    return output_data
