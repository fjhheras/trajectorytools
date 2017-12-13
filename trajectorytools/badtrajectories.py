import numpy as np
import random
from sklearn.neighbors import NearestNeighbors

from trajectorytools import Trajectories

# This library is not for general use. It is used to check for the effect of bad detection of 
# identities after crossings by idTracker.ia.
# Therefore, the convention of idTracker is used, i.e. np.NaN when there is a crossing

def crossing_pairs_in_frame(frame, next_frame, max_distance = 90, frame_index = None):
    """Returns crossings in a frame, given information of the current frame and the
    following one. 

    In this simple code, crossings are considered only if:
    * Both individuals are seen in frame
    * Both individuals are not seen if next_frame
    * distance among them is smaller than max_distance

    Crossings of more than two individuals are only lazily accounted for, and weird things can happen if
    they exist!

    :param frame: numpy array with size num_individuals x 2
    :param next_frame: numpy array with size num_individuals x 2
    :param max_distance: maximum distance to consider as crossing
    :param frame_index: frame index, used only as a tag in the first element of the crossing
    """
    num_individuals = frame.shape[0]
    crossing_index = np.logical_and(np.logical_not(np.isnan(frame[:,1])), np.isnan(next_frame[:,1]))
    num_crossing = np.sum(crossing_index)
    frame_crossing = frame[crossing_index]
    id_crossing = np.arange(num_individuals)[crossing_index] 
    crossing_pairs = []
    if num_crossing > 1:
        nbrs = NearestNeighbors(n_neighbors=2, algorithm='ball_tree').fit(frame_crossing)
        distances_, indices_ = nbrs.kneighbors(frame_crossing)
        distance_to_nearest = list(distances_[:,1])
        indice_nearest = list(id_crossing[indices_[:,1]])
        changed = set([]) ## Set of indices that have been detected in crossings
        for dist, i, nearest in sorted(zip(distance_to_nearest, list(id_crossing), indice_nearest)):
            if dist > max_distance:
                break
            if not ((i in changed) or (nearest in changed)):
                crossing_pairs.append([frame_index,dist,i,nearest])
                changed.add(i)
                changed.add(nearest)
    return crossing_pairs 

def get_crossing_pairs(t):
    """Obtain a list of crossings

    :param t: np.array
    """
    return [crossings for i in range(t.shape[0]-1) for crossings in crossing_pairs_in_frame(t[i],t[i+1], frame_index = i)]

def bad_in_list_of_crossings(t, crossing_pairs):
    """Produce trajectories with tracking failing in a list of crossings.
    
    Returns a version of t that would have been produced by a tracking system swapping the identity
    at every crossing of the list.

    :param t: numpy array
    :param crossing_pairs: list of crossings [frame, distance, first_individual, second_individual]
    """
    badt = t.copy()
    for frame,_,a,b in reversed(crossing_pairs):
        badt[(frame+1):,[a,b]] = badt[(frame+1):,[b,a]]
    return badt

def reverse_bad_crossings(t, crossing_pairs):
    """Undoes bad_in_list_of_crossings

    :param t: numpy array
    :param crossing_pairs: list of crossings
    """
    goodt = t.copy()
    for frame,_,a,b in crossing_pairs:
        goodt[(frame+1):,[a,b]] = goodt[(frame+1):,[b,a]]
    return goodt

def fail_a_fraction_of_crossings(t, fail_fraction = 0.1):
    crossing_pairs = get_crossing_pairs(t) 
    badcrossing_pairs = int(len(crossing_pairs)*fail_fraction)
    sublist_crossing_pairs = sorted(random.sample(crossing_pairs, badcrossing_pairs))
    badt = bad_in_list_of_crossings(t, sublist_crossing_pairs)
    return badt

class BadTrajectories(Trajectories):
    """BadTrajectories
    A thin wrapper around trajectorytools.Trajectories that messes up the identities in a fraction
    of simple (two-individual) crossings.
    """
    @classmethod
    def from_positions(cls, t, fail_fraction = 0.1):
        badt = fail_a_fraction_of_crossings(t, fail_fraction = fail_fraction)
        return Trajectories.from_positions(badt)
    @classmethod
    def from_idtracker(cls, trajectories_path, fail_fraction = 0.1):
        trajectories_dict = np.load(trajectories_path, encoding = 'latin1').item()
        t = trajectories_dict['trajectories']
        return cls.from_positions(t, fail_fraction = fail_fraction)


