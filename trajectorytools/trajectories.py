from argparse import Namespace
import numpy as np
import trajectorytools as tt 
import trajectorytools.socialcontext as ttsocial

class Trajectories():
    def __init__(self, t):
        tt.normalise_trajectories(t)
        tt.interpolate_nans(t)
        [self.s, self.v, self.a] = tt.smooth_several(t, derivatives=[0,1,2])
        self.speed = tt.norm(self.v)
        self.acceleration = tt.norm(self.a)
        self.distance_to_center = tt.norm(self.s)
        self.e = tt.normalise(self.v)
        self.curvature = tt.curvature(self.v, self.a)
        self.center_of_mass = self.calculate_center_of_mass()
    @classmethod
    def from_idtracker(cls, trajectories_path):
        trajectories_dict = np.load(trajectories_path, encoding = 'latin1').item()
        ### Bring here the properties that we need from the dictionary
        t = trajectories_dict['trajectories']
        return cls(t)
    @property
    def number_of_frame(self):
        return self.s.shape[0]
    @property
    def number_of_individuals(self):
        return self.s.shape[1]
    def calculate_center_of_mass(self):
        s = np.average(self.s, axis = -2)
        v = np.average(self.v, axis = -2)
        speed = tt.norm(v) 
        distance_to_center = tt.norm(s)
        return Namespace(s = s, v = v, speed = speed, distance_to_center = distance_to_center)




