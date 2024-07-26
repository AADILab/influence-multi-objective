from copy import deepcopy, copy
import numpy as np

def distance(pos0, pos1):
    return np.linalg.norm(pos0-pos1)


class Env():
    def __init__(self, agents, pois, map_):
        self.map = map_
        self.agents = agents
        self.pois = pois

    def update(self):
        pass

class Map():
    """Map holds the obstacles and ocean currents (unkown to robots)"""
    def __init__(self, occupancy_grid, bounds):
        self.occupancy_grid = occupancy_grid
        self.bounds = bounds

class Obstacle():
    """Obstacles are defined as multi-rectangle polygons along the occupancy grid, not in terms of actual xs and ys"""
    def __init__(self, xs, ys):
        self.xs = xs
        self.ys = ys

def main():
    obstacles = [
        Obstacle(xs=[
            [0,0],

        ],
        ys=[
            [0,0]
        ])
    ]
    occupancy_grid = np.zeros((100,100))
    bounds = np.array([100.,100.])
    map_ = Map(occupancy_grid, bounds)
