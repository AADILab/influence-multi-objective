from pathlib import Path
import os
import pandas as pd
import numpy as np

def load_waves(csv_dir):
    df = pd.read_csv(csv_dir)
    def wave_x(x):
        return df["x_a"][0]*np.sin(x/df["x_b"][0] + df["x_c"][0])
    def wave_y(y):
        return df["y_a"][0]*np.sin(y/df["y_b"][0] + df["y_c"][0])
    return wave_x, wave_y

class Mission():
    """Mission includes the connectivity grid, planned paths, pois, waves"""
    def __init__(self, mission_dir: Path):
        self.setup_dirs(mission_dir)
        self.load_mission()

    @staticmethod
    def get_path_int(path_child):
        return int(path_child.split('.')[0].split('_')[-1])

    def get_path_dirs(self, mission_dir):
        children = os.listdir(mission_dir)
        path_children = [child for child in children if child[:4] == "path"]
        path_children = sorted(path_children, key = lambda child: self.get_path_int(child))
        path_dirs = [mission_dir / child for child in path_children]
        return path_dirs

    def setup_dirs(self, mission_dir):
        self.mission_dir = mission_dir
        self.connect_dir = self.mission_dir / "connectivity.csv"
        self.poi_dir = self.mission_dir / "pois.csv"
        self.path_dirs = self.get_path_dirs(mission_dir)
        self.root_node_dir = self.mission_dir / "root_node.csv"
        self.wave_dir = self.mission_dir / "waves.csv"

    # Useful functions
    @staticmethod
    def load_connectivity_grid(csv_dir):
        df = pd.read_csv(csv_dir)
        return df.to_numpy(int)[:,1:]

    @staticmethod
    def load_pois(csv_dir):
        df = pd.read_csv(csv_dir)
        return df.to_numpy(float)[:,1:]

    @staticmethod
    def load_path(csv_dir):
        df = pd.read_csv(csv_dir)
        return df.to_numpy(float)[:,1:]

    @staticmethod
    def load_root_note(root_node_dir):
        df = pd.read_csv(root_node_dir)
        return df.to_numpy(int)[:,1:]

    def load_mission(self):
        self.connectivity_grid = self.load_connectivity_grid(self.connect_dir)
        self.pois = self.load_pois(self.poi_dir)
        self.paths = [self.load_path(path_dir) for path_dir in self.path_dirs]
        self.root_node = self.load_root_note(self.root_node_dir)
        self.wave_x, self.wave_y = load_waves(self.wave_dir)
