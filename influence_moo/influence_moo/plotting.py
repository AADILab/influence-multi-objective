import matplotlib.pyplot as plt
import numpy as np
from influence_moo.waves import sample_waves

# Low level utility plotting functions

def plot_grid(grid, ax=None, *args, **kwargs):
    """This is a utility function that plots grids with the correct (x,y) positions"""
    if ax is None:
        plt.imshow(np.rot90(grid), *args, **kwargs, extent=(0.0, grid.shape[1], 0.0, grid.shape[0]))
    else:
        ax.imshow(np.rot90(grid), *args, **kwargs, extent=(0.0, grid.shape[1], 0.0, grid.shape[0]))

def plot_pts(pts, ax=None, *args, **kwargs):
    """This is a utility function that plots the (x,y) points specified"""
    if ax is None:
        plt.plot(pts[:,0], pts[:,1], *args, **kwargs)
    else:
        ax.plot(pts[:,0], pts[:,1], *args, **kwargs)

def plot_vectors(vectors, ax=None, *args, **kwargs):
    """This is a utility function that plots waves as arrows following the wave gradient on a grid"""
    for vector in vectors:
        if ax is None:
            plt.plot([vector.startpt[0], vector.endpt[0]], [vector.startpt[1], vector.endpt[1]], *args, **kwargs)
        else:
            ax.plot([vector.startpt[0], vector.endpt[0]], [vector.startpt[1], vector.endpt[1]], *args, **kwargs)

# High level convenience plotting functions

def plot_mission(mission, ax=None):
    plot_grid(mission.connectivity_grid, ax, cmap='tab10_r')
    plot_pts(mission.root_node, ax, '+', color='orange')
    plot_pts(mission.pathA, ax, ls=(0, (1,2)), color='pink', lw=1)
    plot_pts(mission.pathB, ax, ls='dashed', color='purple', lw=1)
    plot_pts(mission.pathC, ax, ls='dashdot', color='tab:cyan', lw=1)
    plot_pts(mission.pathD, ax, ls=(0, (1,3)), color='tab:orange', lw=1)
    plot_pts(mission.pois, ax, marker='o', fillstyle='none', linestyle='none',color='tab:green')
    wave_vectors = sample_waves(mission.connectivity_grid, 75, 75, mission.wave_x, mission.wave_y)
    plot_vectors(wave_vectors, ax, color='navy',lw=0.3)

def plot_rollout(env, ax=None):
    plot_grid(env.mission.connectivity_grid, ax, cmap='tab10_r')

    plot_pts(env.mission.pois, ax, marker='o', fillstyle='none', linestyle='none',color='tab:green')

    plot_pts(np.array(env.auvs[0].path), ax, ls='solid', color='pink', lw=0.5)
    plot_pts(np.array(env.auvs[1].path), ax, ls='solid', color='purple', lw=0.5)
    plot_pts(np.array(env.auvs[2].path), ax, ls='solid', color='tab:cyan', lw=0.5)
    plot_pts(np.array(env.auvs[3].path), ax, ls='solid', color='tab:orange', lw=0.5)

    for asv in env.asvs:
        plot_pts(np.array(asv.path), ax, ls='solid', color='magenta', lw=0.5)

    for auv in env.auvs:
        if auv.crashed:
            plot_pts(np.array([auv.position]), ax, 'x', color='tab:red')

    for asv in env.asvs:
        if asv.crashed:
            plot_pts(np.array([asv.position]), ax, 'x', color='tab:red')

    wave_vectors = sample_waves(env.mission.connectivity_grid, 75, 75, env.mission.wave_x, env.mission.wave_y)
    plot_vectors(wave_vectors, ax, color='navy', lw=0.3)
