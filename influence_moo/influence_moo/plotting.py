import matplotlib.pyplot as plt
import numpy as np
from influence_moo.waves import sample_waves

# Low level utility plotting functions

def plot_grid(grid, ax=None, *args, **kwargs):
    """This is a utility function that plots grids with the correct (x,y) positions"""
    if ax is None:
        plt.imshow(np.rot90(grid), *args, **kwargs, extent=(0.0, grid.shape[0], 0.0, grid.shape[1]))
    else:
        ax.imshow(np.rot90(grid), *args, **kwargs, extent=(0.0, grid.shape[0], 0.0, grid.shape[1]))

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
            plt.plot([vector.startpt[0]], [vector.startpt[1]], marker= 'o',*args, **kwargs)
        else:
            ax.plot([vector.startpt[0], vector.endpt[0]], [vector.startpt[1], vector.endpt[1]], *args, **kwargs)
            ax.plot([vector.startpt[0]], [vector.startpt[1]], marker= 'o',*args, **kwargs)

# High level convenience plotting functions

def plot_mission(mission, ax=None, include_waves=True):
    plot_grid(mission.connectivity_grid, ax, cmap='tab10_r')
    for asv_start_position in mission.asv_start_positions:
        plot_pts(np.array([asv_start_position]), ax, '+', color='orange')
    colors = ['pink', 'purple', 'tab:cyan', 'tab:orange']
    ls_l = [(0,(1,2)), 'dashed', 'dashdot', (0,(1,3))]
    for i, path in enumerate(mission.paths):
        plot_pts(path, ax, ls=ls_l[i%len(ls_l)], color=colors[i%len(colors)], lw=1)
    plot_pts(mission.pois, ax, marker='o', fillstyle='none', linestyle='none',color='tab:green')
    if include_waves:
        wave_vectors = sample_waves(mission.connectivity_grid, 75, 75, mission.wave_x, mission.wave_y)
        plot_vectors(wave_vectors, ax, color='navy',lw=0.3)
    if ax is not None:
        ax.set_xlim([0,mission.connectivity_grid.shape[0]])
        ax.set_ylim([0,mission.connectivity_grid.shape[1]])
    else:
        plt.xlim([0, mission.connectivity_grid.shape[0]])
        plt.ylim([0, mission.connectivity_grid.shape[1]])

def plot_rollout(env, ax=None,include_waves=True):
    plot_grid(env.mission.connectivity_grid, ax, cmap='tab10_r')

    plot_pts(env.mission.pois, ax, marker='o', fillstyle='none', linestyle='none',color='tab:green')

    colors = ['pink', 'purple', 'tab:cyan', 'tab:orange']
    for i, auv in enumerate(env.auvs):
        plot_pts(np.array(auv.path), ax, ls='solid', color=colors[i%len(colors)], lw=0.5)

    for asv in env.asvs:
        plot_pts(np.array(asv.path), ax, ls='solid', color='magenta', lw=0.5)

    for auv in env.auvs:
        if auv.crashed:
            plot_pts(np.array([auv.position]), ax, 'x', color='tab:red')

    for asv in env.asvs:
        if asv.crashed:
            plot_pts(np.array([asv.position]), ax, 'x', color='tab:red')

    if include_waves:
        wave_vectors = sample_waves(env.mission.connectivity_grid, 75, 75, env.mission.wave_x, env.mission.wave_y)
        plot_vectors(wave_vectors, ax, color='navy', lw=0.3)
