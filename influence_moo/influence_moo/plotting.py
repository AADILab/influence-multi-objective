import matplotlib.pyplot as plt
import numpy as np
from influence_moo.waves import sample_waves
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import pprint
from influence_moo.config import load_config
from pathlib import Path

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
def plot_rollout(env, ax=None,include_waves=True):
    plot_grid(env.connectivity_grid, ax, cmap='tab10_r')

    plot_pts(env.pois, ax, marker='o', fillstyle='none', linestyle='none',color='tab:green')

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
        wave_vectors = sample_waves(env.connectivity_grid, 75, 75, env.wave_x, env.wave_y)
        plot_vectors(wave_vectors, ax, color='navy', lw=0.3)

# Functions for plotting commandline tools
def get_nps(trials_dir):
    dind_trials = [str_ for str_ in os.listdir(trials_dir) if str_[:5]=="trial"]
    dind_trials = sorted(dind_trials, key=lambda x: int(x.split('_')[-1]))
    # print(trials_dir/dind_trials[0]/"fitness.csv")
    dind_dfs = [pd.read_csv(trials_dir/dind_trial/"fitness.csv") for dind_trial in dind_trials]
    dind_nps = [df["team_0_fitness"].to_numpy() for df in dind_dfs]
    # print(dind_nps[0])
    return dind_nps

def get_stats(nps, window_size=None):
    # First index is the trial number. Axis 0
    # Second index is the fitness at the generation. Axis 1
    # window_size is for a moving average of specified size

    # Truncate based on shortest trial
    smallest_dim = np.inf
    for n in nps:
        if n.shape[0] < smallest_dim:
            smallest_dim = n.shape[0]

    for i in range(len(nps)):
        nps[i] = nps[i][:smallest_dim]


    arr = np.array(nps)

    avg = np.average(arr, axis=0)
    dev = np.std(arr, axis=0)
    err = dev/np.sqrt(arr.shape[0])

    # Convolution for moving average
    if window_size is not None:
        # Add padding
        pad_len = window_size-1
        avg = np.concatenate([
            np.ones(pad_len)*avg[0],
            avg
        ])
        dev = np.concatenate([
            np.ones(pad_len)*dev[0],
            dev
        ])
        err = np.concatenate([
            np.ones(pad_len)*err[0],
            err
        ])
        # Apply convolution
        avg = np.convolve(avg, np.ones(window_size)/window_size, mode='valid')
        dev = np.convolve(dev, np.ones(window_size)/window_size, mode='valid')
        err = np.convolve(err, np.ones(window_size)/window_size, mode='valid')

    # print(avg)
    return avg, dev, err

def plot_stats(avg, err, color, ax):
    if ax is None:
        h1, = plt.plot(avg, color=color)
        h2 = plt.fill_between(np.arange(avg.shape[0]), avg+err, avg-err, alpha=0.1, color=color)
    else:
        h1, = ax.plot(avg, color=color)
        h2 = ax.fill_between(np.arange(avg.shape[0]), avg+err, avg-err, alpha=0.1, color=color)
    return h1, h2

def process_trials(trials_dir, color, ax, window_size):
    nps = get_nps(trials_dir)
    avg, dev, err = get_stats(nps, window_size)
    return plot_stats(avg, err, color, ax)

def process_experiment(root_dir):
    fig, ax = plt.subplots(1,1)
    dindirect_dir = root_dir/"D-Indirect"
    g_dir = root_dir/"G"
    d_dir = root_dir/"D"
    h1g, h2g = process_trials(dindirect_dir, color='green', ax=ax)
    h1b, h2b = process_trials(g_dir, color='blue', ax=ax)
    h1o, h2o = process_trials(d_dir, color='orange', ax=ax)
    ax.legend([h1g, h1b, h1o], ["New D-Indirect", "G", "D"], loc="lower left")
    title = ".".join(str(root_dir).split("/")[5:])
    ax.set_title(title)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Performance on G")
    ax.set_ylim([0,15])
    ax.set_xlim([0,1000])
    fig.savefig(title+".png")

def process_experiment2(root_dir, window_size):
    fig, ax = plt.subplots(1,1)
    color_map =  ['#006BA4', '#FF800E', '#ABABAB', '#595959', '#5F9ED1', '#C85200', '#898989', '#A2C8EC', '#FFBC79', '#CFCFCF']
    handles = []
    labels = []
    dis_dir = root_dir/"D-Indirect-Step"
    if os.path.exists(dis_dir):
        h1c, h2c = process_trials(dis_dir, color=color_map[1], ax=ax, window_size=window_size)
        handles.append(h1c)
        labels.append("D-Indirect-Step")
    dit_dir = root_dir/"D-Indirect-Traj"
    if os.path.exists(dit_dir):
        h1g, h2g = process_trials(dit_dir, color=color_map[0], ax=ax, window_size=window_size)
        handles.append(h1g)
        labels.append("D-Indirect-Traj")
    g_dir = root_dir/"G"
    if os.path.exists(g_dir):
        h1b, h2b = process_trials(g_dir, color=color_map[2], ax=ax, window_size=window_size)
        handles.append(h1b)
        labels.append("G")
    d_dir = root_dir/"D"
    if os.path.exists(d_dir):
        h1o, h2o = process_trials(d_dir, color=color_map[3], ax=ax, window_size=window_size)
        handles.append(h1o)
        labels.append("D")
    f_dir = root_dir/"Fitness-Critic"
    if os.path.exists(f_dir):
        h1f, h2f = process_trials(f_dir, color=color_map[4], ax=ax, window_size=window_size)
        handles.append(h1f)
        labels.append("Fitness-Critic")
    a_dir = root_dir/"Alignment"
    if os.path.exists(a_dir):
        h1a, h2a = process_trials(a_dir, color=color_map[6], ax=ax, window_size=window_size)
        handles.append(h1a)
        labels.append("Alignment")
    diss_dir = root_dir/"D-Indirect-Step-System-Influence"
    if os.path.exists(diss_dir):
        h1ss, h2ss = process_trials(diss_dir, color=color_map[5], ax=ax, window_size=window_size)
        handles.append(h1ss)
        labels.append("D-Indirect-Step-System-Influence")
    disd_dir = root_dir/"D-Indirect-Step-Difference-Influence"
    if os.path.exists(disd_dir):
        h1sd, h2sd = process_trials(disd_dir, color=color_map[7], ax=ax, window_size=window_size)
        handles.append(h1sd)
        labels.append("D-Indirect-Step-Difference-Influence")
    ax.legend(handles, labels, loc="lower left")
    title = ".".join(str(root_dir).split("/")[5:])
    ax.set_title(title)
    ax.set_xlabel("Generations")
    ax.set_ylabel("Performance on G")
    # ax.set_ylim([0,15])
    # ax.set_xlim([0,1000])
    fig.savefig(title+".png")

def generate_traj_plots(traj_dir):
    """Generate plots of the joint trajectory specified in traj_dir in the same directory as traj_dir"""

    config_dir = Path(traj_dir).parent.parent.parent / 'config.yaml'
    # config_dir = '~/influence-multi-objective/results/simple_a/gens1000/Alignment/config.yaml'
    # traj_dir = "~/influence-multi-objective/example_results/D/trial_0/gen_0/eval_team_0_joint_traj.csv"
    # config_dir = "~/influence-multi-objective/example_results/D/config.yaml"

    save_folder = Path(traj_dir).parent

    df = pd.read_csv(os.path.expanduser(traj_dir))
    c = load_config(os.path.expanduser(config_dir))
    cg = np.array(c['env']['connectivity_grid'])
    n_asvs = len(c['env']['asvs'])
    n_auvs = len(c['env']['auvs'])
    it = df.shape[0]
    xl = len(c['env']['connectivity_grid'])
    yl = len(c['env']['connectivity_grid'][0])

    for i in range(it):
        # Plot a frame
        fig, ax = plt.subplots(1,1)

        # Plot the path of each asv and auv
        for a in range(n_asvs):
            ax.plot(df['asv'+str(a)+'_x'][:i].to_numpy(), df['asv'+str(a)+'_y'][:i].to_numpy(), lw=0.5, color='purple')
        for a in range(n_auvs):
            ax.plot(df['auv'+str(a)+'_x'][:i].to_numpy(), df['auv'+str(a)+'_y'][:i].to_numpy(), lw=0.5, color='orange')

        plot_grid(cg, ax, cmap='tab10_r')

        ax.set_xlim([0, xl])
        ax.set_ylim([0, yl])

        fig.savefig( os.path.expanduser(save_folder / ('frame'+str(i)+'.png') ))
        plt.close()

def generate_final_traj_plots(top_dir):
    """Generate plots of joint trajectories for all final gen
    trajectories found crawling under the top_dir"""

    # Identify config runs
    config_runs = []
    for dirpath, _, filenames in os.walk(os.path.expanduser(top_dir)):
        if 'eval_team_0_joint_traj.csv' in filenames:
            cr = Path(dirpath).parent.parent
            if cr not in config_runs:
                config_runs.append(cr)

    # Get the trials of each config run
    c_dicts = {}
    for cr in config_runs:
        ts = [t for t in os.listdir(cr) if t[:6] == 'trial_']
        ts.sort(key=lambda t : int(t.split('_')[-1]))
        c_dicts[cr] = {'ts':ts}

    # Get final generation of each trial we have traj data for
    for p, d in c_dicts.items():
        c_dicts[p]['gs'] = []
        for t in d['ts']:
            gst = [g for g in os.listdir(p / t) if g[:4] == 'gen_']
            gst.sort(key=lambda g : int(g.split('_')[-1]))
            c_dicts[p]['gs'].append(gst[-1])

    # Now plot the final traj of each trial of each config run
    for cr, d in c_dicts.items():
        for t, g in zip(d['ts'], d['gs']):
            traj_dir = cr / t / g / 'eval_team_0_joint_traj.csv'
            generate_traj_plots(traj_dir)

    return None
