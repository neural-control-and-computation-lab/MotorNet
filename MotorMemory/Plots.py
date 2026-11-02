import os
import sys

# Add project root to Python path for imports
this_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(this_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


import motornet as mn
import matplotlib.pyplot as plt
import numpy as np
import torch as th
from sklearn.decomposition import PCA
from collections import Counter
from scipy.stats import gaussian_kde
from matplotlib.lines import Line2D


plotor = mn.plotor.plot_pos_over_time

#  Plot arm trajectories and errors relative to the target
def plot_simulations(xy, target_xy, title = None):
    target_x = target_xy[:, -1, 0]
    target_y = target_xy[:, -1, 1]

    fg = plt.figure(figsize=(12,4))
    if title is not None:
      fg.suptitle(title, fontsize=16)

    # trajectory in workspace
    plt.subplot(1,2,1)
    plt.ylim([-0.3, 1])
    plt.xlim([-0.7, 0.7])
    plotor(axis=plt.gca(), cart_results=xy)
    plt.scatter(target_x, target_y, s = 10)

    # deviation from target
    plt.subplot(1,2,2)
    plt.ylim([-0.5, 0.5])
    plt.xlim([-0.5, 0.5])
    plotor(axis=plt.gca(), cart_results=xy - target_xy)
    plt.axhline(0, c="grey")
    plt.axvline(0, c="grey")
    plt.xlabel("X distance to target")
    plt.ylabel("Y distance to target")
    plt.show()



# Plot neural activity, muscle activity, inputs, targets and trajectory for a given episode
def plot_episode(batch,n_batch, interval,episode_data, title = None):
    xy = episode_data['xy']
    obs = episode_data['vis_inp']
    fingertip = episode_data['xy']
    xy_traj = episode_data['traj']
    all_targets = episode_data['targets']
    all_hidden = episode_data['hidden']
    all_muscle = episode_data['muscle']
    inp = episode_data['inp']

    if (batch == 0 or batch == n_batch - 1):
      if title is None:
        suptitle = f'Batch {batch}'
      else:
        suptitle = title
      fg, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
      fg.suptitle(suptitle, fontsize=16)
      ind = 0
      ax[0, 0].plot(np.squeeze(all_hidden.detach().cpu().numpy()[ind, :, :]))
      ax[0, 0].title.set_text('Neural activity')
      ax[0, 1].plot(np.squeeze(all_muscle.detach().cpu().numpy()[ind, :, :]))
      ax[0, 1].title.set_text('Muscle activity')
      ax[1, 0].plot(np.squeeze(inp.detach().cpu().numpy()[ind, :, :]))
      ax[1, 0].title.set_text('Inputs')
      ax[1, 1].plot(np.concatenate((all_targets.detach().cpu().numpy()[ind, :, 0:2],
                                    xy.detach().cpu().numpy()[ind, :, 0:2]),axis=1))
      ax[1, 1].title.set_text('Targets and outputs')
      plt.show()
      plot_simulations(xy=xy.detach()[:, :, :2], target_xy=all_targets.detach(), title=suptitle)


def plot_episode_new(batch,episode_data, title = None):
    xy = episode_data['xy']
    obs = episode_data['vis_inp']
    fingertip = episode_data['xy']
    xy_traj = episode_data['traj']
    all_targets = episode_data['targets']
    all_hidden = episode_data['hidden']
    all_muscle = episode_data['muscle']
    inp = episode_data['inp']

    mm = mn.muscle.RigidTendonHillMuscle()
    ee = mn.effector.RigidTendonArm26(muscle=mm, timestep=0.01)
    base_joint = np.deg2rad([40., 110., 0., 0.]).astype(np.float32)
    center_position = ee.joint2cartesian(th.tensor(base_joint))
    radius = 0.1
    [x_min, y_min] = center_position[0, :2].detach().numpy()
    [x_max, y_max] = [x_min + radius, y_min + radius]

    fg1, ax1 = plt.subplots(nrows=3, ncols=1, figsize=(4,6))
    fg1.suptitle('Input', fontsize=16)
    ind = 0
    to_plot = 150
    time_vec = np.arange(to_plot)
    batch_size = inp.size(0)
    starting_point = np.repeat(xy[ind, 0, :].detach().cpu().numpy()[None, :], to_plot, axis=0)
    ax1[0].plot(time_vec/100,(starting_point[:, 0] - x_min) , color = 'orange', label = 'x')
    ax1[0].plot(time_vec/100, (starting_point[:, 1] - y_min) , color = 'blue', label = 'y')
    ax1[0].legend(['x','y'], loc='upper right', fontsize=14, frameon=False)
    # ax1[0].set_title('Inputs', fontsize=12)
    ax1[0].set_ylabel('Start [m]', fontsize=14)
    ax1[0].spines["top"].set_visible(False)
    ax1[0].spines["right"].set_visible(False)
    ax1[0].set_ylim([-0.05,0.15])
    ax1[0].set_xticks([])
    ax1[0].set_yticks([0, 0.1])
    ax1[0].tick_params(axis='y', labelsize=14)

    target = np.repeat(all_targets[ind, -1, :].detach().cpu().numpy()[None, :], to_plot, axis=0)
    ax1[1].plot(time_vec/100, (target[:,0] - x_min) , color = 'orange')
    ax1[1].plot(time_vec/100, (target[:,1] - y_min) , color='blue')
    ax1[1].set_ylabel('Target [m]', fontsize=14)
    # ax1[1].legend(['x','y'], loc='upper right', fontsize=10, frameon=False)
    ax1[1].spines["top"].set_visible(False)
    ax1[1].spines["right"].set_visible(False)
    ax1[1].set_ylim([-0.05,0.12])
    ax1[1].set_xticks([])
    ax1[1].set_yticks([0, 0.1])
    ax1[1].tick_params(axis='y', labelsize=14)

    ax1[2].plot(time_vec/100, np.squeeze(inp.detach().cpu().numpy()[ind, :to_plot, 2]), color='green')
    ax1[2].set_ylabel('Go Cue', fontsize=14)
    ax1[2].set_xlabel('Time [s]', fontsize=14)
    ax1[2].spines["top"].set_visible(False)
    ax1[2].spines["right"].set_visible(False)
    ax1[2].set_ylim([-0.05, 1.05])
    ax1[2].set_yticks([0,  1.0])
    ax1[2].set_xticks([0, 1.5])
    ax1[2].tick_params(axis='both', labelsize=14)

    fg2, ax2 = plt.subplots(nrows=2, ncols=1, figsize=(4, 4))
    fg2.suptitle('Output', fontsize=16)

    ax2[0].plot(time_vec/100, (np.squeeze(xy.detach().cpu().numpy()[ind, :to_plot, 0]) - x_min) , color='orange', zorder=1)
    ax2[0].plot(time_vec/100, (np.squeeze(xy.detach().cpu().numpy()[ind, :to_plot, 1]) - y_min) , color='blue', zorder=1)
    ax2[0].plot(time_vec/100, (np.squeeze(all_targets.detach().cpu().numpy()[ind, :to_plot, 0]) - x_min) , color='orange', linestyle='--', zorder=0)
    ax2[0].plot(time_vec/100, (np.squeeze(all_targets.detach().cpu().numpy()[ind, :to_plot, 1]) - y_min) , color='blue', linestyle='--', zorder= 0)
    ax2[0].set_ylabel('Hand Pos [m]', fontsize=14)
    # ax2[0].legend(['x', 'y'], loc='upper right', fontsize=10, frameon=False)
    ax2[0].spines["top"].set_visible(False)
    ax2[0].spines["right"].set_visible(False)
    ax2[0].set_ylim([-0.05, 0.2])
    ax2[0].set_xticks([])
    ax2[0].text(55, 0.12, "goal", fontsize=10, ha="left", va="center")
    ax2[0].set_yticks([0, 0.1])
    ax2[0].tick_params(axis='y', labelsize=14)

    ax2[1].plot(time_vec/100, np.squeeze(xy.detach().cpu().numpy()[ind, :to_plot, 2]), color='orange')
    ax2[1].plot(time_vec/100, np.squeeze(xy.detach().cpu().numpy()[ind, :to_plot, 3]), color='blue')
    ax2[1].set_ylabel('Vel [m/s]', fontsize=14)
    # ax2[1].legend(['x', 'y'], loc='upper right', fontsize=10, frameon=False)
    ax2[1].spines["top"].set_visible(False)
    ax2[1].spines["right"].set_visible(False)
    ax2[1].set_xlabel('Time [s]', fontsize=14)
    # ax2[1].set_ylim([0, 0.6])
    ax2[1].set_yticks([0, 0.6])
    ax2[1].set_xticks([0, 1.5])
    ax2[1].tick_params(axis='both', labelsize=14)

    plt.show()


# Plot hand position, muscle activity, actions, force, neural activity, and their 3 first components for a given episode
def plot_motion(batch,episode_data, title = None, plot_from = 0, plot_to = None):
    xy = episode_data['xy']
    xy_traj = episode_data['traj']
    all_hidden = episode_data['hidden']
    all_muscle = episode_data['muscle']
    all_force = episode_data['force']
    all_actions = episode_data['actions']
    all_targets = episode_data['targets']

    fg, ax = plt.subplots(figsize=(5, 5))

    time_frame = episode_data['xy'].shape[1]
    if plot_to is None:
        plot_to = time_frame
    time_vec = np.arange(plot_to)

    ind = 0
    muscle_name = ['SF', 'SE', 'EF', 'EE', 'BF', 'BE']
    for i in range(all_muscle.shape[2]):
      ax.plot(time_vec[plot_from:plot_to]/100, np.squeeze(all_muscle.detach().cpu().numpy()[ind, plot_from:plot_to, i]), label= muscle_name[i])
    # ax.set_ylabel('Muscle activity', fontsize=16)
    ax.set_ylabel('Muscle Activity [a.u]', fontsize=14)
    ax.set_xlabel('Time [s]', fontsize=14)

    ax.legend(fontsize=14, frameon=False, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(axis='both', labelsize=14)
    ax.set_yticks([0, 0.4, 0.8, 0.12])
    ax.set_xticks([0, 1.5])
    # ax.set_xlim([plot_from, plot_to])
    ax.set_ylim([-0.01, 0.13])

    # plt.show()

    # plot_simulations(xy=xy.detach()[:, :, :2], target_xy=all_targets.detach(), title = title)
    # plot_simulations(xy = xy_traj.detach()[:, :, :2], target_xy=all_targets.detach(), title=title)
    plot_episode_new(batch, episode_data, title=None)


# Plot endpoint and lateral deviation across training or test batches
def plot_deviation(lateral_dev, endpoint_dev, title = None, batch_values = None):

    endpoint_dev_np = np.array([x.detach().cpu().item() for x in endpoint_dev])
    lateral_dev_np = np.array([x.detach().cpu().item() for x in lateral_dev])
    # lateral_dev_np_signed = -lateral_dev_np if invert_vals else lateral_dev_np

    if batch_values is not None:
        n_batch_endpoint = np.array(batch_values)
        n_batch_lateral = np.array(batch_values)
    else:
        n_batch_endpoint = np.arange(len(endpoint_dev_np))
        n_batch_lateral = np.arange(len(lateral_dev_np))

    fig, axs = plt.subplots(2, 1, figsize=(10, 8))

    axs[0].plot(n_batch_endpoint, endpoint_dev_np, label='Endpoint Deviation', color='blue')
    axs[0].set_ylabel("Endpoint Deviation")
    axs[0].set_title("Deviation Across Batches")
    axs[0].grid(True)
    axs[0].set_ylim(-0.01, 0.1)

    axs[1].plot(n_batch_lateral, lateral_dev_np, label='Lateral Deviation', color='orange')
    axs[1].set_xlabel("Batch Index")
    axs[1].set_ylabel("Lateral Deviation")
    axs[1].grid(True)
    axs[1].set_ylim(-0.06, 0.06)
    # axs[1].set_ylim(-0.1, 0.1)

    if title:
      plt.suptitle(title)

    plt.show()



# Plot total and component losses across training batches
def plot_training_loss(losses, title = None):

    fig, axs = plt.subplots(2, 1)
    fig.set_tight_layout(True)
    fig.set_size_inches((8, 6))
    # Plot total loss
    axs[0].plot(losses['total'], label='total')
    axs[0].set_ylabel("Total Loss")
    axs[0].set_xlabel("Iteration #")
    axs[0].legend()
    axs[0].grid(True)

    # Plot all other losses
    for key in losses:
      if key != 'total':
        axs[1].plot(losses[key], label=key)

    axs[1].set_ylabel("Losses Components")
    axs[1].set_xlabel("Iteration #")
    axs[1].legend()
    axs[1].grid(True)

    if title:
      plt.suptitle(title)

    plt.show()


# Plot hidden neural activity subspaces for an individual batch for a phase, experiment, network.
def plot_subspace_ind(data, t_before_go = 10, title = None):

    go_signal = data['inp'][:, :, 2]
    is_zero = (go_signal < 0.01)
    indices = th.where(is_zero.any(dim=1), th.argmax(is_zero.int(), dim=1), go_signal.shape[1])
    timepoints = indices - th.full((go_signal.shape[0],), t_before_go, dtype=th.int32)
    batch_indices = th.arange(indices.shape[0], dtype=th.int32)

    hidden = data['hidden'][batch_indices, timepoints, :]
    n_components = 2
    pca = PCA(n_components=n_components)
    hidden_reduced = pca.fit_transform(np.squeeze(hidden.detach().cpu().numpy()))

    fg = plt.figure(figsize=(5, 4))
    if title is not None:
      fg.suptitle(title, fontsize=8)

    plt.scatter(hidden_reduced[:, 0],hidden_reduced[:, 1], marker='o', s=30, color='C0')

    n_targets = 8
    for i in range(n_targets):
      plt.annotate(
        str(i + 1),
        (hidden_reduced[i, 0], hidden_reduced[i, 1]),
        textcoords="offset points",
        xytext=(0, 5),
        ha='center',
        fontsize=10,
        fontweight='bold',
        color='black'
      )
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.xlim(-0.5, 0.5)
    plt.ylim(-0.5, 0.5)
    plt.show()



# Plot hidden neural activity subspaces for all batches across phases, or experiments, networks.

def _plot_kde_projection(ax, x, y, zdir, offset, color, xlim, ylim,
                          levels=6, thresh=0.15, alpha=0.3, grid_size=60):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 3:
        return  # not enough points for a stable KDE

    try:
        kde = gaussian_kde(np.vstack([x, y]))
    except np.linalg.LinAlgError:
        return  # degenerate/collinear points

    xx, yy = np.meshgrid(
        np.linspace(xlim[0], xlim[1], grid_size),
        np.linspace(ylim[0], ylim[1], grid_size)
    )
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)

    zz_norm = zz / zz.max()

    # thresh cuts off the lowest-density contour band, like sns.kdeplot(thresh=...)
    contour_levels = np.linspace(thresh, 1, levels)

    ax.contourf(xx, yy, zz_norm, levels=contour_levels, zdir=zdir, offset=offset,
                colors=[color], alpha=alpha)


def _lighten_color(color, amount=0.5):
    r, g, b, a = color
    r = r + (1 - r) * amount
    g = g + (1 - g) * amount
    b = b + (1 - b) * amount
    return (r, g, b, a)


def plot_subspace_batch(all_hidden_reduced, all_labels, all_batch_size, title=None, ax=None,
                         data_label=None, elev=20, azim=45, cmaps=None,
                         show_projections=True, projection_alpha=0.15, projection_lighten=0.5,
                         show_legend=True, explained_variance=None):
    if ax is None:
        fg, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})

    if title is not None:
        ax.set_title(title, fontsize=20)

    if cmaps is None:
        cmaps = [plt.cm.Greens, plt.cm.Reds]

    xlim = (-0.8, 0.8)
    ylim = (-0.8, 0.8)
    zlim = (-0.8, 0.8)
                             
    dataset_totals = Counter(dataset_idx for dataset_idx, _ in all_labels)
    shade_lookup = {
        dataset_idx: np.linspace(0.3, 1, max(count, 1))
        for dataset_idx, count in dataset_totals.items()
    }
    running_idx = {dataset_idx: 0 for dataset_idx in dataset_totals}

    phase_shades = {}
    offset = 0

    for batch_size, (dataset_idx, phase_name) in zip(all_batch_size, all_labels):
        seq_idx = running_idx[dataset_idx]
        shade = shade_lookup[dataset_idx][seq_idx]
        running_idx[dataset_idx] = seq_idx + 1

        color = cmaps[dataset_idx](shade)
        proj_color = _lighten_color(color, amount=projection_lighten)
        phase_shades.setdefault((dataset_idx, phase_name), []).append(shade)
        plot_data = all_hidden_reduced[offset: offset + batch_size, :]
        
        ax.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2],
                   marker='o', s=20, color=color)

        if show_projections:
            # Floor
            ax.scatter(plot_data[:, 0], plot_data[:, 1], zlim[0],
                       marker='o', s=20, color=proj_color, alpha=projection_alpha, zdir='z')
            # Back wall
            ax.scatter(plot_data[:, 0], plot_data[:, 2], ylim[0],
                       marker='o', s=20, color=proj_color, alpha=projection_alpha, zdir='y')
            # Side wall
            ax.scatter(plot_data[:, 1], plot_data[:, 2], xlim[0],
                       marker='o', s=20, color=proj_color, alpha=projection_alpha, zdir='x')

        offset += batch_size

    if show_legend:
        legend_handles = []
        seen_order = list(dict.fromkeys(all_labels))  # preserve first-seen order
        for dataset_idx, phase_name in seen_order:
            avg_shade = np.mean(phase_shades[(dataset_idx, phase_name)])
            color = cmaps[dataset_idx](avg_shade)
            if data_label is not None:
                label = f'{data_label[dataset_idx]} - {phase_name}'
            else:
                label = f'Dataset {dataset_idx} - {phase_name}'
            legend_handles.append(Line2D([0], [0], marker='o', color='w',
                                          markerfacecolor=color, markersize=8, label=label))
        ax.legend(handles=legend_handles, loc='upper left', fontsize=15)

    if explained_variance is not None:
        ax.set_xlabel(f'Comp. 1 ({explained_variance[0] * 100:.1f}%)', fontsize=15)
        ax.set_ylabel(f'Comp. 2 ({explained_variance[1] * 100:.1f}%)', fontsize=15)
        ax.set_zlabel(f'Comp. 3 ({explained_variance[2] * 100:.1f}%)', fontsize=15)
    else:
        ax.set_xlabel('Component 1', fontsize=15)
        ax.set_ylabel('Component 2', fontsize=15)
        ax.set_zlabel('Component 3', fontsize=15)
    
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.view_init(elev=elev, azim=azim)    




# Plot distance matrix (used for neural activity analysis)
def plot_distance_matrix(distance_matrix, ordered_keys, data_label=None, title=None, ax=None, colorbar_show=True, max_value = None, ylabel_flag = True):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    if max_value is None:
        distance_matrix_norm = distance_matrix / distance_matrix.max()
    else:
        distance_matrix_norm = distance_matrix / max_value

    im = ax.imshow(distance_matrix_norm, cmap='viridis', vmin=0, vmax=1)
    
    if colorbar_show:
        ax.figure.colorbar(im, ax=ax)


    groups = {}
    for i, (ds, phase, b) in enumerate(ordered_keys):
        key = (ds, phase)
        if key not in groups:
            groups[key] = []
        groups[key].append(i)

    group_centers = []
    group_labels = []
    for (ds, phase), indices in groups.items():
        center = np.mean(indices)
        label = f'{ds} - {phase}'
        group_centers.append(center)
        group_labels.append(label)

    ax.set_xticks(group_centers)
    ax.set_yticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=10, rotation=90)
    if ylabel_flag:
        ax.set_yticklabels(group_labels, fontsize=10)
    else:
        ax.set_yticklabels([])

    phase_boundaries = []
    current_phase = None
    for i, (_, phase, _) in enumerate(ordered_keys):
        if phase != current_phase:
            if current_phase is not None:
                phase_boundaries.append(i)
            current_phase = phase

    for boundary in phase_boundaries:
        ax.axhline(boundary - 0.5, color='white', linewidth=1.5)
        ax.axvline(boundary - 0.5, color='white', linewidth=1.5)

    if title:
        ax.set_title(title, fontsize=14)

    return im

        ax.set_title(title, fontsize=14)

    return im



def plot_ridge_matrix(score_matrix, ordered_keys, title=None, ax=None,
                       colorbar_show=True, vmin=None, vmax=None, cmap='viridis'):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    im = ax.imshow(score_matrix, cmap='RdBu', vmin=vmin, vmax=vmax)
    if colorbar_show:
        ax.figure.colorbar(im, ax=ax)

    # group by phase (analogous to grouping by (ds, phase) in the original)
    groups = {}
    for i, (phase, batch) in enumerate(ordered_keys):
        groups.setdefault(phase, []).append(i)

    group_centers = []
    group_labels = []
    for phase, indices in groups.items():
        group_centers.append(np.mean(indices))
        group_labels.append(phase)

    ax.set_xticks(group_centers)
    ax.set_yticks(group_centers)
    ax.set_xticklabels(group_labels, fontsize=10)
    ax.set_yticklabels(group_labels, fontsize=10)

    phase_boundaries = []
    current_phase = None
    for i, (phase, _) in enumerate(ordered_keys):
        if phase != current_phase:
            if current_phase is not None:
                phase_boundaries.append(i)
            current_phase = phase

    for boundary in phase_boundaries:
        ax.axhline(boundary - 0.5, color='white', linewidth=1.5)
        ax.axvline(boundary - 0.5, color='white', linewidth=1.5)

    if title:
        ax.set_title(title, fontsize=20)

    return im


# Plot neural activity as a heatmap: y = neurons, x = time
def plot_neural_heatmap(hidden, title=None, batch_ind=0, ax=None, cmap='viridis'):
    to_plot = 150
    hidden_np = np.squeeze(hidden.detach().cpu().numpy()[batch_ind, :to_plot, :])  # (time, neurons)

    if ax is None:
        fig, ax = plt.subplots(figsize=(5,5))

    im = ax.imshow(hidden_np.T, aspect='auto', origin='lower', cmap=cmap, extent=[0, hidden_np.shape[0]/100, 0, hidden_np.shape[1]])

    ax.set_xlabel('Time [s]', fontsize=14)
    ax.set_ylabel('Neuron', fontsize=14)
    ax.set_xticks([0,1.5])
    ax.set_yticks([0,250])
    ax.tick_params(axis='both', labelsize=14)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.set_label('Activity [a.u.]', fontsize=14)
    cbar.ax.tick_params(labelsize=14)

    if title:
        ax.set_title(title, fontsize=16)

    plt.show()
    return im



# The following are helpful for looking at individual network, phase or batch performance
if __name__ == "__main__":
    saveLoc = '/Users/pounemirzazadeh/Motornet/MultiNet/Modular_version'

    exp_names = ['center_out']      # 'center_out', 'mov_diff_loc', 'pro_loc', 'vis_loc'
    phases = ['NF1', 'FF1', 'NF2']
    force_fields = ['CW']           # 'CW', 'CCW'
    num_nets = 1


    for net_id in range(num_nets):
        for exp in exp_names:
            # Training results
            path_result = os.path.join(saveLoc, f"task_{net_id}", f"results_{exp}")
            print(path_result)
            results = th.load(path_result)
            for phase, data in results.items():
                 plot_deviation(data['lateral_dev'], data['endpoint_dev'], title=f"Training Deviation, Network = {net_id}, Exp = {exp}, Phase = {phase}")
                 plot_training_loss(data['losses'], title=f"Loss, Network = {net_id}, Exp = {exp}, Phase = {phase}")
            for force_field in force_fields:
                # Test results
                path_test = os.path.join(saveLoc, f"task_{net_id}", f"test_data_{exp}_{force_field}")
                path_dev = os.path.join(saveLoc, f"task_{net_id}", f"deviations_{exp}_{force_field}")
                test_data = th.load(path_test)
                dev_data = th.load(path_dev)
                for phase in phases:
                    last_batch = list(test_data[phase])[-1]
                    first_batch = list(test_data[phase])[0]
                    lateral_list = [th.tensor(dev_data[phase][b]['lateral']) for b in sorted(dev_data[phase].keys())]
                    endpoint_list = [th.tensor(dev_data[phase][b]['endpoint']) for b in sorted(dev_data[phase].keys())]
                    lateral_tensor = th.cat(lateral_list).flatten()
                    endpoint_tensor = th.cat(endpoint_list).flatten()
                    batch_list = sorted(dev_data[phase].keys(), key=lambda x: int(x))
                    batch_list_ints = [int(b) for b in batch_list]
                    # Training deviations at the batch numbers we have saved to reconstruct during testing.
                    train_lateral_all = results[phase]['lateral_dev']
                    train_endpoint_all = results[phase]['endpoint_dev']
                    train_lateral_at_checkpoints = []
                    train_endpoint_at_checkpoints = []
                    for b in batch_list_ints:
                        train_lateral_at_checkpoints.append(train_lateral_all[b])
                        train_endpoint_at_checkpoints.append(train_endpoint_all[b])
                    plot_subspace_ind(data = test_data[phase][last_batch], t_before_go=10, title=f"Activity, Network = {net_id}, Exp = {exp}, Phase = {phase}, Force_Field = {force_field}")
                    plot_deviation(lateral_tensor, endpoint_tensor, title=f"Test Deviation, Network = {net_id}, Exp = {exp}, Phase = {phase}", batch_values=batch_list)
                    plot_deviation(train_lateral_at_checkpoints, train_endpoint_at_checkpoints, title=f"Train Deviation, Network = {net_id}, Exp = {exp}, Phase = {phase}", batch_values=batch_list)
                    plot_motion(batch=first_batch, episode_data=test_data[phase][first_batch],
                                title=f"{phase}, first batch", plot_from=0, plot_to=150)
                    plot_motion(batch=last_batch, episode_data=test_data[phase][last_batch],
                                title=f"{phase}, last batch", plot_from=0, plot_to=150)
                    plot_neural_heatmap(test_data[phase][last_batch]['hidden'], title=f"Hidden Neural Activity")
