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


# Plot hand position, muscle activity, actions, force, neural activity, and their 3 first components for a given episode
def plot_motion(batch,episode_data, title = None, plot_from = 0, plot_to = None):
    xy = episode_data['xy']
    xy_traj = episode_data['traj']
    all_hidden = episode_data['hidden']
    all_muscle = episode_data['muscle']
    all_force = episode_data['force']
    all_actions = episode_data['actions']
    all_targets = episode_data['targets']

    fg, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
    if title is None:
        fg.suptitle(f'Batch {batch}', fontsize=16)
    else:
        fg.suptitle(title, fontsize=16)

    time_frame = episode_data['xy'].shape[1]
    if plot_to is None:
        plot_to = time_frame
    time_vec = np.arange(plot_to)

    ind = 0
    ax[0, 0].plot(time_vec[plot_from:plot_to], np.squeeze(xy.detach().cpu().numpy()[ind, plot_from:plot_to, 0]), label='x')
    ax[0, 0].plot(time_vec[plot_from:plot_to], np.squeeze(xy.detach().cpu().numpy()[ind, plot_from:plot_to, 1]), label='y')
    ax[0, 0].title.set_text('Position')
    ax[0, 0].legend()

    for i in range(all_muscle.shape[2]):
      ax[0, 1].plot(time_vec[plot_from:plot_to], np.squeeze(all_muscle.detach().cpu().numpy()[ind, plot_from:plot_to, i]), label= f'muscle {i+1}')
      ax[0, 2].plot(time_vec[plot_from:plot_to], np.squeeze(all_actions.detach().cpu().numpy()[ind, plot_from:plot_to, i]), label= f'muscle {i+1}')
      ax[1, 0].plot(time_vec[plot_from:plot_to], np.squeeze(all_force.detach().cpu().numpy()[ind, plot_from:plot_to, i]), label= f'muscle {i+1}')

    ax[0, 1].title.set_text('Muscle activity')
    ax[0, 1].legend()

    ax[0, 2].title.set_text('Action')
    ax[0, 2].legend()

    ax[1, 0].title.set_text('Force')
    ax[1, 0].legend()

    ax[1, 1].plot(time_vec[plot_from:plot_to], np.squeeze(all_hidden.detach().cpu().numpy()[ind, plot_from:plot_to, :]))
    ax[1, 1].title.set_text('Neural activity')

    n_components = 3
    pca = PCA(n_components=n_components)
    hidden_reduced = pca.fit_transform(np.squeeze(all_hidden.detach().cpu().numpy()[ind, :, :]))
    for n in range(n_components):
      ax[1, 2].plot(time_vec[plot_from:plot_to], hidden_reduced[plot_from:plot_to, n], label= f'Component {n+1}')
    ax[1, 2].title.set_text('Neural activity (PCA)')
    ax[1, 2].legend()

    plt.show()

    plot_simulations(xy=xy.detach()[:, :, :2], target_xy=all_targets.detach(), title = title)
    # plot_simulations(xy = xy_traj.detach()[:, :, :2], target_xy=all_targets.detach(), title=title)



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



# Plot hidden neural activity subspaces for an all batches across phases, or experiments, networks.
def plot_subspace_batch(all_hidden_reduced, all_labels, all_batch_size, title=None, ax=None, data_color = None, data_label = None, elev = 20, azim = 45, cmaps = None):
    if ax is None:
        fg, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})

    if title is not None:
        ax.set_title(title, fontsize=12)

    if cmaps is None:
        cmaps = [plt.cm.RdPu, plt.cm.Oranges, plt.cm.Reds, plt.cm.Blues, plt.cm.Greens, plt.cm.Purples]

    unique_counts = Counter(all_labels)
    data_phase = sorted(list(unique_counts.keys()))
    color_mapping = {label: index for index, label in enumerate(data_phase)}

    def get_color(dataset_idx, phase_name, batch_index, data_color = None):
        if data_color is None:
            cmap = cmaps[color_mapping[(dataset_idx, phase_name)] % len(cmaps)]
        else:
            cmap = data_color
        phase_shades = np.linspace(0.3, 1, max(unique_counts[(dataset_idx, phase_name)], 1))
        shade = phase_shades[batch_index]
        return cmap(shade)

    offset = 0
    pre_data, pre_phase = None, None
    i_data_phase = 0
    for i, (batch_size, (dataset_idx, phase_name)) in enumerate(zip(all_batch_size, all_labels)):
        if pre_data == dataset_idx and pre_phase == phase_name:
            i_data_phase += 1
        else:
            i_data_phase = 0
        plot_data = all_hidden_reduced[offset: offset + batch_size, :]
        color = get_color(dataset_idx, phase_name, i_data_phase, data_color)
        if data_label is not None:
            label = f'{data_label[dataset_idx]} - {phase_name}'
        else:
            label = f'Dataset {dataset_idx} – {phase_name}'
        ax.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:, 2],
                   marker='o', s=20, color=color,
                   label=label if i_data_phase == (unique_counts[(dataset_idx, phase_name)] // 2) else None)
        offset += batch_size
        pre_data, pre_phase = dataset_idx, phase_name


    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_xlim(-0.8, 0.8)
    ax.set_ylim(-0.8, 0.8)
    ax.set_zlim(-0.8, 0.8)
    ax.view_init(elev=elev, azim=azim)
    ax.legend(loc='upper left', fontsize=12)



# Plot distance matrix (used for neural activity analysis)
def plot_distance_matrix(distance_matrix, ordered_keys, data_label=None, title=None, ax=None):

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    distance_matrix_norm = distance_matrix / distance_matrix.max()
    im = ax.imshow(distance_matrix_norm, cmap='viridis', vmin=0, vmax=1)
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
    ax.set_yticklabels(group_labels, fontsize=10)

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
        ax.set_title(title, fontsize=12)





# The following are helpful for looking at individual network, phase or batch performance
if __name__ == "__main__":
    saveLoc = '/Users/pounemirzazadeh/Motornet/MultiNet/Modular_version/'

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
                    # Training deviations at the batch numbers we have saved to reconstruct during the testing.
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
                    plot_motion(batch = first_batch, episode_data = test_data[phase][first_batch],
                                title=f"Network = {net_id}, Exp = {exp}, Phase = {phase}, Force_Field = {force_field}, batch = {first_batch}", plot_from=30, plot_to=150)
                    plot_motion(batch=last_batch, episode_data=test_data[phase][last_batch],
                                title=f"Network = {net_id}, Exp = {exp}, Phase = {phase}, Force_Field = {force_field}, batch = {last_batch}",  plot_from=30, plot_to=150)
