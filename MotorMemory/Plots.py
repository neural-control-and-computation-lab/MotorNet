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



def plot_motion(batch,episode_data, title = None):
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

    ind = 0
    ax[0, 0].plot(np.squeeze(xy.detach().cpu().numpy()[ind, :, 0]), label='x')
    ax[0, 0].plot(np.squeeze(xy.detach().cpu().numpy()[ind, :, 1]), label='y')
    ax[0, 0].title.set_text('Position')
    ax[0, 0].legend()

    for i in range(all_muscle.shape[2]):
      ax[0, 1].plot(np.squeeze(all_muscle.detach().cpu().numpy()[ind, :, i]), label= f'muscle {i+1}')
      ax[0, 2].plot(np.squeeze(all_actions.detach().cpu().numpy()[ind, :, i]), label= f'muscle {i+1}')
      ax[1, 0].plot(np.squeeze(all_force.detach().cpu().numpy()[ind, :, i]), label= f'muscle {i+1}')

    ax[0, 1].title.set_text('Muscle activity')
    ax[0, 1].legend()

    ax[0, 2].title.set_text('Action')
    ax[0, 2].legend()

    ax[1, 0].title.set_text('Force')
    ax[1, 0].legend()

    ax[1, 1].plot(np.squeeze(all_hidden.detach().cpu().numpy()[ind, :, :]))
    ax[1, 1].title.set_text('Neural activity')

    n_components = 3
    pca = PCA(n_components=n_components)
    hidden_reduced = pca.fit_transform(np.squeeze(all_hidden.detach().cpu().numpy()[ind, :, :]))
    for n in range(n_components):
      ax[1, 2].plot(hidden_reduced[:, n], label= f'Component {n+1}')
    ax[1, 2].title.set_text('Neural activity (PCA)')
    ax[1, 2].legend()

    plt.show()

    plot_simulations(xy=xy.detach()[:, :, :2], target_xy=all_targets.detach(), title = title)



# Plot endpoint and lateral deviation across training batches
def plot_deviation(lateral_dev, endpoint_dev, title = None, batch_values = None):

    endpoint_dev_np = np.array([x.detach().cpu().item() for x in endpoint_dev])
    lateral_dev_np = np.array([x.detach().cpu().item() for x in lateral_dev])

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

    axs[1].plot(n_batch_lateral, lateral_dev_np, label='Lateral Deviation', color='orange')
    axs[1].set_xlabel("Batch Index")
    axs[1].set_ylabel("Lateral Deviation")
    axs[1].grid(True)
    axs[1].set_ylim(-0.06, 0.06)

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

    fg = plt.figure(figsize=(4, 4))
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

def plot_subspace_batch(data, batch_list, t_before_go = 10, title = None):

    all_data_hidden_reduced = {}
    all_hidden = []
    all_batch_size = []
    all_hidden_reduced = []


    for train_batch_number in batch_list:
        data_temp = data[train_batch_number]
        go_signal = data_temp['inp'][:, :, 2]
        is_zero = (go_signal < 0.01)
        indices = th.where(is_zero.any(dim=1), th.argmax(is_zero.int(), dim=1), go_signal.shape[1])
        timepoints = indices - th.full((go_signal.shape[0],), t_before_go, dtype=th.int32)
        batch_size = indices.shape[0]
        all_batch_size.append(batch_size)
        batch_indices = th.arange(batch_size, dtype=th.int32)
        all_hidden.append(data_temp['hidden'][batch_indices, timepoints, :])

    n_components = 3
    pca = PCA(n_components=n_components)
    all_hidden_reduced = pca.fit_transform(np.squeeze(th.cat(all_hidden, dim=0).detach().cpu().numpy()))
    print('Explained Variance = ', pca.explained_variance_ratio_)

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(batch_list)))

    fg, ax = plt.subplots(figsize=(6, 6), subplot_kw={'projection': '3d'})
    if title is not None:
        fg.suptitle(title, fontsize=8)




    for i in range(len(batch_list)):
        plot_data = all_hidden_reduced[i * all_batch_size[i]:(i + 1) * all_batch_size[i], :]
        all_data_hidden_reduced[f'training_batch_{batch_list[i]}'] = plot_data
        ax.scatter(plot_data[:, 0], plot_data[:, 1], plot_data[:,2], marker='o', s=20, color=colors[i],label=f'Batch {batch_list[i]}')


    ax.set_xlabel('Component 1')
    ax.set_ylabel('Component 2')
    ax.set_zlabel('Component 3')
    ax.set_xlim(-0.5, 0.5)
    ax.set_ylim(-0.5, 0.5)
    ax.set_zlim(-0.5, 0.5)
    #ax.legend(loc='upper left')
    plt.show()



if __name__ == "__main__":
    saveLoc = '/Users/pounemirzazadeh/Motornet/MultiNet/Modular_version'

    exp_names = ['center_out', 'mov_diff_loc', 'pro_loc', 'vis_loc']
    phases = ['NF1', 'FF1', 'NF2']
    force_fields = ['CW']


    num_nets = 4



    for net_id in range(num_nets):
        for exp in exp_names:
            path_result = os.path.join(saveLoc, f"task_{net_id}", f"results_{exp}")
            print(path_result)
            results = th.load(path_result)
            for phase, data in results.items():
                plot_deviation(data['lateral_dev'], data['endpoint_dev'], title=f"Training Deviation, Network = {net_id}, Exp = {exp}, Phase = {phase}")
                plot_training_loss(data['losses'], title=f"Loss, Network = {net_id}, Exp = {exp}, Phase = {phase}")
            for force_field in force_fields:
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
                    batch_list = sorted(list(dev_data[phase].keys()))
                    plot_subspace_ind(data = test_data[phase][last_batch], t_before_go=10, title=f"Activity, Network = {net_id}, Exp = {exp}, Phase = {phase}, Force_Field = {force_field}")
                    plot_subspace_batch(data=test_data[phase], batch_list = list(test_data[phase]), t_before_go=10,
                                title=f"Activity, Network = {net_id}, Exp = {exp}, Phase = {phase}, Force_Field = {force_field}")
                    plot_deviation(lateral_tensor, endpoint_tensor, title=f"Test Deviation, Network = {net_id}, Exp = {exp}, Phase = {phase}", batch_values=batch_list)
                    plot_motion(batch = first_batch, episode_data = test_data[phase][first_batch],
                                title=f"Network = {net_id}, Exp = {exp}, Phase = {phase}, Force_Field = {force_field}, batch = {first_batch}")
                    plot_motion(batch=last_batch, episode_data=test_data[phase][last_batch],
                                title=f"Network = {net_id}, Exp = {exp}, Phase = {phase}, Force_Field = {force_field}, batch = {last_batch}")
