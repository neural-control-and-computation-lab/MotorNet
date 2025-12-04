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

  fg = plt.figure(figsize=(10,4))
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
def plot_episode(task, batch,n_batch, interval,episode_data):
  xy = episode_data['xy']
  obs = episode_data['vis_inp']
  fingertip = episode_data['xy']
  xy = task.shift_traj(xy)
  all_targets = episode_data['targets']
  all_hidden = episode_data['hidden']
  all_muscle = episode_data['muscle']
  inp = episode_data['inp']


  # if (batch % interval == 0):
  if (batch == 0 or batch == n_batch - 1):
    fg, ax = plt.subplots(nrows=2, ncols=2, figsize=(10, 7))
    fg.suptitle(f'Batch {batch}', fontsize=16)
    ind = 0
    ax[0, 0].plot(np.squeeze(all_hidden.detach().cpu().numpy()[ind, :, :]))
    ax[0, 0].title.set_text('Neural activity')
    ax[0, 1].plot(np.squeeze(all_muscle.detach().cpu().numpy()[ind, :, :]))
    ax[0, 1].title.set_text('Muscle activity')
    ax[1, 0].plot(np.squeeze(inp.detach().cpu().numpy()[ind, :, :]))
    ax[1, 0].title.set_text('Inputs')
    ax[1, 1].plot(
      np.concatenate((all_targets.detach().cpu().numpy()[ind, :, 0:2], xy.detach().cpu().numpy()[ind, :, 0:2]),
                     axis=1))
    ax[1, 1].title.set_text('Targets and outputs')
    plt.show()
    plot_simulations(xy=xy.detach()[:, :, :2], target_xy=all_targets.detach(), title=f'Batch {batch}')
    plot_motion(task = task, batch=batch, episode_data=episode_data)



def plot_motion(task, batch,episode_data):
  xy = episode_data['xy']
  xy = task.shift_traj(xy)
  all_hidden = episode_data['hidden']
  all_muscle = episode_data['muscle']
  all_force = episode_data['force']
  all_actions = episode_data['actions']
  fg, ax = plt.subplots(nrows=2, ncols=3, figsize=(10, 7))
  fg.suptitle(f'Batch {batch}', fontsize=16)
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


# Plot endpoint and lateral deviation across training batches
def plot_deviation(lateral_dev, endpoint_dev, title = None):

  endpoint_dev_np = np.array([x.detach().cpu().item() for x in endpoint_dev])
  lateral_dev_np = np.array([x.detach().cpu().item() for x in lateral_dev])

  n_batch_endpoint = np.arange(len(endpoint_dev_np))
  n_batch_lateral = np.arange(len(lateral_dev_np))

  fig, axs = plt.subplots(2, 1, figsize=(10, 8))

  axs[0].plot(n_batch_endpoint, endpoint_dev_np, label='Endpoint Deviation', color='blue')
  axs[0].set_ylabel("Endpoint Deviation")
  axs[0].set_title("Deviation Across Batches")
  axs[0].grid(True)
  # axs[0].set_ylim(0, 0.4)

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


def plot_subspace(data, t_before_go = 10, title = None):

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
    fg.suptitle(title, fontsize=12)

  plt.scatter(hidden_reduced[:, 0],hidden_reduced[:, 1], marker='o', s=30, color='C0')

  n_targets = 8 #Should change the code so I can have access to the value in task
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
