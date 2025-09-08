import motornet as mn
import matplotlib.pyplot as plt
import numpy as np

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
  plt.scatter(target_x, target_y)

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
def plot_episode(batch,n_batch, interval,episode_data):
  xy = episode_data['xy']
  all_targets = episode_data['targets']
  all_hidden = episode_data['hidden']
  all_muscle = episode_data['muscle']
  inp = episode_data['inp']

  if (batch % interval == 0):
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

# At the start and end of each training batch, plot simulation trajectories and deviations from target
  if (batch == 0 or batch == n_batch - 1):
    plot_simulations(xy=xy.detach()[:, :, :2], target_xy=all_targets.detach(), title = f'Batch {batch}')


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

  axs[1].plot(n_batch_lateral, lateral_dev_np, label='Lateral Deviation', color='orange')
  axs[1].set_xlabel("Batch Index")
  axs[1].set_ylabel("Lateral Deviation")
  axs[1].grid(True)

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