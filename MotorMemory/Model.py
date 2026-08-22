import torch as th
import numpy as np
from Plots import plot_episode
from utils import applied_load, movement_phase_mask
import copy

# Run a single episode
def run_episode(env, task, policy, batch_size, n_t, device, force_field = 'null', contextual_cue = 'null', k = 0, *args, **kwargs ):
    if force_field == 'random':
        force_dir = np.random.choice(['CW', 'CCW'])
    elif force_field == 'CW':
        force_dir = 'CW'
    elif force_field == 'CCW':
        force_dir = 'CCW'
    else:
        force_dir = None

    if contextual_cue == 'null':
        task.cue = 'null'
    elif contextual_cue == 'random':
        task.cue = np.random.choice(['right', 'left'])
    elif contextual_cue == 'right':
        task.cue = 'right'
    elif contextual_cue == 'left':
        task.cue = 'left'
    elif contextual_cue == 'force_dependent':
        if force_dir == 'CW':
            task.cue = 'right'
        elif force_dir == 'CCW':
            task.cue = 'left'
    else:
        task.cue = 'null'

    inputs, targets, init_states = task.generate(batch_size, n_t, *args, **kwargs)
    targets = th.tensor(targets[:, :, 0:2], device=device, dtype=th.float)
    inp = th.tensor(inputs, device=device, dtype=th.float)
    init_states = th.tensor(init_states, device=device, dtype=th.float)

    h = policy.init_hidden(batch_size)
    obs, info = env.reset(options={'batch_size': batch_size, 'joint_state': init_states})
    terminated = False

    # initialize things we want to keep track of
    xy = []
    all_actions = []
    all_muscle = []
    all_hidden = []
    all_force = []
    all_targets = []
    all_inp = []
    all_vis_inp = []
    all_joint = []



    while not terminated:  # will run until `max_ep_duration` is reached
        t_step = int(env.elapsed / env.dt)
        obs = task.shift_obs(obs) # This line added for Exp 8
        obs = th.concat((obs, inp[:, t_step, :]), dim=1)
        action, h = policy(obs, h)

        # The task's go cue is noisy: approximately 1 while holding and 0 after go.
        # Recover that discrete phase with a midpoint threshold so the curl field is
        # continuously on during movement instead of flickering with input noise.
        force_mask = movement_phase_mask(inp[:, t_step, 2])
        applied_force  = applied_load(endpoint_vel = info['states']['cartesian'][:, 2:], k = k, mode = force_dir)
        masked_force_field = applied_force * force_mask

        obs, reward, terminated, truncated, info = env.step(action=action, endpoint_load = masked_force_field)
        xy.append(info['states']['cartesian'][:, None, :])
        all_actions.append(action[:, None, :])
        all_muscle.append(info['states']['muscle'][:, 0, None, :])
        all_force.append(info['states']['muscle'][:, -1, None, :])
        all_hidden.append(h[:, None, :])
        all_targets.append(th.unsqueeze(targets[:, t_step, :], dim=1))
        all_inp.append(th.unsqueeze(inp[:, t_step, :], dim=1))
        all_vis_inp.append(obs[:, None, :2])
        all_joint.append(info['states']['joint'][:, None, :])

    ext_force = force_dir


    return {
        'xy': th.cat(xy, dim=1),
        'hidden' : th.cat(all_hidden, dim=1),
        'actions' : th.cat(all_actions, dim=1),
        'muscle' : th.cat(all_muscle, dim=1),
        'force' : th.cat(all_force, dim=1),
        'targets' : th.cat(all_targets, dim=1),
        'inp' : th.cat(all_inp, dim=1),
        'vis_inp' : th.cat(all_vis_inp, dim=1),
        'joint' : th.cat(all_joint, dim=1),
        'ext_force' : ext_force,
        'traj' : task.shift_traj(th.cat(xy, dim=1)),
    }


# Compute losses for a single episode.
def calculate_loss(task,episode_data):
    xy = episode_data['xy']
    xy = task.shift_traj(xy)
    all_targets = episode_data['targets']
    all_hidden = episode_data['hidden']
    all_force = episode_data['force']

    cartesian_loss = 1e3 * th.mean(th.sum(th.abs(xy[:, :, :2] - all_targets), dim=-1))
    muscle_loss = 1e-1 * th.mean(th.sum(all_force, dim=-1))
    spectral_loss = 1e4 * th.mean(th.sum(th.square(th.diff(all_hidden, 2, dim=1)), dim=-1))
    jerk_loss = 1e3 * th.mean(th.sum(th.square(th.diff(xy[:, :, 2:], 2, dim=1)), dim=-1))

    total_loss = cartesian_loss + muscle_loss + jerk_loss + spectral_loss

    return {
        'total': total_loss,
        'cartesian': cartesian_loss,
        'muscle': muscle_loss,
        'spectral': spectral_loss,
        'jerk': jerk_loss
    }


# Compute deviations for a single episode.
def calculate_deviations(task, episode_data, n_t, eps = 1e-6):
    xy = episode_data['xy']
    xy = task.shift_traj(xy)
    targets = episode_data['targets']

    point1 = th.tile(th.unsqueeze(targets[:, 0, :], 1), (1, n_t - 1, 1))
    point2 = th.tile(th.unsqueeze(targets[:, -1, :], 1), (1, n_t - 1, 1))
    diff = point2 - point1
    numerator = (diff[:, :, 1] * xy[:, :, 0] - diff[:, :, 0] * xy[:, :, 1]
                 + point2[:, :, 0] * point1[:, :, 1] - point2[:, :, 1] * point1[:, :, 0])
    denominator = th.sqrt(th.pow(diff[:, :, 1], 2) + th.pow(diff[:, :, 0], 2))
    nonzero_den = (denominator.abs() > eps).all(dim=1)
    ratio = numerator[nonzero_den] / denominator[nonzero_den]
    lateral = th.mean(ratio[th.arange(ratio.shape[0]),(th.max(th.abs(ratio), dim=1)[1])])

    endpoint = th.mean(th.norm(targets[:, -1, :] - xy[:, -1, :2], dim=1))

    return lateral, endpoint


# Run multiple training batches and optimize the policy.
def train_batches(env, task, policy, optimizer, n_batches, batch_size, interval, ep_dur, device, run_mode, force_field, contextual_cue, k, *args, **kwargs ):

    total_losses = []
    cartesian_losses = []
    muscle_losses = []
    spectral_losses = []
    jerk_losses = []
    endpoint_dev = []
    lateral_dev = []
    weight_dic = {}
    task.run_mode = run_mode
    n_t = int(ep_dur / env.effector.dt) + 1
    for batch in range(n_batches):
        episode_data = run_episode(env = env, task = task, policy = policy, batch_size = batch_size, n_t = n_t, device = device, force_field = force_field, contextual_cue=contextual_cue, k=k, *args, **kwargs)
        loss_dict = calculate_loss(task, episode_data)
        total_losses.append(loss_dict['total'].item())
        cartesian_losses.append(loss_dict['cartesian'].item())
        muscle_losses.append(loss_dict['muscle'].item())
        spectral_losses.append(loss_dict['spectral'].item())
        jerk_losses.append(loss_dict['jerk'].item())

        lat_dev, end_dev = calculate_deviations(task, episode_data, n_t)
        if task.cue == 'left':
            lat_dev = - lat_dev
        else:
            lat_dev = lat_dev
        # Detach: lat_dev/end_dev carry grad_fn from episode_data's autograd graph; appending
        # the live tensors to a list across batches pins the per-episode graph and leaks memory.
        lateral_dev.append(lat_dev.detach())
        endpoint_dev.append(end_dev.detach())

        # Save the weights before updating them.
        if batch % interval == 0  or batch == n_batches - 1:
            weight_dic[f'{batch}'] = copy.deepcopy(policy.state_dict())

        # backward pass & update weights
        optimizer.zero_grad()
        loss_dict['total'].backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(),max_norm=1)  # important to make sure gradients don't get crazy
        optimizer.step()
        # Gradient masks alone do not freeze parameters for stateful optimizers:
        # retained Adam momentum (or weight decay) can still move zero-gradient
        # entries. Restore the exact phase-boundary values after every update.
        if hasattr(policy, 'enforce_frozen_weights'):
            policy.enforce_frozen_weights()



    return {'total' : total_losses, 'cartesian': cartesian_losses, 'muscle': muscle_losses, 'spectral': spectral_losses,
            'jerk': jerk_losses}, endpoint_dev, lateral_dev, weight_dic


# Run test episodes without training.
def test_batches(env , task, policy, n_batches, batch_size, interval, ep_dur, device, run_mode, force_field, contextual_cue, k, *args, **kwargs ):
    task.run_mode = run_mode
    n_t = int(ep_dur / env.effector.dt) + 1
    endpoint_dev = []
    lateral_dev = []
    for batch in range(n_batches):
        episode_data = run_episode(env=env, task=task, policy=policy, batch_size=batch_size, n_t=n_t, device=device, force_field=force_field, contextual_cue=contextual_cue, k=k, *args, **kwargs)
        lat_dev, end_dev = calculate_deviations(task, episode_data, n_t)
        if task.cue == 'left':
            lat_dev = - lat_dev
        else:
            lat_dev = lat_dev
        lateral_dev.append(lat_dev)
        endpoint_dev.append(end_dev)

    return episode_data, endpoint_dev, lateral_dev




