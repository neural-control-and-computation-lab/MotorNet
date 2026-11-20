import torch as th
import numpy as np
from Plots import plot_episode
from utils import applied_load, applied_external_load, movement_phase_mask
import copy
import motornet as mn


# Run a single episode
def run_episode(env, task, policy, batch_size, n_t, device, force_field='null', contextual_cue='null', k=0,
                is_channel=None, chan_k=2000., chan_b=125., flip_cue=False, *args, **kwargs):
    if force_field == 'random':
        force_dir = str(np.random.choice(['CW', 'CCW']))
    elif force_field == 'CW':
        force_dir = 'CW'
    elif force_field == 'CCW':
        force_dir = 'CCW'
    else:
        force_dir = None

    if contextual_cue == 'null':
        task.cue = 'null'
    elif contextual_cue == 'random':
        task.cue = str(np.random.choice(['right', 'left']))
    elif contextual_cue == 'right':
        task.cue = 'right'
    elif contextual_cue == 'left':
        task.cue = 'left'
    elif contextual_cue == 'force_dependent':
        if force_dir == 'CW':
            task.cue = 'left' if flip_cue else 'right'
        elif force_dir == 'CCW':
            task.cue = 'right' if flip_cue else 'left'
        else:
            task.cue = 'null'
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
    all_channel_force = []

    # Integer counter avoids int(elapsed / dt) float truncation skipping steps.
    t_step = 0
    while not terminated:  # will run until `max_ep_duration` is reached
        obs = task.shift_obs(obs)  # This line added for Exp 8
        obs = th.concat((obs, inp[:, t_step, :]), dim=1)
        action, h = policy(obs, h)

        # Compute endpoint load (force field)
        # The task's go cue is noisy: approximately 1 while holding and 0 after go.
        # Recover that discrete phase with a midpoint threshold so the curl field is
        # continuously on during movement instead of flickering with input noise.
        force_mask = movement_phase_mask(inp[:, t_step, 2])
        applied_force, channel_force = applied_external_load(task=task, info=info, target=targets, time_step=t_step,
                                                             curl_gain=k, chan_k=chan_k, chan_b=chan_b,
                                                             force_mask=force_mask, mode=force_dir,
                                                             is_channel=is_channel)

        obs, reward, terminated, truncated, info = env.step(action=action, endpoint_load=applied_force)
        xy.append(info['states']['cartesian'][:, None, :])
        all_actions.append(action[:, None, :])
        all_muscle.append(info['states']['muscle'][:, 0, None, :])
        all_force.append(info['states']['muscle'][:, -1, None, :])
        all_hidden.append(h[:, None, :])
        all_targets.append(th.unsqueeze(targets[:, t_step, :], dim=1))
        all_inp.append(th.unsqueeze(inp[:, t_step, :], dim=1))
        all_vis_inp.append(obs[:, None, :2])
        all_joint.append(info['states']['joint'][:, None, :])
        all_channel_force.append(channel_force[:, None, :])
        t_step += 1

    ext_force = force_dir

    return {
        'xy': th.cat(xy, dim=1),
        'hidden': th.cat(all_hidden, dim=1),
        'actions': th.cat(all_actions, dim=1),
        'muscle': th.cat(all_muscle, dim=1),
        'force': th.cat(all_force, dim=1),
        'targets': th.cat(all_targets, dim=1),
        'inp': th.cat(all_inp, dim=1),
        'vis_inp': th.cat(all_vis_inp, dim=1),
        'joint': th.cat(all_joint, dim=1),
        'ext_force': ext_force,
        'traj': task.shift_traj(th.cat(xy, dim=1)),
        'channel_force': th.cat(all_channel_force, dim=1),
    }


# Compute losses for a single episode.
def calculate_loss(task, episode_data):
    xy = episode_data['xy']
    xy = task.shift_traj(xy)
    all_targets = episode_data['targets']
    all_hidden = episode_data['hidden']
    all_force = episode_data['force']

    cartesian_loss = 1e3 * th.mean(th.sum(th.abs(xy[:, :, :2] - all_targets), dim=-1))
    # Match the Two Target headline objective's weaker effort penalty and stronger
    # trajectory-smoothness penalty while leaving every loss definition unchanged.
    muscle_loss = 1e-1 * th.mean(th.sum(all_force, dim=-1))
    spectral_loss = 1e2 * th.mean(th.sum(th.square(th.diff(all_hidden, 2, dim=1)), dim=-1))
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
def calculate_deviations(task, episode_data, n_t, eps=1e-6):
    # xy = episode_data['xy']
    # xy = task.shift_traj(xy)
    xy = episode_data['traj']
    targets = episode_data['targets']
    n_steps = xy.shape[1]

    point1 = th.tile(th.unsqueeze(targets[:, 0, :], 1), (1, n_steps, 1))
    point2 = th.tile(th.unsqueeze(targets[:, -1, :], 1), (1, n_steps, 1))
    diff = point2 - point1
    numerator = (diff[:, :, 1] * xy[:, :, 0] - diff[:, :, 0] * xy[:, :, 1]
                 + point2[:, :, 0] * point1[:, :, 1] - point2[:, :, 1] * point1[:, :, 0])
    denominator = th.sqrt(th.pow(diff[:, :, 1], 2) + th.pow(diff[:, :, 0], 2))
    nonzero_den = (denominator.abs() > eps).all(dim=1)
    if not nonzero_den.any():
        return xy.new_zeros(()), xy.new_zeros(()), xy.new_zeros((0,)), xy.new_zeros((0,))
    ratio = numerator[nonzero_den] / denominator[nonzero_den]

    # Restrict peak lateral error search to movement steps so hold drift does not mask reach error.
    move_mask = episode_data['inp'][nonzero_den, :, 2] < 0.5
    ratio_eval = th.where(move_mask.any(dim=1, keepdim=True), th.where(move_mask, ratio, th.zeros_like(ratio)), ratio)
    peak_idx = th.max(th.abs(ratio_eval), dim=1)[1]

    lateral_batch = ratio_eval[th.arange(ratio_eval.shape[0], device=xy.device), peak_idx]
    endpoint_batch = th.norm(targets[nonzero_den, -1, :] - xy[nonzero_den, -1, :2], dim=1)
    lateral_mean = th.mean(lateral_batch)
    endpoint_mean = th.mean(endpoint_batch)

    return lateral_mean, endpoint_mean, lateral_batch, endpoint_batch


def calculate_force_compensation(task, episode_data, curl_gain=13.0, eps=1e-6, flip_cue=False):
    """Compute percent force compensation on channel probe trials (Howard et al., 2013)."""
    xy = episode_data['traj']
    vel = xy[:, :, 2:]
    targets = episode_data['targets']
    wall = episode_data['channel_force']

    line_vec = targets[:, -1, :2] - targets[:, 0, :2]
    dist = th.norm(line_vec, dim=-1, keepdim=True)
    valid = dist.squeeze(-1) > eps
    if not valid.any():
        return xy.new_zeros(()), xy.new_zeros((0,))

    u = line_vec[valid] / dist[valid]
    n_cw = th.stack([u[:, 1], -u[:, 0]], dim=-1)
    ext_force = episode_data.get('ext_force', None)
    ccw_cue = 'right' if flip_cue else 'left'
    is_ccw = (ext_force == 'CCW') or (ext_force is None and task.cue == ccw_cue)
    n_field = -n_cw if is_ccw else n_cw

    move_mask = movement_phase_mask(episode_data['inp'][valid, :, 2]).squeeze(-1)
    comp_force_int = th.sum(th.sum(wall[valid] * n_field[:, None, :], dim=-1) * move_mask, dim=-1)
    ideal_force_int = curl_gain * th.sum(th.norm(vel[valid], dim=-1) * move_mask, dim=-1)
    comp_pct_batch = th.where(
        ideal_force_int > 1e-3,
        100.0 * comp_force_int / ideal_force_int.clamp_min(eps),
        th.zeros_like(comp_force_int),
    )
    return th.mean(comp_pct_batch), comp_pct_batch


# Run multiple training batches and optimize the policy.
def train_batches(env, task, policy, optimizer, n_batches, batch_size, interval, ep_dur, device, run_mode, force_field,
                  contextual_cue, k, flip_cue=False, *args, **kwargs):
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
        episode_data = run_episode(env=env, task=task, policy=policy, batch_size=batch_size, n_t=n_t, device=device,
                                   force_field=force_field, contextual_cue=contextual_cue, k=k, flip_cue=flip_cue, *args, **kwargs)
        loss_dict = calculate_loss(task, episode_data)
        total_losses.append(loss_dict['total'].item())
        cartesian_losses.append(loss_dict['cartesian'].item())
        muscle_losses.append(loss_dict['muscle'].item())
        spectral_losses.append(loss_dict['spectral'].item())
        jerk_losses.append(loss_dict['jerk'].item())

        lat_dev, end_dev, lat_dev_batch, end_dev_batch = calculate_deviations(task, episode_data, n_t)
        ccw_cue = 'right' if flip_cue else 'left'
        if (episode_data['ext_force'] == 'CCW') or (episode_data['ext_force'] is None and task.cue == ccw_cue):
            lat_dev = -lat_dev
        # Detach: lat_dev/end_dev carry grad_fn from episode_data's autograd graph; appending
        # the live tensors to a list across batches pins the per-episode graph and leaks memory.
        lateral_dev.append(lat_dev.detach())
        endpoint_dev.append(end_dev.detach())

        # Save the weights before updating them.
        if batch % interval == 0 or batch == n_batches - 1:
            weight_dic[f'{batch}'] = copy.deepcopy(policy.state_dict())

        # backward pass & update weights
        optimizer.zero_grad()
        loss_dict['total'].backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=1)  # important to make sure gradients don't get crazy
        optimizer.step()
        # Gradient masks alone do not freeze parameters for stateful optimizers:
        # retained Adam momentum (or weight decay) can still move zero-gradient
        # entries. Restore the exact phase-boundary values after every update.
        if hasattr(policy, 'enforce_frozen_weights'):
            policy.enforce_frozen_weights()

    return {'total': total_losses, 'cartesian': cartesian_losses, 'muscle': muscle_losses, 'spectral': spectral_losses,
            'jerk': jerk_losses}, endpoint_dev, lateral_dev, weight_dic


# Run test episodes without training.
def test_batches(env, task, policy, n_batches, batch_size, interval, ep_dur, device, run_mode, force_field,
                 contextual_cue, k, flip_cue=False, *args, **kwargs):
    task.run_mode = run_mode
    n_t = int(ep_dur / env.effector.dt) + 1
    endpoint_dev_mean = []
    lateral_dev_mean = []
    endpoint_dev_batch = []
    lateral_dev_batch = []
    ccw_cue = 'right' if flip_cue else 'left'
    with th.no_grad():
        for batch in range(n_batches):
            episode_data = run_episode(env=env, task=task, policy=policy, batch_size=batch_size, n_t=n_t, device=device,
                                       force_field=force_field, contextual_cue=contextual_cue, k=k, flip_cue=flip_cue, *args, **kwargs)
            lat_dev_mean, end_dev_mean, lat_dev_batch, end_dev_batch = calculate_deviations(task, episode_data, n_t)
            if (episode_data['ext_force'] == 'CCW') or (episode_data['ext_force'] is None and task.cue == ccw_cue):
                lat_dev_mean = -lat_dev_mean
                lat_dev_batch = -lat_dev_batch

            lateral_dev_mean.append(lat_dev_mean.detach())
            endpoint_dev_mean.append(end_dev_mean.detach())
            lateral_dev_batch.append(lat_dev_batch.detach())
            endpoint_dev_batch.append(end_dev_batch.detach())

    return episode_data, endpoint_dev_mean, lateral_dev_mean, endpoint_dev_batch, lateral_dev_batch



