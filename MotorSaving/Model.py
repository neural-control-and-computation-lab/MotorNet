import torch as th
from Plots import plot_episode
from utils import applied_load

# Run a single episode
def run_episode(env, task, policy, batch_size, n_t, device, k = 0, *args, **kwargs ):
    inputs, targets, init_states = task.generate(batch_size, n_t)
    targets = th.tensor(targets[:, :, 0:2], device=device, dtype=th.float)
    inp = th.tensor(inputs['inputs'], device=device, dtype=th.float)
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
    all_joint = []

    while not terminated:  # will run until `max_ep_duration` is reached
        t_step = int(env.elapsed / env.dt)
        obs = th.concat((obs, inp[:, t_step, :]), dim=1)
        action, h = policy(obs, h)

        # Compute endpoint load (force field)
        force_mask = (inp[:, t_step, 2].abs() < 1e-3).float().unsqueeze(1)
        force_field  = applied_load(endpoint_vel = info['states']['cartesian'][:, 2:], k = k, mode = 'CW')
        masked_force_field = force_field * force_mask

        obs, reward, terminated, truncated, info = env.step(action=action, endpoint_load = masked_force_field)
        xy.append(info['states']['cartesian'][:, None, :])
        all_actions.append(action[:, None, :])
        all_muscle.append(info['states']['muscle'][:, 0, None, :])
        all_force.append(info['states']['muscle'][:, -1, None, :])
        all_hidden.append(h[:, None, :])
        all_targets.append(th.unsqueeze(targets[:, t_step, :], dim=1))
        all_inp.append(th.unsqueeze(inp[:, t_step, :], dim=1))
        all_joint.append(info['states']['joint'][:, None, :])

    return {
        'xy': th.cat(xy, dim=1),
        'hidden' : th.cat(all_hidden, dim=1),
        'actions' : th.cat(all_actions, dim=1),
        'muscle' : th.cat(all_muscle, dim=1),
        'force' : th.cat(all_force, dim=1),
        'targets' : th.cat(all_targets, dim=1),
        'inp' : th.cat(all_inp, dim=1),
        'joint' : th.cat(all_joint, dim=1)
    }


# Compute losses for a single episode.
def calculate_loss(episode_data):
    xy = episode_data['xy']
    all_targets = episode_data['targets']
    all_hidden = episode_data['hidden']
    all_force = episode_data['force']

    cartesian_loss = 1e3 * th.mean(th.sum(th.abs(xy[:, :, :2] - all_targets), dim=-1))
    muscle_loss = 1e-1 * th.mean(th.sum(all_force, dim=-1))
    spectral_loss = 1e4 * th.mean(th.sum(th.square(th.diff(all_hidden, 2, dim=1)), dim=-1))
    # jerk_loss = 1e4 * th.mean(th.sum(th.square(th.diff(xy[:, :, 2:], 2, dim=1)), dim=-1))
    jerk_loss = 1e3 * th.mean(th.sum(th.square(th.diff(xy[:, :, 2:], 2, dim=1)), dim=-1))

    total_loss = cartesian_loss + muscle_loss + jerk_loss + spectral_loss

    return {
        'total' : total_loss,
        'cartesian': cartesian_loss,
        'muscle': muscle_loss,
        'spectral': spectral_loss,
        'jerk': jerk_loss
    }


# Compute deviations for a single episode.
def calculate_deviations(episode_data, n_t, eps = 1e-6):
    xy = episode_data['xy']
    targets = episode_data['targets']

    point1 = th.tile(th.unsqueeze(targets[:, 0, :], 1), (1, n_t - 1, 1))
    point2 = th.tile(th.unsqueeze(targets[:, -1, :], 1), (1, n_t - 1, 1))
    diff = point2 - point1
    numerator = (diff[:, :, 1] * xy[:, :, 0] - diff[:, :, 0] * xy[:, :, 1]
                 + point2[:, :, 0] * point1[:, :, 1] - point2[:, :, 1] * point1[:, :, 0])
    denominator = th.sqrt(th.pow(diff[:, :, 1], 2) + th.pow(diff[:, :, 0], 2))
    nonzero_den = (denominator.abs() > eps).all(dim=1)
    lateral = th.mean(th.max(th.abs(numerator[nonzero_den]) / denominator[[nonzero_den]], dim=1)[0])

    ## Check zero values in numerator and denominator
    # eps = 1e-6
    # print(f'numerator size: {numerator.shape} and denominator size: {denominator.shape}')
    # zero_den = denominator.abs() < eps
    # zero_num = numerator.abs() < eps
    # both_zero = zero_den & zero_num
    # zero_den_nonzero_num = zero_den & (~zero_num)
    # print("number of times where denominator near 0 and numerator not:")
    # print(zero_den_nonzero_num.nonzero().shape)
    # print("number of times where both are near 0:")
    # print(both_zero.nonzero().shape)

    endpoint = th.mean(th.norm(targets[:, -1, :] - xy[:, -1, :2], dim=1))

    return lateral, endpoint


# Run multiple training batches and optimize the policy.
def run_batches(env, task, policy, optimizer, n_batches, batch_size, interval, ep_dur, device, run_mode, *args, **kwargs ):

    total_losses = []
    cartesian_losses = []
    muscle_losses = []
    spectral_losses = []
    jerk_losses = []
    endpoint_dev = []
    lateral_dev = []
    task.run_mode = run_mode
    n_t = int(ep_dur / env.effector.dt) + 1
    for batch in range(n_batches):
        episode_data = run_episode(env, task, policy, batch_size, n_t, device, *args, **kwargs)
        loss_dict = calculate_loss(episode_data)
        total_losses.append(loss_dict['total'].item())
        cartesian_losses.append(loss_dict['cartesian'].item())
        muscle_losses.append(loss_dict['muscle'].item())
        spectral_losses.append(loss_dict['spectral'].item())
        jerk_losses.append(loss_dict['jerk'].item())

        lat_dev, end_dev = calculate_deviations(episode_data, n_t)
        lateral_dev.append(lat_dev)
        endpoint_dev.append(end_dev)

        if run_mode == 'train' or run_mode == 'train_center_out':
            # backward pass & update weights
            loss_dict['total'].backward()
            th.nn.utils.clip_grad_norm_(policy.parameters(),
                                        max_norm=1)  # important to make sure gradients don't get crazy
            optimizer.step()
            optimizer.zero_grad()

        # Optional visualization
        plot_episode(batch, n_batches, interval, episode_data)

    return {'total' : total_losses, 'cartesian': cartesian_losses, 'muscle': muscle_losses, 'spectral': spectral_losses,
            'jerk': jerk_losses}, endpoint_dev, lateral_dev


# Run test episodes without training.
def test_batches(env, task, policy, n_batches, batch_size, interval, ep_dur, device, run_mode, *args, **kwargs ):
    task.run_mode = run_mode
    n_t = int(ep_dur / env.effector.dt) + 1
    for batch in range(n_batches):
        episode_data = run_episode(env = env, task = task, policy = policy, batch_size = batch_size, n_t = n_t, device = device, *args, **kwargs)

        plot_episode(batch, n_batches, interval, episode_data)

    return episode_data





