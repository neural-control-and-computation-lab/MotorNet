import numpy as np
import torch as th

def creat_policy(env, inputs, device, policy_func, optimizer_mod, learning_rate) :
    vision_mask = [1]
    proprio_mask = [1]
    task_mask = [1]
    output_mask = [1]
    connectivity_mask = np.array([[1]])
    connectivity_mask[connectivity_mask > 1] = 1
    connectivity_delay = np.zeros_like(connectivity_mask)
    module_sizes = [256]#[128]
    spectral_scaling = 1
    # input sparsity
    vision_dim = np.arange(env.get_vision().shape[1])
    proprio_dim = np.arange(env.get_proprioception().shape[1]) + vision_dim[-1] + 1
    task_dim = np.arange(inputs['inputs'].shape[2]) + proprio_dim[-1] + 1
    policy = policy_func(env.observation_space.shape[0] + inputs['inputs'].shape[2], module_sizes, env.n_muscles,
                              vision_dim=vision_dim, proprio_dim=proprio_dim, task_dim=task_dim,
                              vision_mask=vision_mask, proprio_mask=proprio_mask, task_mask=task_mask,
                              connectivity_mask=connectivity_mask, output_mask=output_mask,
                              connectivity_delay=connectivity_delay,
                              proportion_excitatory=None, input_gain=1.,
                              spectral_scaling=spectral_scaling, device=device, activation='rect_tanh', output_delay=1)
    # Initialize the optimizer
    if optimizer_mod == 'Adam':
        optimizer = th.optim.Adam(policy.parameters(), lr= learning_rate)
    else:
        optimizer = th.optim.SGD(policy.parameters(), lr=learning_rate)


    return policy, optimizer
