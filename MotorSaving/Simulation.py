# ===========================================================================
#           Main script for training and testing MotorNet model
# ===========================================================================

import os
import sys

# Add project root to Python path for imports
this_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(this_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import numpy as np
import torch as th
import motornet as mn
import Env, Task
from PolicyLoad import creat_policy
from Plots import plot_deviation, plot_training_loss
from Model import run_batches
from Model import test_batches
from utils import policy_mod

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

device = th.device("cpu")

# Save location for model weights and test data
saveLoc = '/Users/pounemirzazadeh/Motornet/Modular'

dt = 0.01 # time step in seconds
ep_dur = 1.0 # episode duration in seconds

mm = mn.muscle.RigidTendonHillMuscle() # muscle model
ee = mn.effector.RigidTendonArm26(muscle=mm, timestep=dt) # effector model

# Initialize the environment
env = Env.ExpTaskEnv(max_ep_duration=ep_dur, effector=ee,
                     proprioception_delay=0.01, vision_delay=0.07,
                     proprioception_noise=1e-3, vision_noise=1e-3, action_noise=1e-4)
obs, info = env.reset()
n_t = int(ep_dur / env.effector.dt)

# Initialize the task
task = Task.ExpTask(effector=env.effector)
inputs, targets, init_states = task.generate(1, n_t)


# Select simulation mode: "train" or "test"
simulation_mode = 'test'

# Training loop
if simulation_mode == 'train':
    phases = ['growing_up', 'NF1', 'FF1', 'NF2', 'FF2']     # Training phases
    k_values = [0,0,8,0,8]                                  # strength of force field
    n_batches = [2000,1000,700,1000,700]
    training_random = [True, False, False, False, False]    # Loading weights from previous phase
    run_modes = ['train', 'train_center_out', 'train_center_out', 'train_center_out', 'train_center_out'] # Type of starting point and targets; random or center_out

    results = {}

    for i , (phase, k_value, n_batch, training_random_flag, run_mode) in enumerate(zip(phases, k_values, n_batches, training_random, run_modes)):
        print(f"Running phase: {phase}")
        input_freeze, output_freeze, optimizer_mod, learning_rate = policy_mod(phase)
        policy, optimizer = creat_policy(env, inputs, device, policy_func=mn.policy.ModularPolicyGRU, optimizer_mod=optimizer_mod, learning_rate=learning_rate)

        if not training_random_flag:
            policy.load_state_dict(th.load(saveLoc + f'weights_{phases[i-1]}'))
            policy.freeze(input_freeze=input_freeze, output_freeze=output_freeze)
        else:
            pass

        losses, endpoint_dev, lateral_dev = run_batches(env = env, task = task, policy = policy, optimizer = optimizer,
                                                    n_batches = n_batch, batch_size = 32, interval = 500, ep_dur = ep_dur,
                                                    device = device, run_mode = run_mode, k = k_value)

        th.save(policy.state_dict(), saveLoc + f'weights_{phases[i]}')

        # Store results
        results[phase] = {
            'losses': losses,
            'endpoint_dev': endpoint_dev,
            'lateral_dev': lateral_dev
        }

    th.save(results, saveLoc + "results")

    #  Plot the loss and deviations for each phase
    for phase, data in results.items():
        plot_deviation(data['lateral_dev'], data['endpoint_dev'], title=f"Deviation - {phase}")
        plot_training_loss(data['losses'], title=f"Loss - {phase}")

# Testing loop
elif simulation_mode == 'test':

    # Choose phase to test center_out reaching experiment
    learned_phase = 'NF1'

    print(f"Testing phase: {learned_phase} ")
    input_freeze, output_freeze, optimizer_mod, learning_rate = policy_mod(phase = 'growing_up')
    policy, optimizer = creat_policy(env, inputs, device, policy_func=mn.policy.ModularPolicyGRU, optimizer_mod=optimizer_mod, learning_rate=learning_rate)
    policy.load_state_dict(th.load(saveLoc + f'weights_{learned_phase}'))
    testdata = test_batches(env = env, task = task, policy = policy, n_batches = 1, batch_size = 32, interval = 1, ep_dur = ep_dur, device = device, run_mode = 'test_center_out')

    th.save(testdata, saveLoc + f'data_{learned_phase}')
