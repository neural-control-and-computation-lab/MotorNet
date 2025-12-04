import os
import sys

from numpy.f2py.crackfortran import endifs

# Add project root to Python path for imports
this_file = os.path.abspath(__file__)
project_root = os.path.dirname(os.path.dirname(this_file))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import numpy as np
import torch as th
import motornet as mn
import Env, Task
from PolicyLoad import creat_policy
from Plots import plot_deviation, plot_training_loss,plot_subspace
from Model import run_batches
from Model import test_batches
from utils import policy_mod
from joblib import Parallel, delayed

print('All packages imported.')
print('pytorch version: ' + th.__version__)
print('numpy version: ' + np.__version__)
print('motornet version: ' + mn.__version__)

device = th.device("cpu")


def train_model(task_id):
    saveLoc = f"/Users/pounemirzazadeh/Motornet/MultiNet/task_{task_id}/"
    os.makedirs(os.path.dirname(saveLoc), exist_ok=True)

    inputs, targets, init_states = task.generate(1, n_t)

    phases = ['growing_up']  # Training phases
    k_values = [0]  # strength of force field
    n_batches = [5000]
    training_random = [True]  # Training for the first time!
    load_baseline = [False]   # Loading weights from growing up
    run_modes = ['train']
    results = {}
    simulation_mode = 'train'
    exp = 'baseline'
    

    # phases = ['NF1', 'FF1','NF2']     # Training phases
    # k_values = [0,8,0]                       # strength of force field
    # n_batches = [1000,1000,500]
    # training_random = [False, False, False]
    # load_baseline = [True, False, False]
    # run_modes = ['train_vis_loc', 'train_vis_loc', 'train_vis_loc']
    # simulation_mode = 'train'
    # exp = 'vis_loc'
    # results = {}

    for i , (phase, k_value, n_batch, load_baseline_flag, train_random_flag, run_mode) in enumerate(zip(phases, k_values, n_batches, load_baseline, training_random, run_modes)):
        print(f"Running phase: {phase}")
        input_freeze, output_freeze, optimizer_mod, learning_rate = policy_mod(phase)
        policy, optimizer = creat_policy(env = env, inputs = inputs, device = device, policy_func=mn.policy.ModularPolicyGRU, optimizer_mod=optimizer_mod, learning_rate=learning_rate)

      if train_random_flag:
            pass
      elif load_baseline_flag:
            policy.load_state_dict(th.load(saveLoc + f'weights_growing_up_baseline'))
            policy.freeze(input_freeze=input_freeze, output_freeze=output_freeze)
      else:
            policy.load_state_dict(th.load(saveLoc + f'weights_{phases[i - 1]}_{exp}'))
            policy.freeze(input_freeze=input_freeze, output_freeze=output_freeze)
  

        if phase == 'FF1':
            force_field = 'random'
        else:
            force_field = 'null'

        if phase == 'NF1' or phase == 'NF2':
            contextual_cue = 'random'
        else:
            contextual_cue = 'force_dependent'

        losses, endpoint_dev, lateral_dev, weights = run_batches(env=env, task=task, policy=policy, optimizer=optimizer,
                                                                 n_batches=n_batch, batch_size=32, interval=200,
                                                                 ep_dur=ep_dur,
                                                                 device=device, run_mode=run_mode,
                                                                 simulation_mode=simulation_mode,
                                                                 force_field=force_field, contextual_cue=contextual_cue,
                                                                 k=k_value)

        # Store results
        results[phase] = {
            'losses': losses,
            'endpoint_dev': endpoint_dev,
            'lateral_dev': lateral_dev
        }

        th.save(policy.state_dict(), saveLoc + f'weights_{phases[i]}_{exp}')

    th.save(results, saveLoc + f"results_{exp}")

if __name__ == "__main__":
    dt = 0.01  # time step in seconds
    ep_dur = 2.0  # episode duration in seconds

    mm = mn.muscle.RigidTendonHillMuscle()  # muscle model
    ee = mn.effector.RigidTendonArm26(muscle=mm, timestep=dt)  # effector model

    # Initialize the environment
    env = Env.ExpTaskEnv(max_ep_duration=ep_dur, effector=ee,
                         proprioception_delay=0.01, vision_delay=0.07,
                         proprioception_noise=1e-3, vision_noise=1e-3, action_noise=1e-4)

    obs, info = env.reset()
    n_t = int(ep_dur / env.effector.dt)

    # Initialize the task
    task = Task.ExpTask(effector=env.effector)

    num_tasks = 8
    Parallel(n_jobs = 8, batch_size = 'auto')(
        delayed(train_model)(task_id) for task_id in range(num_tasks)
    )
