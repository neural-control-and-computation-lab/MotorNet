import sys
import os
import torch as th
import motornet as mn
import copy

# Get the directory
current_dir = os.path.dirname(os.path.abspath(__file__))

if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

import Env, Task
from utils import policy_mod
from PolicyLoad import creat_policy
from Model import train_batches, test_batches, calculate_deviations
from joblib import Parallel, delayed

class ExpConfig:
    def __init__(self, net_id, mode = 'test', exp_name='center_out'):
        self.mode = mode
        self.net_id = net_id
        self.exp = exp_name
        self.dt = 0.01
        self.ep_dur = 2.0
        self.device = th.device("cpu")
        self.saveLoc = f"/Users/pounemirzazadeh/Motornet/MultiNet/Modular_version/new_test/task_{net_id}/"
        os.makedirs(self.saveLoc, exist_ok=True)
        # Configuration for phases:
        if self.mode == 'baseline':
            self._setup_baseline()
        elif self.mode == 'train':
            self._setup_training()
        elif self.mode == 'baseline_test':
            self._setup_baseline_testing()
        else:
            self._setup_testing()

    def _setup_baseline(self):
        self.batch_size = 32
        self.phases = ['growing_up']
        self.k_values = [0]
        self.n_batches = [5000]
        self.load_baseline = [False]
        self.run_modes = ["train_rand"]
        self.exp_save_names = ["baseline"]

    def _setup_baseline_testing(self):
        self.batch_size = 128
        self.phases = ['growing_up']
        self.k_values = [0]
        self.n_batches = [1]
        self.run_modes = ["test_rand"]
        self.exp_save_names = ["baseline"]
        self.contextual_cues = ['null']
        self.force_fields = ['null']

    def _setup_training(self):
        self.batch_size = 32
        self.phases = ['NF1', 'FF1', 'NF2']
        self.k_values = [0,12,0]
        self.n_batches = [500,100,50]#[700,70,50]
        self.load_baseline = [True, False, False]   # This is useful for the phases needing growing up weights
        self.run_modes = [f"{self.mode}_{self.exp}"]*3
        self.exp_save_names = [f"{self.exp}"]*3


    def _setup_testing(self):
        self.batch_size = 128
        self.phases = ['NF1', 'FF1', 'NF2']
        self.k_values = [0, 12, 0]
        self.n_batches = [1, 1, 1]
        self.run_modes = [f"{self.mode}_{self.exp}"]*3
        self.contextual_cues = ['left','right']
        self.force_fields = ['CCW','CW']



def create_setup(cfg):
    mm = mn.muscle.RigidTendonHillMuscle()
    ee = mn.effector.RigidTendonArm26(muscle=mm, timestep=cfg.dt)
    env = Env.ExpTaskEnv(max_ep_duration=cfg.ep_dur, effector=ee, proprioception_delay=0.01, vision_delay=0.07,
                         proprioception_noise=1e-4, vision_noise=1e-3, action_noise=1e-4)
    obs, info = env.reset()
    task = Task.ExpTask(effector=ee)
    return env, task


def run_training(env, task, cfg):
    n_t = int(cfg.ep_dur / env.effector.dt) + 1 # Why + 1 (I added recently)
    inputs, targets, init_states = task.generate(1, n_t)
    results = {}

    for i , (phase, k_value, n_batch, load_baseline_flag, run_mode, exp_save_name) in (
            enumerate(zip(cfg.phases, cfg.k_values, cfg.n_batches, cfg.load_baseline, cfg.run_modes, cfg.exp_save_names))):

        print(f"Running phase: {phase}")
        input_freeze, output_freeze, optimizer_mod, learning_rate = policy_mod(phase)
        policy, optimizer = creat_policy(env = env, inputs = inputs, device = cfg.device, policy_func=mn.policy.ModularPolicyGRU, optimizer_mod=optimizer_mod, learning_rate=learning_rate)

        if "growing_up" in cfg.phases:
            pass
        else:
            if load_baseline_flag:
                weightsDic = th.load(cfg.saveLoc + f'weightsDic_growing_up_baseline')
                policy.load_state_dict(weightsDic[next(reversed(weightsDic))])

                policy.freeze(input_freeze=input_freeze, output_freeze=output_freeze)
                prev_opt_path = cfg.saveLoc + 'optStateDic_growing_up_baseline'
            else:
                weightsDic = th.load(cfg.saveLoc + f'weightsDic_{cfg.phases[i-1]}_{cfg.exp_save_names[i-1]}')
                policy.load_state_dict(weightsDic[next(reversed(weightsDic))])
                policy.freeze(input_freeze=input_freeze, output_freeze=output_freeze)
                prev_opt_path = cfg.saveLoc + f'optStateDic_{cfg.phases[i-1]}_{cfg.exp_save_names[i-1]}'
            # Carry Adam's m/v across phase boundaries to avoid the lr*sign(g) first-step blow-up.
            if os.path.exists(prev_opt_path):
                optimizer.load_state_dict(th.load(prev_opt_path))
            # The loaded optimizer contains moments for weights frozen at this
            # phase boundary. Zero only those entries; recurrent weights retain
            # their optimizer history and continue adapting normally.
            policy.clear_frozen_optimizer_state(optimizer)

        force_field = 'random' if phase == 'FF1' else 'null'
        contextual_cue = 'random' if phase in ['NF1', 'NF2'] else 'force_dependent'
        interval = 100 if phase == 'growing_up' else 50 if phase == 'NF1' else 5

        losses, endpoint_dev, lateral_dev, weights = train_batches(
            env=env, task=task, policy=policy, optimizer=optimizer, n_batches=n_batch,
            batch_size=cfg.batch_size, interval=interval, ep_dur=cfg.ep_dur, device=cfg.device, run_mode=run_mode,
            force_field=force_field, contextual_cue=contextual_cue, k=k_value)

        # Store results
        results[phase] = {
            'losses': losses,
            'endpoint_dev': endpoint_dev,
            'lateral_dev': lateral_dev
        }


        th.save(weights, cfg.saveLoc + f'weightsDic_{cfg.phases[i]}_{cfg.exp_save_names[i]}')
        # Persist Adam moments so the next phase can resume without a fresh-optimizer first-step jump.
        th.save(optimizer.state_dict(), cfg.saveLoc + f'optStateDic_{cfg.phases[i]}_{cfg.exp_save_names[i]}')

    th.save(results, cfg.saveLoc + f"results_{cfg.exp}")



def run_testing(env, task, cfg):
    n_t = int(cfg.ep_dur / env.effector.dt) + 1
    inputs, targets, init_states = task.generate(1, n_t)
    test_data = {}
    deviations = {}
    for force_field,contextual_cue in zip(cfg.force_fields, cfg.contextual_cues):
        for i, (phase, k_value, n_batch, run_mode) in enumerate(zip(cfg.phases, cfg.k_values, cfg.n_batches, cfg.run_modes)):
            test_data[phase] = {}
            deviations[phase] = {}
            input_freeze, output_freeze, optimizer_mod, learning_rate = policy_mod(phase='growing_up')
            policy, optimizer = creat_policy(env=env, inputs=inputs, device=cfg.device, policy_func=mn.policy.ModularPolicyGRU,
                                             optimizer_mod=optimizer_mod, learning_rate=learning_rate)
            weights = th.load(cfg.saveLoc + f'weightsDic_{phase}_{cfg.exp}')
            for batch_number in weights.keys():
                deviations[phase][batch_number] = {}
                policy.load_state_dict(weights[batch_number])

                testdata, endpoint_dev, lateral_dev = test_batches(env=env, task=task, policy=policy, n_batches=n_batch, batch_size=cfg.batch_size, interval=1,
                                        ep_dur=cfg.ep_dur, device=cfg.device, run_mode=run_mode,
                                        simulation_mode=cfg.mode, force_field=force_field, contextual_cue=contextual_cue,
                                        k=k_value, title=f'{cfg.exp} - {phase} - {cfg.net_id} - batch = {batch_number}')
                test_data[phase][batch_number] = testdata
                deviations[phase][batch_number]['lateral'] = lateral_dev
                deviations[phase][batch_number]['endpoint'] = endpoint_dev

        th.save(test_data, cfg.saveLoc + f"test_data_{cfg.exp}_{force_field}")
        th.save(deviations, cfg.saveLoc + f"deviations_{cfg.exp}_{force_field}")




def exp_simulation(net_id = 1, mode='test', exp='center_out'):
    cfg = ExpConfig(net_id, mode = mode, exp_name=exp)
    env, task = create_setup(cfg)


    if mode == 'train' or mode == 'baseline':
        run_training(env = env, task = task, cfg = cfg)
    elif mode == 'test' or mode == 'baseline_test':
        run_testing(env = env, task = task, cfg = cfg)


if __name__ == "__main__":

    num_networks = 4 # Number of networks/subjects
    mode = ('test')  # Set to 'train' to run the learning loop and 'test' to run the test loop
    experiments = ['center_out','mov_diff_loc', 'pro_loc','vis_loc']


    if mode == 'train':
        Parallel(n_jobs=num_networks)(
            delayed(exp_simulation)(net_id=i, mode='baseline', exp='baseline') for i in range(num_networks)
        )
        print(f"The baseline training completed across {num_networks} networks.")

    for exp in experiments:
        Parallel(n_jobs=num_networks)(
            delayed(exp_simulation)(net_id=i, mode=mode, exp=exp) for i in range(num_networks)
        )
        print(f"The {mode} mode for experiment {exp} completed across {num_networks} networks.")
