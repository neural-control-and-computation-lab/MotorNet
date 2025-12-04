import numpy as np
import torch as th

# Here we are defining the Task. This is what we update to change the tasks the network has to learn.
class ExpTask:
    def __init__(self, effector, **kwargs):
        self.effector = effector
        self.dt = self.effector.dt
        self.delay_range = kwargs.get('delay_range', [0.1, 0.3]) # delay period
        self.run_mode = kwargs.get('run_mode', 'train')# run mode – this is useful to switch between a training mode and an experimental mode, or different versions of a task
        self.cue = kwargs.get('cue', 'null')
        self.movement_array = [0.0, 0.0, 0.0, 0.0]
        self.shift_vis = kwargs.get('shift_vis', False)
        # self.k = kwargs.get('k', 0)
        # self.force_type = kwargs.get('force_type', 'CW')

    def generate(self, batch_size, n_timesteps, **kwargs):
        # Draw the goal states randomly from all reachable states
        goal_states = self.effector.joint2cartesian(self.effector.draw_random_uniform_states(batch_size=batch_size)).to("cpu").numpy()
        # tile over time
        targets = np.tile(np.expand_dims(goal_states, axis=1), (1, n_timesteps, 1))
        # create empty input matrix with 2 dimensions for the x and y coordinates, and 1 for the go cue
        inputs = np.zeros(shape=(batch_size, n_timesteps, 2 + 1))


        delay_range = self.delay_range

        base_joint = np.deg2rad([40., 110., 0., 0.]).astype(np.float32)

        # Circular targets
        rad = 0.1
        n = 8
        angle = np.linspace(0,2*np.pi,n, endpoint=False)
        circular_offset = rad*np.array([np.cos(angle), np.sin(angle),np.zeros(n),np.zeros(n)])
        movement_shift = 0.2

        if self.run_mode == 'test_center_out': 
            catch_chance = 0.
            delay_range = [0.5, 0.5] #[0.2, 0.2]
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
            start_states = init_states
        elif self.run_mode == 'test':
            catch_chance = 0.
            delay_range = [0.5,0.5] #[0.2, 0.2]
            init_states = self.effector.draw_random_uniform_states(batch_size).detach().cpu().numpy()
            start_states = init_states
        elif self.run_mode == 'test_mov_diff_loc':
            catch_chance = 0.
            delay_range = [0.5, 0.5] #[0.2, 0.2]
            base_cartesian = self.effector.joint2cartesian(th.tensor(base_joint, dtype=th.float32))
            if self.cue == 'right':
                self.movement_array = th.tensor([0.1, 0, 0, 0], dtype=th.float32)
            elif self.cue == 'left':
                self.movement_array = th.tensor([-0.1, 0, 0, 0], dtype=th.float32)
            else:
                self.movement_array = th.tensor([0, 0, 0, 0], dtype=th.float32)
            shifted_cartesian = base_cartesian + self.movement_array
            shifted_base_joint = self.effector.cartesian2joint(shifted_cartesian).cpu().numpy().reshape(-1)
            init_states = np.repeat(np.expand_dims(shifted_base_joint, axis=0), batch_size, axis=0)
            start_states = init_states
        elif self.run_mode == 'test_pro_loc':
            catch_chance = 0.
            delay_range = [0.5, 0.5] #[0.2, 0.2]
            self.shift_vis = True
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
            start_states = init_states
            base_cartesian = self.effector.joint2cartesian(th.tensor(base_joint, dtype=th.float32))
            if self.cue == 'right':
                self.movement_array = th.tensor([0.1, 0, 0, 0], dtype=th.float32)
            elif self.cue == 'left':
                self.movement_array = th.tensor([-0.1, 0, 0, 0], dtype=th.float32)
            else:
                self.movement_array = th.tensor([0, 0, 0, 0], dtype=th.float32)
            shifted_cartesian = base_cartesian + self.movement_array
            shifted_base_joint = self.effector.cartesian2joint(shifted_cartesian).cpu().numpy().reshape(-1)
            init_states = np.repeat(np.expand_dims(shifted_base_joint, axis=0), batch_size, axis=0)
        elif self.run_mode == 'test_vis_loc':
            catch_chance = 0.
            delay_range = [0.5, 0.5] #[0.2, 0.2]
            self.shift_vis = True
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
            base_cartesian = self.effector.joint2cartesian(th.tensor(base_joint, dtype=th.float32))
            if self.cue == 'right':
                self.movement_array = th.tensor([-0.1, 0, 0, 0], dtype=th.float32)
            elif self.cue == 'left':
                self.movement_array = th.tensor([0.1, 0, 0, 0], dtype=th.float32)
            else:
                self.movement_array = th.tensor([0, 0, 0, 0], dtype=th.float32)
            shifted_cartesian = base_cartesian - self.movement_array
            shifted_base_joint = self.effector.cartesian2joint(shifted_cartesian).cpu().numpy().reshape(-1)
            start_states = np.repeat(np.expand_dims(shifted_base_joint, axis=0), batch_size, axis=0)
        elif self.run_mode == 'train_center_out':
            delay_range = [0.2, 0.8]  # Added
            catch_chance = 0.25  # 0.5
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
            start_states = init_states
        elif self.run_mode == 'train_mov_diff_loc':
            delay_range = [0.2, 0.8] # Added
            catch_chance = 0.25 #0.5
            base_cartesian = self.effector.joint2cartesian(th.tensor(base_joint, dtype=th.float32))
            if self.cue == 'right':
                self.movement_array = th.tensor([0.1, 0, 0, 0], dtype=th.float32)
            elif self.cue == 'left':
                self.movement_array = th.tensor([-0.1, 0, 0, 0], dtype=th.float32)
            else:
                self.movement_array = th.tensor([0, 0, 0, 0], dtype=th.float32)
            shifted_cartesian = base_cartesian + self.movement_array
            shifted_base_joint = self.effector.cartesian2joint(shifted_cartesian).cpu().numpy().reshape(-1)
            init_states = np.repeat(np.expand_dims(shifted_base_joint, axis=0), batch_size, axis=0)
            start_states = init_states
        elif self.run_mode == 'train_pro_loc':
            delay_range = [0.2, 0.8] # Added
            catch_chance = 0.25 #0.5
            self.shift_vis = True
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
            start_states = init_states
            base_cartesian = self.effector.joint2cartesian(th.tensor(base_joint, dtype=th.float32))
            if self.cue == 'right':
                self.movement_array = th.tensor([0.1, 0, 0, 0], dtype=th.float32)
            elif self.cue == 'left':
                self.movement_array = th.tensor([-0.1, 0, 0, 0], dtype=th.float32)
            else:
                self.movement_array = th.tensor([0, 0, 0, 0], dtype=th.float32)
            shifted_cartesian = base_cartesian + self.movement_array
            shifted_base_joint = self.effector.cartesian2joint(shifted_cartesian).cpu().numpy().reshape(-1)
            init_states = np.repeat(np.expand_dims(shifted_base_joint, axis=0), batch_size, axis=0)
        elif self.run_mode == 'train_vis_loc':
            delay_range = [0.2, 0.8] # Added
            catch_chance = 0.25 #0.5
            self.shift_vis = True
            init_states = np.repeat(np.expand_dims(base_joint, axis=0), batch_size, axis=0)
            base_cartesian = self.effector.joint2cartesian(th.tensor(base_joint, dtype=th.float32))
            if self.cue == 'right':
                self.movement_array = th.tensor([-0.1, 0, 0, 0], dtype=th.float32)
            elif self.cue == 'left':
                self.movement_array = th.tensor([0.1, 0, 0, 0], dtype=th.float32)
            else:
                self.movement_array = th.tensor([0, 0, 0, 0], dtype=th.float32)
            shifted_cartesian = base_cartesian - self.movement_array
            shifted_base_joint = self.effector.cartesian2joint(shifted_cartesian).cpu().numpy().reshape(-1)
            start_states = np.repeat(np.expand_dims(shifted_base_joint, axis=0), batch_size, axis=0)
        else: # train random
            delay_range = [0.2, 0.8] # Added
            catch_chance = 0.25 #0.5 # catch trials with no go cue are useful to prevent the network from anticipating the go cue
            init_states = self.effector.draw_random_uniform_states(batch_size).detach().cpu().numpy() # random initial state
            start_states = init_states

        # Create inputs and targets for all individual trials in the batch
        for i in range(batch_size):
            delay_time = generate_delay_time(delay_range[0] / self.dt, delay_range[1] / self.dt, 'random')
            # start_point = self.effector.joint2cartesian(th.tensor(init_states[i,:])).to("cpu").numpy()
            start_point = self.effector.joint2cartesian(th.tensor(start_states[i, :])).to("cpu").numpy()

            if np.greater_equal(np.random.rand(), catch_chance):
                is_catch = False
            else:
                is_catch = True

            # Targets
            if self.run_mode == 'test_center_out':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + circular_offset[:,i % n].T, axis = 1), (1, n_timesteps, 1))
            elif self.run_mode == 'test':
                targets[i, :, :] = np.tile(np.expand_dims(self.effector.joint2cartesian(self.effector.draw_random_uniform_states(1)).detach().cpu().numpy(), axis=1), (1, n_timesteps, 1))
            elif self.run_mode == 'test_mov_diff_loc':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + circular_offset[:,i % n].T, axis = 1),(1, n_timesteps, 1))
            elif self.run_mode == 'test_pro_loc':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + circular_offset[:,i % n].T, axis = 1),(1, n_timesteps, 1))
            elif self.run_mode == 'test_vis_loc':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + circular_offset[:,i % n].T, axis = 1),(1, n_timesteps, 1))
            elif self.run_mode == 'train_center_out':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + circular_offset[:,i % n].T, axis = 1),(1, n_timesteps, 1))
            elif self.run_mode == 'train_mov_diff_loc':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + circular_offset[:,i % n].T, axis = 1),(1, n_timesteps, 1))
            elif self.run_mode == 'train_pro_loc':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + circular_offset[:,i % n].T, axis = 1),(1, n_timesteps, 1))
            elif self.run_mode == 'train_vis_loc':
                targets[i, :, :] = np.tile(np.expand_dims(start_point + circular_offset[:,i % n].T, axis = 1),(1, n_timesteps, 1))
            else:
                targets[i, :, :] = np.tile(np.expand_dims(self.effector.joint2cartesian(self.effector.draw_random_uniform_states(1)).detach().cpu().numpy(), axis=1),(1, n_timesteps, 1))

            inputs[i, :, 0:2] = targets[i, -1, 0:2]
            # Go cue
            inputs[i, :, 2] = 1.

            if not is_catch:
                targets[i, 0:delay_time, :] = start_point
                inputs[i, delay_time:, 2] = 0 # turn off go cue to initiate movement
            else:
                targets[i, :, :] = start_point

            # Add noise to the inputs
            inputs[i, :, :] = inputs[i, :, :] + np.random.normal(loc=0., scale=1e-3,
                                                                 size=(inputs.shape[1], inputs.shape[2]))

        all_inputs = {"inputs": inputs}
        return [all_inputs, targets, init_states]
        #return [inputs, targets, init_states]

    def shift_obs(self, obs):
        if self.shift_vis:
            obs[:,:2] = obs[:,:2] - self.movement_array[:2]
        return obs

    def shift_traj(self, xy):
        if self.shift_vis:
            xy = xy - self.movement_array
        return xy


def generate_delay_time(delay_min, delay_max, delay_mode):
    if delay_mode == 'random':
        delay_time = np.random.uniform(delay_min, delay_max)
    elif delay_mode == 'noDelayInput':
        delay_time = 0
    else:
        raise AttributeError

    return int(delay_time)
