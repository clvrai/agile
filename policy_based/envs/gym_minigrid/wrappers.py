import math
import operator
from functools import reduce

import numpy as np
import gym
from gym import error, spaces, utils
from .minigrid import OBJECT_TO_IDX, COLOR_TO_IDX
import cv2
import random

skill_types = {
        0: u"\u21D2", # Right Arrow
        1: u"\u21D3", # Down Arrow
        2: u"\u21D0", # Left Arrow
        3: u"\u21D1", # Up Arrow
        4: u"\u21D8", # Right-Down
        5: u"\u21D7", # Right-Up
        6: u"\u21D6", # Left-Down
        7: u"\u21D9", # Left-Up
        8: u"\u21BA", # Turn-Left
        9: u"\u21BB", # Turn-Right
        10: u"\u27A4", # Move-Forward
        11: u"\u00D7" + "O", # Dig-Orange
        12: u"\u00D7" + "P", # Dig-Pink
        }

def skill_category(i):
    if i in range(0, 8):
        return 'Step'
    elif i in range(8, 10):
        return 'Turn'
    elif i == 10:
        return 'Forward'
    elif i == 11:
        return 'Dig Orange'
    elif i == 12:
        return 'Dig Pink'

diagonal_action_dict = {
        0: [0, 1],
        1: [1, 2],
        2: [2, 3],
        3: [3, 0]
        }
def generate_jump_actions():
    '''
        0-3: move_right, move_down, move_left, move_up
        4-7: right_down, right_up, left_down, left_up
        8-10: turn_face_left, turn_face_right, move_forward
        11: jump_lava_right
        12: jump_plava_right
    '''
    
    action_type = np.array([
                    [0,0,0,0,1],
                    [0,0,0,1,0],
                    [0,0,1,0,0],
                    [0,1,0,0,0],
                    [1,0,0,0,0]
                ])

    simple_type = np.array([
                    [-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                    [0, 0]
                ])
    complex_type = np.array([
                    [-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                    [0, 0]
                ])

    turn_type = np.array([
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                    [0, 0]
                ])

    Action = np.zeros([13,
            action_type.shape[-1] + simple_type.shape[-1] + complex_type.shape[-1] + turn_type.shape[-1] ])
    action_types = np.zeros([13, 2])

    for i in range(13):
        if i in [0,1,2,3]:
            Action[i] = np.concatenate((
                    action_type[0],
                    simple_type[i],
                    complex_type[-1],
                    turn_type[-1],
                    ), axis=-1)
            action_types[i] = [0, i]
                    
        elif i in [4,5,6,7]:
            Action[i] = np.concatenate((
                    action_type[1],
                    simple_type[-1],
                    complex_type[i-4],
                    turn_type[-1],
                    ), axis=-1)
            action_types[i] = [1, i-4]

        elif i in [8, 9, 10]:
            Action[i] = np.concatenate((
                    action_type[2],
                    simple_type[-1],
                    complex_type[-1],
                    turn_type[i-8],
                    ), axis=-1)
            action_types[i] = [2, i-8]

        elif i in [11, 12]:
            Action[i] = np.concatenate((
                    action_type[i-8],
                    simple_type[-1],
                    complex_type[-1],
                    turn_type[-1],
                    ), axis=-1)
            action_types[i] = [i-8, 0]
    return Action, action_types

def generate_many_jump_actions():
    '''
        0-3: move_right, move_down, move_left, move_up
        4-7: right_down, right_up, left_down, left_up
        8-10: turn_face_left, turn_face_right, move_forward
        11-13: jump_lava_right, jump_lava_right_down, jump_lava_right_up
        14-16: jump_plava_right, jump_plava_right_down, jump_plava_right_up
    '''
    
    action_type = np.array([
                    [0,0,0,0,1],
                    [0,0,0,1,0],
                    [0,0,1,0,0],
                    [0,1,0,0,0],
                    [1,0,0,0,0]
                ])

    simple_type = np.array([
                    [-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                    [0, 0]
                ])
    complex_type = np.array([
                    [-1, -1],
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                    [0, 0]
                ])

    turn_type = np.array([
                    [-1, 1],
                    [1, -1],
                    [1, 1],
                    [0, 0]
                ])

    jump_type_1 = np.array([
                    [-1, -1],
                    [-1, 1],
                    [1, -1],
                    [0, 0]
                ])

    jump_type_2 = np.array([
                    [-1, -1],
                    [-1, 1],
                    [1, -1],
                    [0, 0]
                ])
    Action = np.zeros([17,
            action_type.shape[-1] + simple_type.shape[-1] + complex_type.shape[-1] + \
                    turn_type.shape[-1] + jump_type_1.shape[-1] + jump_type_2.shape[-1]
                    ])
    action_types = np.zeros([17, 2])
    for i in range(17):
        if i in [0,1,2,3]:
            Action[i] = np.concatenate((
                    action_type[0],
                    simple_type[i],
                    complex_type[-1],
                    turn_type[-1],
                    jump_type_1[-1],
                    jump_type_2[-1]
                    ), axis=-1)
            action_types[i] = [0, i]
                    
        elif i in [4,5,6,7]:
            Action[i] = np.concatenate((
                    action_type[1],
                    simple_type[-1],
                    complex_type[i-4],
                    turn_type[-1],
                    jump_type_1[-1],
                    jump_type_2[-1]
                    ), axis=-1)
            action_types[i] = [1, i-4]

        elif i in [8, 9, 10]:
            Action[i] = np.concatenate((
                    action_type[2],
                    simple_type[-1],
                    complex_type[-1],
                    turn_type[i-8],
                    jump_type_1[-1],
                    jump_type_2[-1]
                    ), axis=-1)
            action_types[i] = [2, i-8]

        elif i in [11, 12, 13]:
            Action[i] = np.concatenate((
                    action_type[3],
                    simple_type[-1],
                    complex_type[-1],
                    turn_type[-1],
                    jump_type_1[i-11],
                    jump_type_2[-1]
                    ), axis=-1)
            action_types[i] = [3, i-11]
    
        elif i in [14, 15, 16]:
            Action[i] = np.concatenate((
                    action_type[4],
                    simple_type[-1],
                    complex_type[-1],
                    turn_type[-1],
                    jump_type_1[-1],
                    jump_type_2[i-14]
                    ), axis=-1)
            action_types[i] = [4, i-14]
    return Action, action_types

class ReseedWrapper(gym.core.Wrapper):
    """
    Wrapper to always regenerate an environment with the same set of seeds.
    This can be used to force an environment to always keep the same
    configuration when reset.
    """

    def __init__(self, env, seeds=[0], seed_idx=0):
        self.seeds = list(seeds)
        self.seed_idx = seed_idx
        super().__init__(env)

    def reset(self, **kwargs):
        seed = self.seeds[self.seed_idx]
        self.seed_idx = (self.seed_idx + 1) % len(self.seeds)
        self.env.seed(seed)
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return obs, reward, done, info

class ActionBonus(gym.core.Wrapper):
    """
    Wrapper which adds an exploration bonus.
    This is a reward to encourage exploration of less
    visited (state,action) pairs.
    """

    def __init__(self, env):
        self.__dict__.update(vars(env))  # Pass values to super wrapper
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        env = self.unwrapped
        tup = (tuple(env.agent_pos), env.agent_dir, action)

        # Get the count for this (s,a) pair
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this (s,a) pair
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class StateBonus(gym.core.Wrapper):
    """
    Adds an exploration bonus based on which positions
    are visited on the grid.
    """

    def __init__(self, env):
        self.__dict__.update(vars(env))  # Pass values to super wrapper
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Tuple based on which we index the counts
        # We use the position after an update
        env = self.unwrapped
        tup = (tuple(env.agent_pos))

        # Get the count for this key
        pre_count = 0
        if tup in self.counts:
            pre_count = self.counts[tup]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[tup] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

class ImgObsWrapper(gym.core.ObservationWrapper):
    """
    Use the image as the only observation output, no language/mission.
    """

    def __init__(self, env):
        self.__dict__.update(vars(env))  # Pass values to super wrapper
        super().__init__(env)

        self.observation_space = env.observation_space.spaces['image']

    def observation(self, obs):
        return obs['image']

class FullyObsWrapper(gym.core.ObservationWrapper):
    """
    Fully observable gridworld using a compact grid encoding

    NOTE: this is more of a hijacked wrapper for the purpose of this project.
    It does much more than just make the enivornment fully observable
    """

    def __init__(self, env, args, dual_lava=False):
        self.__dict__.update(vars(env))  # Pass values to super wrapper
        super().__init__(env)

        self.trajectory_len = args.trajectory_len
        self.option_penalty = args.option_penalty
        self.args = args
        self.flatten = args.grid_flatten

        if self.flatten:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.env.width * self.env.height + \
                        (self.env.width * self.env.height if self.args.grid_agent_pos else 2) + \
                        2 * self.args.grid_append_dig_availability,),
                dtype='uint8'
            )
        else:
            self.observation_space = spaces.Box(
                low=0,
                high=255,
                shape=(self.env.width, self.env.height, 1),
                dtype='uint8'
            )

        self.overall_aval_actions = args.overall_aval_actions
        self.play = args.grid_playing

        self.action_bank = args.action_bank
        self.action_random_sample = args.action_random_sample

        if not dual_lava:
            if self.play:
                self.aval_actions = self.overall_aval_actions
            elif args.fixed_action_set:
                if args.action_set_size is None:
                    args.action_set_size = len(self.overall_aval_actions)
                    args.training_action_set_size = args.action_set_size
                self.aval_actions = np.array(list(self.overall_aval_actions)[:args.action_set_size])
            elif self.action_random_sample:
                self.sample_aval_actions()
            else:
                self.aval_actions = args.overall_aval_actions
            args.aval_actions = self.aval_actions
            self.fixed_action_set = args.fixed_action_set

            # 0 = right
            # 1 = down
            # 2 = left
            # 3 = up
            self.action_space = spaces.Discrete(len(self.aval_actions))

        self.env.render_info = args.render_info_grid
        self.render_info = args.render_info_grid


    def get_aval(self):
        return self.aval_actions

    def sample_aval_actions(self):
        if self.action_random_sample:
            self.aval_actions = self._sample_ac(list(self.args.overall_aval_actions))


    def _sample_ac(self, all_actions):
        rng = np.random.RandomState()
        if self.args.gt_clusters:
            raise ValueError('Not implemented!')
        else:
            return rng.choice(all_actions, self.args.action_set_size, replace=False)

        group_keys = list(ac_groups.keys())

        set_arr = []
        while(len(set_arr) < self.args.action_set_size):
            if self.args.half_tools and self.args.half_tool_ratio is not None:
                group_keys_samples = random.sample(
                    group_keys, int(self.args.half_tool_ratio * len(group_keys)))
            else:
                group_keys_samples = group_keys[:]
            type_selections = rng.choice(group_keys_samples, self.args.action_set_size, replace=True)
            for type_sel in type_selections:
                for trial in range(10):
                    idx = rng.choice(ac_groups[type_sel])
                    if idx not in set_arr:
                        break
                set_arr.append(idx)
                if len(set_arr) == self.args.action_set_size:
                    break
        return np.array(set_arr)

    def __str__(self):
        return str(self.env)

    def step(self, action):
        if isinstance(action, np.ndarray):
            assert len(action) == 1
            action = action[0]
        if not self.play:
            action_i = self.aval_actions[action]
        else:
            action_i = action
        action_seq = self.action_bank[action_i]

        if self.trajectory_len is not None:
            assert self.trajectory_len > len(action_seq)
            states = [self.env.agent_pos]
            actions = []

        total_reward = 0.0

        done = False

        all_frames = []

        for small_action in action_seq:
            if done:
                break

            actual_action_seq = [small_action]
            if small_action == 4:
                actual_action_seq = [0, 1]
            elif small_action == 5:
                actual_action_seq = [1, 2]
            elif small_action == 6:
                actual_action_seq = [2, 3]
            elif small_action == 7:
                actual_action_seq = [3, 0]

            for actual_action in actual_action_seq:
                if not (done and self.trajectory_len is None):
                    # Default facing right
                    self.env.agent_dir = actual_action
                    obs, reward, done, info = super().step(self.env.actions.forward)

                    total_reward += reward

                if self.render_info:
                    all_frames.append(info['frame'])

            if self.trajectory_len is not None:
                states.append(self.env.agent_pos)
                actions.append(small_action)

        # If we get done anywhere on the trajectory count this episode as
        # finished. We do not finish when collecting trajectories

        if self.trajectory_len is not None:
            if len(states) < self.trajectory_len:
                # Repeat last states
                extra_states = np.repeat(
                    np.expand_dims(states[-1], axis=0),
                    self.trajectory_len - len(states), axis=0)
                info['states'] = np.vstack([np.vstack(states), extra_states])
            else:
                info['states'] = np.vstack(states)

            # Last action is simply -1
            if len(actions) < self.trajectory_len:
                extra_actions = -np.ones([self.trajectory_len - len(actions),1],
                    dtype=type(small_action))
                info['actions'] = np.vstack([np.vstack(actions), extra_actions])
            else:
                info['actions'] = np.vstack(actions)
            #info['states'] = self.norm_data(info['states'])
            #info['actions'] = self.norm_data(info['actions'])
        if done and not self.play:
            if  not self.fixed_action_set:
                self.sample_aval_actions()

        # total_reward -= self.option_penalty
        # if total_reward > 0 and not done and self.trajectory_len is None:
        #     raise ValueError('Not possible', total_reward)

        # info['aval'] = self.aval_actions
        if self.render_info:
            info['frames'] = all_frames

        # only return the final observation, no intermediate observations
        return obs, total_reward, done, info

    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
            OBJECT_TO_IDX['agent'],
            COLOR_TO_IDX['red'],
            env.agent_dir
        ])
        # When reconverting, just set colors for wall as 5, floor as 0 and goal as 1
        # Also need to set is_locked (i.e. state = 2) for where the agent is located.
        full_grid = np.expand_dims(full_grid[:, :, 0], -1)
        if self.flatten:
            return full_grid.flatten()
        else:
            return full_grid


    @staticmethod
    def norm_data(obs, maxx = 255.):
        norm = (obs - 128.)/maxx
        return norm

    def render_obs(self, input_obs, opt_id):
        text = str(self.action_bank[opt_id])
        frames = []
        for input_ob in input_obs:
            self.env.agent_pos = [int(input_ob[0]), int(input_ob[1])]
            frame = self.render('rgb_array')
            frame = cv2.putText(frame, text, (1, 8), cv2.FONT_HERSHEY_SIMPLEX,
                            0.25, (0, 0, 0), 1, cv2.LINE_AA)

            frames.append(frame)
        frames = np.array(frames)
        return frames

class FullyObsJumpWrapper(FullyObsWrapper):
    """
    Fully observable gridworld using a compact grid encoding

    NOTE: this is more of a hijacked wrapper for the purpose of this project.
    It does much more than just make the enivornment fully observable
    """

    def __init__(self, env, args):
        super().__init__(env, args, dual_lava=True)

        self.action_embs, self.action_types = generate_jump_actions()
        self.overall_aval_actions = np.arange(len(self.action_types))
        self.args.overall_aval_actions = self.overall_aval_actions

        self.action_bank = None
        # self.action_bank = args.action_bank

        # We always use GT Embs
        assert not self.play

        if args.fixed_action_set:
            if args.action_set_size is None:
                args.action_set_size = len(self.overall_aval_actions)
                args.training_action_set_size = args.action_set_size
            self.aval_actions = np.array(list(self.overall_aval_actions)[:args.action_set_size])
        elif self.action_random_sample:
            self.sample_aval_actions()
        else:
            self.aval_actions = args.overall_aval_actions
        args.aval_actions = self.aval_actions

        self.fixed_action_set = args.fixed_action_set

        # 0 = right
        # 1 = down
        # 2 = left
        # 3 = up
        self.action_space = spaces.Discrete(len(self.aval_actions))
        if args.grid_dig_env:
            self.env.grid_dig_env = True


    def _sample_ac(self, all_actions):
        '''
            Have 0-7 always
            Other 2 actions: Sample two from [9, 10, 11, 12]
        '''
        rng = np.random.RandomState()
        if self.args.grid_remove_diagonal:
            sampled_actions = all_actions[:4]
            sampled_actions.extend(all_actions[8:self.args.grid_available_upto])
        else:
            sampled_actions = all_actions[:self.args.grid_available_upto]

        sampled_actions.extend(
                rng.choice(all_actions[self.args.grid_available_upto:], 2, replace=False)
                )

        return np.array(sampled_actions)


    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        self.env.grid.get(self.env.width // 2 - 1, 0).available = (11 in self.aval_actions)
        self.env.grid.get(self.env.width // 2 + 1, 0).available = (12 in self.aval_actions)
        return self.observation(observation)


    def step(self, action):
        if isinstance(action, np.ndarray):
            assert len(action) == 1
            action = action[0]
        if not self.play:
            action_i = self.aval_actions[action]
        else:
            action_i = action

        a_type, a_idx = np.asarray(self.action_types[action_i], dtype=np.int32)

        total_reward = 0.0
        done = False
        all_frames = []

        if a_type == 0:
            self.env.agent_dir = a_idx
            obs, reward, done, info = super(FullyObsWrapper, self).step(self.env.actions.forward)
            total_reward += reward
            if self.render_info:
                all_frames.append(info['frame'])

        elif a_type == 1:
            for (j, actual_action) in enumerate(diagonal_action_dict[a_idx]):
                self.env.agent_dir = actual_action
                self.env.just_move = (j == 0)
                obs, reward, done, info = super(FullyObsWrapper, self).step(
                        self.env.actions.forward
                        )
                self.env.just_move = False
                total_reward += reward
                if self.render_info:
                    all_frames.append(info['frame'])

        elif a_type == 2:
            obs, reward, done, info = super(FullyObsWrapper, self).step(a_idx)
            total_reward += reward
            if self.render_info:
                all_frames.append(info['frame'])
        elif a_type == 3:
            if self.args.grid_dig_env:
                obs, reward, done, info = super(FullyObsWrapper, self).step(self.env.actions.dig_lava)
                total_reward += reward
                if self.render_info:
                    all_frames.append(info['frame'])
            else:
                for j in range(2):
                    self.env.agent_dir = 0
                    self.env.jump_lava = ('lava' if j == 0 else '')
                    obs, reward, done, info = super(FullyObsWrapper, self).step(
                            self.env.actions.forward
                            )
                    self.env.jump_lava = ''
                    total_reward += reward
                    if self.render_info:
                        all_frames.append(info['frame'])
                    if done: break
        elif a_type == 4:
            if self.args.grid_dig_env:
                obs, reward, done, info = super(FullyObsWrapper, self).step(self.env.actions.dig_pinklava)
                total_reward += reward
                if self.render_info:
                    all_frames.append(info['frame'])
            else:
                for j in range(2):
                    self.env.agent_dir = 0
                    self.env.jump_lava=('pinklava' if j == 0 else '')
                    obs, reward, done, info = super(FullyObsWrapper, self).step(
                            self.env.actions.forward
                            )
                    self.env.jump_lava = ''
                    total_reward += reward
                    if self.render_info:
                        all_frames.append(info['frame'])
                    if done: break

        if done and not self.fixed_action_set:
            self.sample_aval_actions()


        if self.render_info:
            info['frames'] = all_frames


        # only return the final observation, no intermediate observations
        return obs, total_reward, done, info


    def observation(self, obs):
        env = self.unwrapped
        full_grid = env.grid.encode()
        if not self.flatten:
            full_grid[env.agent_pos[0]][env.agent_pos[1]] = np.array([
                OBJECT_TO_IDX['agent'],
                COLOR_TO_IDX['red'],
                env.agent_dir
            ])
        # When reconverting, just set colors for wall as 5, floor as 0 and goal as 1
        # Also need to set is_locked (i.e. state = 2) for where the agent is located.
        full_grid = np.expand_dims(full_grid[:, :, 0], -1)
        # Remove Available indicators from state information (they are fixed, but distracting)
        full_grid[:, 0] = 2
        if self.args.grid_agent_pos:
            agent_grid = np.zeros(full_grid.shape, dtype=np.uint8)
            agent_grid[env.agent_pos[0]][env.agent_pos[1]] = 1
        if self.flatten:
            if self.args.grid_append_dig_availability:
                if self.args.grid_agent_pos:
                    flattened = np.concatenate([full_grid.flatten(), agent_grid.flatten(),
                        np.array([11 in self.aval_actions, 12 in self.aval_actions])], axis=-1)
                else:
                    flattened = np.concatenate([
                        full_grid.flatten(),
                        np.array(env.agent_pos, dtype=np.uint8),
                        np.array([
                            11 in self.aval_actions,
                            12 in self.aval_actions])
                        ], axis=-1)
            else:
                if self.args.grid_agent_pos:
                    flattened = np.concatenate([full_grid.flatten(), agent_grid.flatten()], axis=-1)
                else:
                    flattened = np.concatenate([
                        full_grid.flatten(),
                        np.array(env.agent_pos, dtype=np.uint8)
                        ], axis=-1)
            return flattened
        else:
            return full_grid

class FlatObsWrapper(gym.core.ObservationWrapper):
    """
    Encode mission strings using a one-hot scheme,
    and combine these with observed images into one flat array
    """

    def __init__(self, env, maxStrLen=96):
        self.__dict__.update(vars(env))  # Pass values to super wrapper
        super().__init__(env)

        self.maxStrLen = maxStrLen
        self.numCharCodes = 27

        imgSpace = env.observation_space.spaces['image']
        imgSize = reduce(operator.mul, imgSpace.shape, 1)

        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(1, imgSize + self.numCharCodes * self.maxStrLen),
            dtype='uint8'
        )

        self.cachedStr = None
        self.cachedArray = None

    def observation(self, obs):
        image = obs['image']
        mission = obs['mission']

        # Cache the last-encoded mission string
        if mission != self.cachedStr:
            assert len(mission) <= self.maxStrLen, 'mission string too long ({} chars)'.format(len(mission))
            mission = mission.lower()

            strArray = np.zeros(shape=(self.maxStrLen, self.numCharCodes), dtype='float32')

            for idx, ch in enumerate(mission):
                if ch >= 'a' and ch <= 'z':
                    chNo = ord(ch) - ord('a')
                elif ch == ' ':
                    chNo = ord('z') - ord('a') + 1
                assert chNo < self.numCharCodes, '%s : %d' % (ch, chNo)
                strArray[idx, chNo] = 1

            self.cachedStr = mission
            self.cachedArray = strArray

        obs = np.concatenate((image.flatten(), self.cachedArray.flatten()))

        return obs

class AgentViewWrapper(gym.core.Wrapper):
    """
    Wrapper to customize the agent's field of view.
    """

    def __init__(self, env, agent_view_size=7):
        self.__dict__.update(vars(env))  # Pass values to super wrapper
        super(AgentViewWrapper, self).__init__(env)

        # Override default view size
        env.unwrapped.agent_view_size = agent_view_size

        # Compute observation space with specified view size
        observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(agent_view_size, agent_view_size, 3),
            dtype='uint8'
        )

        # Override the environment's observation space
        self.observation_space = spaces.Dict({
            'image': observation_space
        })

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        return self.env.step(action)
