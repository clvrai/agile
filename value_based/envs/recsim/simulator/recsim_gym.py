# coding=utf-8
# coding=utf-8
# Copyright 2019 The RecSim Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A wrapper for using Gym environment."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
import numpy as np
from gym import spaces

from value_based.envs.recsim.simulator.environment import MultiUserEnvironment
from value_based.commons.args import SKIP_TOKEN
from policy_based.envs.action_env import ActionEnv


def _dummy_metrics_aggregator(responses, metrics, info):
    del responses  # Unused.
    del metrics  # Unused.
    del info  # Unused.
    return None


def _dummy_metrics_writer(metrics, add_summary_fn):
    del metrics  # Unused.
    del add_summary_fn  # Unused.
    return None


class RecSimGymEnv(gym.Env):
    """Class to wrap recommender system environment to gym.Env.

    Attributes:
        game_over: A boolean indicating whether the current game has finished
        action_space: A gym.spaces object that specifies the space for possible actions.
        observation_space: A gym.spaces object that specifies the space for possible observations.
    """

    def __init__(self,
                 raw_environment,
                 reward_aggregator,
                 metrics_aggregator=_dummy_metrics_aggregator,
                 metrics_writer=_dummy_metrics_writer):
        """Initializes a RecSim environment conforming to gym.Env.

        Args:
          raw_environment: A recsim recommender system environment.
          reward_aggregator: A function mapping a list of responses to a number.
          metrics_aggregator: A function aggregating metrics over all steps given
            responses and response_names.
          metrics_writer:  A function writing final metrics to TensorBoard.
        """
        self._env = raw_environment
        self._reward_aggregator = reward_aggregator
        self._metrics_aggregator = metrics_aggregator
        self._metrics_writer = metrics_writer

    @property
    def environment(self):
        """Returns the recsim recommender system environment."""
        return self._env

    @property
    def game_over(self):
        return False

    # @property
    # def action_space(self):
    #     """Returns the action space of the environment.
    #
    #     Each action is a vector that specified document slate. Each element in the
    #     vector corresponds to the index of the document in the candidate set.
    #     """
    #     action_space = spaces.MultiDiscrete(self._env.num_allItems * np.ones((self._env.slate_size,)))
    #     if isinstance(self._env, MultiUserEnvironment):
    #         action_space = spaces.Tuple([action_space] * len(self._env.user_model))
    #     return action_space

    # @property
    # def observation_space(self):
    #     """Returns the observation space of the environment.
    #
    #     Each observation is a dictionary with three keys `user`, `doc` and
    #     `response` that includes observation about user state, document and user
    #     response, respectively.
    #     """
    #     if isinstance(self._env, MultiUserEnvironment):
    #         user_obs_space = self._env.user_model[0].observation_space()
    #         resp_obs_space = self._env.user_model[0].response_space()
    #         user_obs_space = spaces.Tuple([user_obs_space] * len(self._env.user_model))
    #         resp_obs_space = spaces.Tuple([resp_obs_space] * len(self._env.user_model))
    #
    #     if isinstance(self._env, SingleUserEnvironment):
    #         user_obs_space = self._env.user_model.observation_space()
    #         resp_obs_space = self._env.user_model.response_space()
    #
    #     return spaces.Dict({
    #         'user': user_obs_space,
    #         'doc': self._env.candidate_set.observation_space(),
    #         'response': resp_obs_space
    #     })

    def step(self, action):
        """Runs one timestep of the environment's dynamics.

        When end of episode is reached, you are responsible for calling `reset()`
        to reset this environment's state. Accepts an action and returns a tuple
        (observation, reward, done, info).

        Args:
            action (np.ndarray): An action provided by the environment

        Returns:
            A four-tuple of (observation, reward, done, info) where:
                observation (object): agent's observation that include
                    1. User's state features
                    2. Document's observation
                    3. Observation about user's slate responses.
                reward (float) : The amount of reward returned after previous action
                done (boolean): Whether the episode has ended, in which case further
                    step() calls will return undefined results
                info (dict): Contains responses for the full slate for debugging/learning.
        """
        user_obs, doc_obs, responses, done, info = self._env.step(action)
        if_ppo_recsim = self._env._document_sampler.feature_dist._args.get('env_name').startswith('RecSim')
        if isinstance(self._env, MultiUserEnvironment):
            # Aggregate the responses of active users at the time-step
            all_responses = tuple(
                tuple(response.create_observation() for response in user_response) for user_response in responses
            )
            # TODO: Repetitive... Move this in self._reward_aggregator in which you iterate through all responses
            # Get the clicked items of active users
            clicked_items = list()
            inslate_position = list()
            if if_ppo_recsim:
                click_mask = [x['click'] for i, x in enumerate(all_responses[0])]
                if np.max(click_mask) == 0:  # if no click happened in a slate
                    clicked_items += [SKIP_TOKEN]
                else:  # if a click happened in a slate
                    clicked_items += [action[np.argmax(click_mask)]]
            else:
                for user_id in range(len(info["prev_activeIds"])):
                    click_mask = [x["click"] for i, x in enumerate(all_responses[user_id])]
                    if np.max(click_mask) == 0:  # if no click happened in a slate
                        clicked_items += [SKIP_TOKEN]
                        inslate_position += [SKIP_TOKEN]
                    else:  # if a click happened in a slate
                        clicked_items += [action[user_id][np.argmax(click_mask)]]
                        inslate_position += [np.argmax(click_mask)]
        else:  # single user environment
            all_responses = tuple(
                response.create_observation() for response in responses
            )
            clicked_items = [i for i, x in enumerate(all_responses) if x["click"]]

        # next obs
        obs = dict(user=user_obs, doc=doc_obs, response=all_responses)

        # extract rewards from responses
        reward = self._reward_aggregator(responses)
        info["response"] = all_responses
        info["gt_items"] = np.asarray(clicked_items)
        info["click_inslate_position"] = np.asarray(inslate_position)
        if if_ppo_recsim:
            return obs, reward[0], done, info
        else:
            return obs, reward[:, None], done, info

    def reset(self):
        user_obs, doc_obs = self._env.reset()
        self.reset_sampler()  # to refresh the random state of each component for each episode: Reproducibility purpose
        return dict(user=user_obs, doc=doc_obs, response=None)

    def reset_sampler(self):
        self._env.reset_sampler()

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError

    def seed(self, seed=None):
        np.random.seed(seed)

    @property
    def active_users(self):
        return self._env.active_users

    @property
    def active_userIds(self):
        return self._env.active_userIds

    @property
    def categories(self):
        return self._env.current_documents

    def set_if_use_log(self, flg):
        """ compatibility purpose """
        pass
