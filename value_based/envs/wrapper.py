""" Design reference: https://github.com/openai/gym/tree/master/gym/wrappers """
import os
import gym
import numpy as np
from gym import spaces
from copy import deepcopy
from collections import Counter

from value_based.commons.args import ENV_RANDOM_SEED
from value_based.embedding.base import BaseEmbedding
from value_based.commons.args import SKIP_TOKEN, HISTORY_SIZE
from value_based.commons.metrics import Metrics, METRIC_NAMES
from value_based.commons.plot_embedding import plot_embedding
from value_based.commons.observation import ObservationFactory
from value_based.commons.utils import logging


class ObsInfoWrapper(gym.Wrapper):
    """ Wrapper class to return the simplified observation and info in numpy array """

    def __init__(self, env):
        super(ObsInfoWrapper, self).__init__(env=env)
        self._env = env

    def step(self, action):
        obs, reward, done, info = self._env.step(action=action)
        return np.asarray(obs["user"]), reward, done, info

    def reset(self, **kwargs):
        obs = self._env.reset()
        return np.asarray(obs["user"])


class ItemEmbeddingWrapper(gym.Wrapper):
    """ To hold the item embedding in the env """

    def __init__(self, env, if_debug: bool = False):
        super(ItemEmbeddingWrapper, self).__init__(env=env)
        self._env = env
        # Get the ground truth item-embedding for all the items(both train/test) of RecSim
        """ === Note
            Sampling item embeddings is assumed to be done by now.
            So, we can just get all the item embeddings in this wrapper.
        """
        _embedding = np.asarray([i.create_observation() for i in env.environment.candidate_set.get_all_documents()])
        self._if_debug = if_debug
        if self._if_debug:
            logging("ItemEmbeddingWrapper>> Original Embedding: {}".format(_embedding.shape))
        self._item_embedding = BaseEmbedding()
        self._item_embedding.load(embedding=_embedding)

    @property
    def item_embedding(self):
        return self._item_embedding


class HistorySequenceWrapper(gym.Wrapper):
    """ To maintain the history sequences of users in the env """

    def __init__(self, env, batch_size: int = 32, if_ppo_recsim=False):
        super(HistorySequenceWrapper, self).__init__(env=env)
        self._env = env
        self._batch_size = batch_size
        self._obs = ObservationFactory(batch_size=self._batch_size)
        if if_ppo_recsim: raise NotImplementedError
        self.if_ppo_recsim = if_ppo_recsim

    def step(self, action):
        """ Transforms the RecSim's original state into the sequence of user-action based obs """
        _obs, reward, done, info = self._env.step(action)

        # Update the history sequence based on user actions
        self._obs.update_history_seq(a_user=info["gt_items"], active_user_mask=info["cur_active_user_mask"])
        return self._obs, reward, done, info

    def reset(self, **kwargs):
        obs = self._env.reset()
        self._obs = ObservationFactory(batch_size=self._batch_size)
        return self._obs


class RecSimMDPWrapper(gym.Wrapper):
    """ MDP based RecSim """

    def __init__(self, env, batch_size: int = 32, if_ppo_recsim=False):
        super(RecSimMDPWrapper, self).__init__(env=env)
        self._env = env
        self._batch_size = batch_size
        self.if_ppo_recsim = if_ppo_recsim

    def step(self, action):
        """ MDP API; State is used instead of obs """
        state, reward, done, info = self._env.step(action)
        # sorry for confusing var name...
        if self.if_ppo_recsim:
            obs = state.squeeze(0)
        else:
            obs = ObservationFactory(batch_size=self._batch_size, if_mdp=True)
            obs.load_state(state=state)
        return obs, reward, done, info

    def reset(self, **kwargs):
        state = self._env.reset()
        if self.if_ppo_recsim:
            obs = state.squeeze(0)
        else:
            obs = ObservationFactory(batch_size=self._batch_size, if_mdp=True)
            obs.load_state(state=state)
        return obs


class TimeLimit(gym.Wrapper):
    """ Ref: https://github.com/openai/gym/blob/master/gym/wrappers/time_limit.py """

    def __init__(self, env, max_episode_steps=None):
        super(TimeLimit, self).__init__(env=env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        if self.env.spec is not None:
            self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert self._elapsed_steps is not None, "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info['TimeLimit.truncated'] = not done
            done = True
        return observation, reward, done, info

    def reset(self, **kwargs):
        self._elapsed_steps = 0
        return self.env.reset(**kwargs)


class DoneWrapper(gym.Wrapper):
    def __init__(self, env, args: dict):
        super(DoneWrapper, self).__init__(env=env)
        self._args = args

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        if self._args.get("if_recsim", False):
            if self._args["recsim_if_no_fix_for_done"]:
                done = ~info["cur_active_user_mask"]
            else:
                if done:  # if we reached the max time-step in an episode
                    done = [done] * len(info["cur_active_user_mask"])
                else:
                    done = ~info["cur_active_user_mask"]
        else:
            done = [done] * self._args.get("batch_step_size")
        return obs, reward, np.asarray(done), info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class CompatibilityWrapper(gym.Wrapper):
    """ To maintain the compatibility across all the envs """

    def __init__(self, env, args: dict):
        super(CompatibilityWrapper, self).__init__(env=env)
        self._args = args
        self._if_recsim = args.get("env_name", "recsim") == "recsim" or args.get('env_name').startswith('RecSim')
        self._item_categories = self._get_allItemCategory()  # (k: itemId, v: fullCategoryId)
        self._category_master_dict = self._get_category_master_dict()  # (k: fullCategoryId, v: list of itemIds)
        self._main_category_master_dict = self._get_main_category_master_dict()
        self._id2category_dict = self._get_id2category_dict()
        self._category_mat = np.eye(self._args.get("recsim_num_categories", 10))
        self._prep_obs_space()
        self._prep_action_space()
        self._get_fns()

    def _prep_obs_space(self):
        # Prep the observation_space
        if not self._args.get("recsim_if_mdp", False):
            # batch_step_size x history_size x dim_item
            user_obs_spec = spaces.Box(low=0.0,
                                       high=1.0,
                                       shape=(HISTORY_SIZE, self._args.get("recsim_num_categories", 10)),
                                       dtype=np.float32)
        else:
            # batch_step_size x dim_item
            user_obs_spec = spaces.Box(low=0.0,
                                       high=1.0,
                                       shape=[self._args.get("recsim_num_categories", 10)],
                                       dtype=np.float32)

        self.observation_space = user_obs_spec

    def _prep_action_space(self):
        _base_action_space = spaces.MultiDiscrete(
            self._args.get("num_candidates", 16) * np.ones((self._args.get("slate_size", 16),))
        )
        self.action_space = _base_action_space

    @property
    def main_category_master_dict(self):
        return self._main_category_master_dict

    @property
    def id2category_dict(self):  # todo: remove after debugging
        return self._id2category_dict

    @property
    def item_categories(self):  # todo: remove after debugging
        return self._item_categories

    def _get_fns(self):
        if self._if_recsim:
            self._get_fullCategory = self.env.environment._candidate_set.get_fullCategory
            self._get_mainCategory = self.env.environment._candidate_set.get_mainCategory
        else:
            self._get_fullCategory = self.env.cc.get_fullCategory
            self._get_mainCategory = self.env.cc.get_mainCategory

    def _get_main_category_master_dict(self):
        # Dict; (k: mainCategoryId, v: list of itemIds)
        if self._if_recsim:
            main_category_master_dict = self.env.environment.main_category_master_dict
        else:
            main_category_master_dict = self.env.cc.main_category_master_dict
        return main_category_master_dict

    def _get_id2category_dict(self):
        # Dict; (k: itemId, v: categoryId)
        if self._if_recsim:
            id2category_dict = self.env.environment.id2category_dict
        else:
            id2category_dict = None
            # raise ValueError
        return id2category_dict

    def _get_category_master_dict(self):
        # Dict; (k: fullCategoryId, v: list of itemIds)
        if self._if_recsim:
            category_master_dict = self.env.environment.category_master_dict
        else:
            category_master_dict = self.env.cc.category_master_dict
        return category_master_dict

    def _get_allItemCategory(self):
        """ Returns the dict of categories of items

        Returns:
            categories (dict): (k: itemId, v: fullCategoryId)
        """
        # Dict; (k: itemId, v: fullCategoryId)
        if self._if_recsim:
            categories = self.env.environment.categories
        else:
            categories = self.env.cc.categories
        return categories

    @property
    def if_eval(self):
        if self._if_recsim:
            return self.env.environment._if_eval
        else:
            return self.env.if_eval

    def set_if_eval(self, flg: bool):
        """ Used to define the sampling dist of user-interest """
        if self._if_recsim:
            self.env.environment.set_if_eval(flg)
        else:
            self.env.set_if_eval(flg)

    def set_if_eval_train_or_test(self, train_or_test: str):
        """ Used to define the sampling dist of user-interest """
        if self._if_recsim:
            self.env.environment.set_if_eval_train_or_test(train_or_test)
        else:
            self.env.set_if_eval_train_or_test(train_or_test)

    def get_state(self):
        """ In RecSim, we have access to the internal state of users but not in other envs! """
        if self._if_recsim:
            _user_state_list = [user.get_state() for user in self.env.environment.user_model]
            _user_state_dict = {k: np.asarray([dic[k] for dic in _user_state_list]) for k in _user_state_list[0].keys()}
            return _user_state_dict
        else:
            return None

    def get_allItemCategory(self):
        """ Returns the dict of categories of items

        Returns:
            categories (dict): (k: itemId, v: fullCategoryId)
        """
        return self._item_categories

    def get_slate_category(self, slate):
        """ Returns the dict of categories of items

        Returns:
            category_mat (np.ndarray): num_categories x num_subcategories or num_categories x 1
        """
        if self._if_recsim:
            return self.env.environment._candidate_set.get_slate_category(slate=slate)
        else:
            return self.env.cc.get_slate_category(slate=slate)

    def get_mainCategory_of_items(self, arr_items: np.ndarray):
        """
        Returns:
            category_mat (np.ndarray): num_categories x num_subcategories or num_categories x 1
        """
        res = list()
        if len(np.array(arr_items).shape) == 0:
            array_items = np.array([arr_items])
        else:
            array_items = arr_items
        for i in np.array(array_items):
            if self._if_recsim:
                res.append(self.env.environment._candidate_set.get_mainCategory(itemId=i))
            else:
                res.append(self.env.cc.get_mainCategory(itemId=i))
        return np.asarray(res)

    def get_fullCategory_of_items(self, arr_items: np.ndarray):
        """
        Returns:
            category_mat (list): list of full category of items
        """
        res = list()
        if len(np.array(arr_items).shape) == 0:
            array_items = np.array([arr_items])
        else:
            array_items = arr_items
        for i in np.array(array_items):
            if self._if_recsim:
                res.append(self.env.environment._candidate_set.get_fullCategory(itemId=i))
            else:
                res.append(self.env.cc.get_fullCategory(itemId=i))
        return res


class NewItemWrapper(gym.Wrapper):
    """ To maintain the list of items available at the specific time in the env """

    def __init__(self,
                 env,
                 num_categories: int = 20,
                 num_candidates: int = 500,
                 num_candidates_sp: int = 2,
                 num_candidates_correct_normal: int = 2,
                 resampling_method: str = "random",
                 resampling_method_sp: str = "random",
                 if_resampling_at_ts: bool = False,
                 if_special_items: bool = False,
                 if_special_items_flg: bool = False,
                 if_recsim: bool = True):
        super(NewItemWrapper, self).__init__(env=env)
        self._num_candidates = num_candidates
        self._num_candidates_sp = num_candidates_sp
        self._num_candidates_correct_normal = num_candidates_correct_normal
        self._num_categories = num_categories
        self._resampling_method = resampling_method
        self._resampling_method_sp = resampling_method_sp
        self._if_resampling_at_ts = if_resampling_at_ts
        self._if_special_items = if_special_items
        self._if_special_items_flg = if_special_items_flg
        self._rng = np.random.RandomState(ENV_RANDOM_SEED)
        self._if_recsim = if_recsim

        if self._if_recsim:
            self.__items_dict = env.environment.items_dict  # original dictionary of items for train / test
            self._items_dict = deepcopy(self.__items_dict)  # dictionary of items for train / test for candidate list

            # Link the dictionary of the candidate_items to the one in the env
            self.env.environment.set_candidate_items_dict(candidate_items_dict=self._items_dict)
        else:
            self.__items_dict = env.items_dict  # original dictionary of items for train / test
            self._items_dict = deepcopy(self.__items_dict)  # dictionary of items for train / test for candidate list

        # Split num_candidates equally
        _num_candidates = self._num_candidates
        num_candidates_each_category = int(self._num_candidates // self._num_categories)

        # Create the master of resampling items from either train / test
        self._resampling_master = {}
        for _category_id in range(self._num_categories - 1):
            _num_candidates -= num_candidates_each_category
            self._resampling_master[_category_id] = num_candidates_each_category
        self._resampling_master[_category_id + 1] = _num_candidates  # put the remaining items!

    def seed(self, seed=None):
        self._rng = np.random.RandomState(seed)
        np.random.seed(seed)

    def get_aval(self):
        return np.array(self._items_dict['train'])

    def get_test_aval(self):
        return np.array(self._items_dict['test'])

    def resample_items(self, flg: str = "train"):
        """ Resample the candidate list of items and this will be used in the agent's decision making

        Args:
            flg (str): specify the train/test set of items to resample
        """

        """ === Sample the normal items === """
        assert flg in ["train", "test"]

        if self._resampling_method == "random":
            # This doesn't account for the item category as such the candidate set might miss some category
            _candidate_list = self._rng.choice(a=self.__items_dict[flg], size=self._num_candidates, replace=False)
            # Concatenate the candidate-set of major category and other categories
            if self._if_special_items:
                # Uniform sampling across categories
                _candidate_list_sp = self._rng.choice(
                    a=self.__items_dict["train_sp" if flg == "train" else "test_sp"],
                    size=self._num_candidates_sp,
                    replace=False).tolist()
                _del_idx = self._rng.choice(a=len(_candidate_list), size=self._num_candidates_sp, replace=False)
                _candidate_list = np.delete(_candidate_list, _del_idx).tolist()
                _candidate_list += _candidate_list_sp
        elif self._resampling_method == "skewed":  # Non uniform item sampling
            # Skewed dist to sample focussing on a specific category for specificity Env
            if self.env._args.get("recsim_if_use_subcategory", False):
                _num_samples = min(
                    int(self._num_candidates * self.env._args.get("recsim_skew_ratio", 0.4)),
                    min([len(x) for x in self.env._main_category_master_dict[flg].values()])
                ) + self._rng.choice([-2, -1, 0])
                # Make sure that at least we can sample items from each sub-category
                _num_samples = max(_num_samples, int(self.env._args.get("recsim_num_subcategories", 3)))
            else:
                _num_samples = min(
                    int(self._num_candidates * self.env._args.get("recsim_skew_ratio", 0.4)),
                    min([len(x) for x in self.env._main_category_master_dict[flg].values()])
                ) + self._rng.choice([-2, -1, 0])

            assert _num_samples > 0
            # Sample a category
            _base_category = self._rng.choice(a=self._num_categories, size=1, replace=False)[0]

            # Sample the items from the selected category
            _candidate_list = self._rng.choice(a=self.env._main_category_master_dict[flg][_base_category],
                                               size=_num_samples,
                                               replace=False).tolist()

            if self.env._args.get("recsim_if_use_subcategory", False):
                # Make sure to have the complete subcategory in the major category
                _cat = self.env.get_fullCategory_of_items(arr_items=_candidate_list)
                if len(set(_cat)) == self.env._args.get("recsim_num_subcategories", 3):
                    _cat = sorted(list(set(_cat)))
                    for i in range(self.env._args["recsim_num_subcategories"]):
                        if f"{_base_category}_{i}" not in _cat:
                            # Sample and add an item from the missing subcategories to the candidate list
                            _candidate_list += self._rng.choice(
                                a=self.env._category_master_dict[flg][f"{_base_category}_{i}"],
                                size=1, replace=False).tolist()

            # Collect the itemIds from the other categories
            _others = list()
            for k, v in self.env._main_category_master_dict[flg].items():
                if k != _base_category:
                    _others += v

            # Sample items from the other categories
            __candidate_list = self._rng.choice(a=np.asarray(_others),
                                                size=self._num_candidates - len(_candidate_list),
                                                replace=False).tolist()

            # Concatenate the candidate-set of major category and other categories
            if self._if_special_items:
                """ === Sample the special items ===
                    We RANDOMLY sample the special items. In order to preserve the major category, we remove
                    the items from the other categories then concatenate the special items
                """
                if self._resampling_method_sp == "random":
                    # Uniform sampling across categories
                    _candidate_list_sp = self._rng.choice(
                        a=self.__items_dict["train_sp" if flg == "train" else "test_sp"],
                        size=self._num_candidates_sp,
                        replace=False).tolist()
                    _del_idx = self._rng.choice(
                        a=len(__candidate_list), size=self._num_candidates_sp, replace=False
                    )
                    __candidate_list = np.delete(__candidate_list, _del_idx).tolist()
                    _candidate_list = _candidate_list + __candidate_list + _candidate_list_sp
                elif self._resampling_method_sp == "category":
                    # Find the special item from the major main category selected above
                    _candidate_list_sp = self.env._main_category_master_dict[f"{flg}_sp"][_base_category]
                    _del_idx = self._rng.choice(a=len(__candidate_list), size=1, replace=False)
                    __candidate_list = np.delete(__candidate_list, _del_idx).tolist()
                    _candidate_list = _candidate_list + __candidate_list + _candidate_list_sp
            else:
                _candidate_list = _candidate_list + __candidate_list
        elif self._resampling_method == "category":
            """ Uniformly sample the equal number of items across categories
                so that the candidate set contains all the categories
            """
            _candidate_list = list()
            _category_master_dict = self.env._main_category_master_dict[flg]
            for _category_id in range(self._num_categories):
                _item_list = _category_master_dict[_category_id]
                _num_samples = self._resampling_master[_category_id]
                __candidate_list = self._rng.choice(a=_item_list, size=_num_samples, replace=False)
                _candidate_list += __candidate_list.tolist()

            # Concatenate the candidate-set of major category and other categories
            if self._if_special_items:
                # Uniform sampling across categories
                _candidate_list_sp = self._rng.choice(
                    a=self.__items_dict["train_sp" if flg == "train" else "test_sp"],
                    size=self._num_candidates_sp,
                    replace=False).tolist()
                _del_idx = self._rng.choice(a=len(_candidate_list), size=self._num_candidates_sp, replace=False)
                _candidate_list = np.delete(_candidate_list, _del_idx).tolist()
                _candidate_list += _candidate_list_sp
        elif self._resampling_method == "multi_common_category":
            """ Multiple categories are selected and sample items accordingly
                Diff to 'category' is that 'category' is to sample the equal num of items from each category
            """
            num_samples_for_each_category = self._num_candidates // self.env._args["recsim_resample_num_categories"]

            # Sample multiple common categories
            _base_categories = self._rng.choice(a=self._num_categories,
                                                size=self.env._args["recsim_resample_num_categories"],
                                                replace=False)

            # Sample the items from each common category
            _candidate_list = list()
            for _base_category in _base_categories:
                # Sample the items from the selected category
                _candidate_list += self._rng.choice(a=self.env._main_category_master_dict[flg][_base_category],
                                                    size=num_samples_for_each_category,
                                                    replace=False).tolist()

            if self._num_candidates > len(_candidate_list):
                # Collect the itemIds from the other categories
                _others = list()
                for k, v in self.env._main_category_master_dict[flg].items():
                    if k not in _base_categories:
                        _others += v

                # Sample items from the other categories
                __candidate_list = self._rng.choice(a=np.asarray(_others),
                                                    size=self._num_candidates - len(_candidate_list),
                                                    replace=False).tolist()
                _candidate_list += __candidate_list
        elif self._resampling_method == "noisy_multi_common_category":
            """ Multiple most common categories + some items from other categories """
            num_samples_for_each_category = (self._num_candidates - self.env._args["recsim_num_noisy_items"]) // \
                                            self.env._args["recsim_resample_num_categories"]

            # Sample multiple common categories
            _base_categories = self._rng.choice(a=self._num_categories,
                                                size=self.env._args["recsim_resample_num_categories"],
                                                replace=False)

            # Sample the items from each common category
            _candidate_list = list()
            for _base_category in _base_categories:
                # Sample the items from the selected category
                _candidate_list += self._rng.choice(a=self.env._main_category_master_dict[flg][_base_category],
                                                    size=num_samples_for_each_category,
                                                    replace=False).tolist()

            # Collect the itemIds from the other categories
            _others = list()
            for k, v in self.env._main_category_master_dict[flg].items():
                if k not in _base_categories:
                    _others += v

            # Sample items from the other categories
            __candidate_list = self._rng.choice(a=np.asarray(_others),
                                                size=self.env._args["recsim_num_noisy_items"],
                                                replace=False).tolist()
            _candidate_list += __candidate_list
        elif self._resampling_method == "half_special":
            # half: normal items, another half: special items

            # Sample the categories for correct pairings
            correct_pairing_category = self._rng.choice(a=self._num_categories,
                                                        size=self._num_candidates_sp,
                                                        replace=False).tolist()

            # Sample items with the correct pairings
            _candidate_list = list()
            for categoryId in correct_pairing_category:
                # Sample a normal item
                _candidate_list += self._rng.choice(a=self.env._main_category_master_dict[flg][categoryId],
                                                    size=self._num_candidates_correct_normal,
                                                    replace=False).tolist()

                # Get the corresponding special item
                _candidate_list += self.env._main_category_master_dict[f"{flg}_sp"][categoryId]

            # Sample the categories for incorrect pairings for normal items
            incorrect_pairing_category = [i for i in range(self._num_categories) if i not in correct_pairing_category]
            incorrect_sp = self._rng.choice(
                a=incorrect_pairing_category, size=self._num_candidates_sp, replace=False
            ).tolist()
            incorrect_normal = [i for i in incorrect_pairing_category if i not in incorrect_sp]

            # Sample items with the incorrect pairings
            num_samples = int(self._num_candidates - len(_candidate_list)) // 2

            # collect the special items first
            _items = list()
            for _categoryId in incorrect_sp:
                _items += self.env._main_category_master_dict[f"{flg}_sp"][_categoryId]

            if num_samples < len(_items):
                _candidate_list += self._rng.choice(a=np.asarray(_items), size=num_samples, replace=False).tolist()
            else:
                _candidate_list += _items

            # Collect the normal items to fill the space of the candidate-set
            _others = list()
            for k, v in self.env._main_category_master_dict[flg].items():
                if k in incorrect_normal:
                    _others += v

            # Sample items from the other categories
            _candidate_list += self._rng.choice(a=np.asarray(_others),
                                                size=self._num_candidates - len(_candidate_list),
                                                replace=False).tolist()
        else:
            raise ValueError

        """
        === Register the sampled candidate-set to the Env ===
        """
        self._items_dict[flg] = sorted(_candidate_list)

        if self._if_recsim:
            # Update the hard constraint based on the newly sampled candidate items; this happens in the core env side
            self.env.environment.set_hard_constraint_rule_dict(flg=flg)

    @property
    def items_dict(self):
        return self._items_dict

    @property
    def baseItems_dict(self):
        res = deepcopy(self.__items_dict)
        if self._if_special_items:
            res["all"] = res["all"] + res["all_sp"]
            res["train"] = res["train"] + res["train_sp"]
            res["test"] = res["test"] + res["test_sp"]
        return res

    def step(self, action):
        if self._if_resampling_at_ts:
            self.resample_items(flg="train")
            self.resample_items(flg="test")
        return self.env.step(action=action)

    def reset(self, **kwargs):
        self.resample_items(flg="train")
        self.resample_items(flg="test")
        return self.env.reset()

    def get_special_item_flg_vec(self, arr_items):
        if self._if_special_items:
            result = list()
            for i in arr_items:
                result.append(i in self.__items_dict["all_sp"])
            return np.asarray(result)
        else:
            return np.asarray([False] * self._num_candidates)

    def append_special_normal_flag_to_embedding(self, item_embedding: BaseEmbedding):
        """ Append the binary flag at the end of the item embedding
            - 1: special item
            - 0: normal item
        """
        if self._if_special_items_flg:
            special_flg = list()
            # from pudb import set_trace; set_trace()
            for itemId in range(item_embedding.shape[0]):
                special_flg.append(itemId in self.__items_dict["all_sp"])
            special_flg = np.asarray(special_flg).astype(np.int8)
            if self.env._args["recsim_if_special_items_flg_one_hot"]:
                special_flg = np.eye(2)[special_flg].astype(np.float32)
            else:
                special_flg = special_flg[:, None].astype(np.float32)
            _embedding = item_embedding.get_all(if_np=True)
            _embedding = np.hstack([_embedding, special_flg])
            logging(f"NewItemWrapper>> New Item Embedding Shape: {_embedding.shape}")
            item_embedding.load(embedding=_embedding)
        return item_embedding


class MetricsWrapper(gym.Wrapper):
    """ To integrate the metrics class """

    def __init__(self, env, metric_names: list = METRIC_NAMES, args: dict = {}):
        super(MetricsWrapper, self).__init__(env=env)
        self._env = env
        self._args = args
        self._if_recsim = args.get("env_name", "recsim") == "recsim" or args.get('env_name').startswith('RecSim')
        self._num_users = self._args.get("batch_step_size", 100)
        self._ep_reward = np.zeros(self._num_users)
        self._pairing_bonus = np.zeros(self._num_users)
        self._metrics = Metrics(item_embedding=env.item_embedding, metric_names=metric_names, args=args)
        self._category_mat = np.eye(self._args["recsim_num_categories"])

        _type_reward = self._args.get("recsim_type_reward", "click")
        if _type_reward in ["click"] + METRIC_NAMES:
            self._type_reward = _type_reward
        else:
            self._type_reward = None

        # prep
        self._prep()

    def _prep(self):
        # Set the metric function
        self._set_metric_fn()
        self._info = dict()

    def step(self, action):
        next_obs, reward, done, self._info = self._env.step(action)

        # Collect the reward by computing the step-wise metrics o/w use the original reward
        if self._type_reward != "click":
            _metrics = self._metrics.get_metrics_as_reward()
            if self._args["recsim_if_add_metric_to_click"]:
                if self._args["env_name"] == "recsim":
                    # Combine the click reward and the metric reward!
                    reward += _metrics[self._type_reward][self._info["prev_activeIds"]]
                else:
                    # Combine the click reward and the metric reward!
                    reward = reward.astype(np.float) + _metrics[self._type_reward]
            else:
                reward = _metrics[self._type_reward][self._info["prev_activeIds"]]

        if self._args["recsim_if_special_items"]:
            bonus = self._compute_special_item_bonus(action=action, if_pairing_cnt=True)
            # reward += (bonus * self._args["recsim_special_bonus_coeff"])[:, None]  # Reward is the col vector
            self._info["pairing_bonus"] = bonus

        return next_obs, reward, done, self._info

    def _compute_special_item_bonus(self, action, if_pairing_cnt: bool = False):
        if self._if_recsim:
            if self.env.environment.if_eval:
                key = self.env.environment.if_eval_train_or_test
            else:
                key = "train"
        else:
            if self.env.if_eval:
                key = self.env.if_eval_train_or_test
            else:
                key = "train"

        if len(action.shape) == 1:
            action = np.asarray([action])

        bonus, pairing_cnt = list(), list()
        for i in range(action.shape[0]):
            # This is inefficient search algo... O(N^2)
            _bonus = list()
            _pairing_cnt = 0
            for itemId in action[i]:
                # Find the category of an item; (key: categoryId, value: list of itemIds)
                _category = self.env.get_mainCategory_of_items(arr_items=[itemId])[0]

                # Binary flag indicating if it's a correct pairing of normal and special items
                flg = 0

                # if it's special then look for the corresponding normal item
                if itemId in self.env.items_dict["all_sp"]:
                    for _itemId in action[i]:
                        if _itemId in self.env.main_category_master_dict[key][_category]:
                            flg = 1
                            _pairing_cnt += 1
                            break
                else:  # if it's normal then look for the corresponding special item
                    for _itemId in action[i]:
                        if _itemId in self.env.main_category_master_dict[f"{key}_sp"][_category]:
                            flg = 1
                _bonus.append(flg)
            pairing_cnt.append(_pairing_cnt)
            bonus.append(_bonus)

        if if_pairing_cnt:
            return np.asarray(pairing_cnt)
        else:
            return np.asarray(bonus[0])

    @property
    def metrics(self):
        return self._metrics

    def update_metrics(self, reward, gt_items, pred_slates):
        """ Update the metrics in Metrics class; Supposed to be Used after env.step()

        Args:
            reward (np.ndarray): (batch_step_size)-size array
            gt_items (np.ndarray): (batch_step_size)-size array
            pred_slates (np.ndarray): batch_step_size x slate_size
        """
        # Info
        info = dict()

        # Get the Ids of active user
        if self._args.get("env_name", "recsim") == "recsim" or self._args.get('env_name').startswith('RecSim'):
            # RecSim has this attribute to main the active users
            info["active_userIds"] = self._info["prev_activeIds"]
        else:
            # Other envs don't maintain this information so that we just update all the rewards
            info["active_userIds"] = list(range(self._num_users))

        # Get the user state
        info["user_state"] = self.env.get_state()

        # Get the categories of items in slate
        info["category_mat"] = np.asarray([self._env.get_slate_category(slate=_slate) for _slate in pred_slates])

        # Get the corresponding item-embedding of items in a slate
        info["embeddings"] = self._env.item_embedding.get(index=pred_slates, if_np=True)
        info["visualise_metric"] = True  # for visualisation purpose!
        info["recsim_if_vector"] = False  # for visualisation purpose!

        self._ep_reward[info["active_userIds"]] += reward
        if self._args["recsim_if_special_items"]:
            self._pairing_bonus[info["active_userIds"]] += self._info["pairing_bonus"]
        self._metrics.update_metrics(gt_items=gt_items, pred_slates=pred_slates, info=info)

    def get_metrics(self):
        """ Returns the collected metrics including the episode return

        Returns:
            result (dict): dictionary of results; (key: metric-name, value: metric-value)
        """
        result = self._metrics.get_metrics()
        result["ep_reward"] = np.mean(self._ep_reward)
        result["pairing_bonus"] = np.mean(self._pairing_bonus)
        # if self._type_reward != "click":
        #     result["ep_reward"] = result[self._type_reward]  # Fill the metric of an episode
        # else:
        #     result["ep_reward"] = np.mean(self._ep_reward)
        return result

    def reset(self, **kwargs):
        self._metrics.reset()
        self._ep_reward = np.zeros(self._num_users)
        self._pairing_bonus = np.zeros(self._num_users)
        return self._env.reset()

    def set_itemEmbedding(self, item_embedding: BaseEmbedding):
        """ Set the item embedding class for Metrics class

        Args:
            item_embedding (BaseEmbedding):
        """
        self.metrics.set_item_embedding(item_embedding=item_embedding)

    def _set_metric_fn(self):
        # Set the metric that is used in User Choice Model
        _fn = self.metrics.metric_fn_factory(
            metric_name=self._args.get("recsim_reward_characteristic", "intrasession_diversity")  # TODO: isn't too deep
        )

        # Set metric computing method
        if self._args.get("env_name", "recsim") == "recsim" or self._args.get('env_name').startswith('RecSim'):
            self.env.environment.set_metric_fn(_fn=_fn)
        else:
            self.env.set_metric_fn(_fn=_fn)

        # Set CPR computing method;
        """ cpr-metric is decided here!
            - So we don't need to set anything for reward_characteristic!!
            - Make sure to set metric-alpha=0 when we wanna use just cpr-score!!
        """
        if self._args.get("recsim_if_use_subcategory", False):
            if self._args.get("env_name", "recsim") == "recsim" or self._args.get('env_name').startswith('RecSim'):
                self.env.environment.set_cpr_metric_fn(_fn=self.metrics.metric_fn_factory(metric_name="cpr_score"))
            else:
                self.env.set_cpr_metric_fn(_fn=self.metrics.metric_fn_factory(metric_name="cpr_score"))

        # Set Special Item bonus computation logic
        if self._args["recsim_if_special_items"]:
            if self._args.get("env_name", "recsim") == "recsim" or self._args.get('env_name').startswith('RecSim'):
                self.env.environment.set_pairing_bonus_fn(_fn=self._compute_special_item_bonus)
            else:
                self.env.set_pairing_bonus_fn(_fn=self._compute_special_item_bonus)


# TODO: Do we really need this...?
class VisualisationWrapper(gym.Wrapper):
    """ To visualise what's going on in RecSim """

    def __init__(self, env, args: dict):
        super(VisualisationWrapper, self).__init__(env=env)
        self._args = args
        self._rng = np.random.RandomState(ENV_RANDOM_SEED)

        # Create a dir if it doesn't exist
        if not os.path.exists(self._args["save_dir"]):
            os.makedirs(self._args["save_dir"])

    def plot(self):
        """ Plots all the visualisations """
        return self._plot()

    def _plot(self):
        """ Internal method of doing all the visualisations """
        self._plot_distance_slate_user()
        self._plot_item_category()
        self._plot_train_vs_test_items()
        self._plot_user_item()

    def _plot_train_vs_test_items(self):
        # Get the labels
        label_dict = dict()
        label_dict["labels"] = np.ones(len(self.env.items_dict["all"])).astype(np.int)
        label_dict["labels"][self.env.items_dict["train"]] = 1
        label_dict["labels"][self.env.items_dict["test"]] = 0
        label_dict["desc"] = {"train": 1, "test": 0}

        plot_embedding(embedding=self.env.item_embedding.get_all(),
                       label_dict=label_dict,
                       save_dir=os.path.join(self._args["save_dir"], "images/"),
                       file_name="{}_{}_train_vs_test".format(
                           self._args["recsim_itemFeat_samplingMethod"], self._args["recsim_itemFeat_samplingDist"]),
                       num_neighbours=self._args["recsim_num_categories"])

    def _plot_item_category(self):
        categories = self.env.get_allItemCategory()
        categories = list(categories.values())

        # Get the labels
        label_dict = dict()
        label_dict["labels"] = categories
        label_dict["desc"] = {k: v for k, v in enumerate(np.unique(categories))}

        plot_embedding(embedding=self.env.item_embedding.get_all(),
                       label_dict=label_dict,
                       save_dir=os.path.join(self._args["save_dir"], "images/"),
                       file_name="{}_{}_item_category".format(
                           self._args["recsim_itemFeat_samplingMethod"], self._args["recsim_itemFeat_samplingDist"]),
                       num_neighbours=self._args["recsim_num_categories"])

    def _plot_user_item(self):
        for _name in [
            "user_interests",
            "user_satisfaction"
        ]:
            user_states = self.env.get_state()
            embedding = np.vstack([user_states[_name], self.env.item_embedding.get_all()])

            # Get the labels
            label_dict = dict()
            label_dict["labels"] = np.zeros(embedding.shape[0]).astype(np.int)
            label_dict["labels"][:user_states[_name].shape[0]] = 0
            label_dict["labels"][user_states[_name].shape[0]:] = 1
            label_dict["desc"] = {"user": 0, "item": 1}

            plot_embedding(embedding=embedding,
                           label_dict=label_dict,
                           save_dir=os.path.join(self._args["save_dir"], "images/"),
                           file_name="{}_{}_user_item_{}".format(
                               self._args["recsim_itemFeat_samplingMethod"],
                               self._args["recsim_itemFeat_samplingDist"],
                               _name),
                           num_neighbours=self._args["recsim_num_categories"])

    def _plot_user_history(self):
        last_obs = self.env.obs  # From HistorySequenceWrapper
        _user_id = self._rng.choice(a=last_obs.data["history_seq"].shape[0], size=1)

        embedding = self.env.item_embedding.get(index=last_obs.data["history_seq"], if_np=True)
        embedding = embedding[_user_id, :]  # get a user's history_seq

        categories = self.env.get_allItemCategory()
        categories = [categories[f"{int(i)}"] for i in last_obs.data["history_seq"][_user_id]]

        label_dict = dict()
        label_dict["labels"] = categories
        label_dict["desc"] = {k: v for k, v in enumerate(np.unique(categories))}

        plot_embedding(embedding=embedding,
                       label_dict=label_dict,
                       save_dir=os.path.join(self._args["save_dir"], "images/"),
                       file_name="{}_{}_user_history".format(
                           self._args["recsim_itemFeat_samplingMethod"], self._args["recsim_itemFeat_samplingDist"]),
                       num_neighbours=self._args["recsim_num_categories"])


class ConsoleVisualisationWrapper(gym.Wrapper):
    """ To visualise what's going on in RecSim on Console """

    def __init__(self, env, args: dict):
        super(ConsoleVisualisationWrapper, self).__init__(env=env)
        self._args = args
        self._rng = np.random.RandomState(ENV_RANDOM_SEED)
        self._userId = 2
        self._action = None
        self._clicked_item = None
        self._if_resampling_at_ts = args.get("if_resampling_at_ts", False)
        self._if_visualise_console = False
        self._if_recsim = args.get("env_name", "recsim") == "recsim" or args.get('env_name').startswith('RecSim')

    def set_if_visualise_console(self, flg: bool = False):
        if (self._args.get("env_name", "recsim") == "recsim" or self._args.get('env_name').startswith(
                'RecSim')) and self._args.get("recsim_visualise_console", False):
            self._if_visualise_console = flg

    def get_test_aval(self):
        self.set_if_visualise_console(flg='True')
        return self.env.get_test_aval()

    def step(self, action):
        next_obs, reward, done, _info = self.env.step(action=action)

        # Only when we need, we visualise a trajectory on the console
        if self._if_visualise_console:
            if self._if_recsim:
                if self._userId in _info["prev_activeIds"] and self._userId in _info["cur_activeIds"]:
                    if self._args.get('env_name').startswith('RecSim'):
                        self._action = action
                    else:
                        self._action = action[_info["prev_activeIds"].index(self._userId)]
                    self._clicked_item = _info["gt_items"][_info["cur_activeIds"].index(self._userId)]
                    if self._if_resampling_at_ts:
                        self._logging_candidate_set()
                    self._logging_one_step()
            else:
                self._action = action[self._userId]
                self._clicked_item = _info["gt_items"][self._userId]
                if self._if_resampling_at_ts:
                    self._logging_candidate_set()
                self._logging_one_step()
        return next_obs, reward, done, _info

    def reset(self, **kwargs):
        res = self.env.reset()
        if self._if_visualise_console: self._logging_candidate_set()
        return res

    def _logging_candidate_set(self):
        if self._if_recsim:
            if self.env.environment.if_eval:
                key = self.env.environment.if_eval_train_or_test
            else:
                key = "train"
        else:
            if self.env.if_eval:
                key = self.env.if_eval_train_or_test
            else:
                key = "train"
        # logging histogram of candidate set in terms of category
        assert self._args["num_candidates"] == len(self.env.items_dict[key])
        candidate_category = self.env.get_mainCategory_of_items(arr_items=self.env.items_dict[key])
        cnt = Counter(candidate_category)
        cnt = {k: v / self._args["num_candidates"] for k, v in cnt.items()}
        cnt = dict(sorted(cnt.items(), key=lambda item: item[1]))
        logging(f"ConsoleVisualisation>> Hist of Candidate Set: {cnt}")

    def _logging_one_step(self):
        # logging slate
        action_category = self.env.get_mainCategory_of_items(arr_items=self._action)

        # if we have special items, then we append the additional postfix
        if self._args["recsim_if_special_items"]:
            _action_category = list()
            for index, itemId in enumerate(self._action):
                post_fix = "s" if itemId in self.env.items_dict["all_sp"] else "n"
                _action_category.append(f"{action_category[index]}_{post_fix}")
            action_category = _action_category

        # logging clicked item
        if self._clicked_item != SKIP_TOKEN:
            clicked_category = self.env.get_mainCategory_of_items(arr_items=[self._clicked_item])[0]
            if self._args["recsim_if_special_items"]:
                post_fix = "s" if self._clicked_item in self.env.items_dict["all_sp"] else "n"
                clicked_category = f"{clicked_category}_{post_fix}"
        else:
            clicked_category = -1

        logging(f"ConsoleVisualisation>> Slate: {action_category}, ClickedItem: {clicked_category}")


class RewardReshapeWrapper(gym.Wrapper):
    """ To hold the item embedding in the env """

    def __init__(self, env, args: dict):
        super(RewardReshapeWrapper, self).__init__(env=env)
        self._env = env
        self._args = args

    def step(self, action):
        next_obs, reward, done, _info = self.env.step(action=action)

        # === Reshape the reward
        if self._args["recsim_slate_reward_type"] == "all":
            _slate_reward = np.zeros((reward.shape[0], self._args["slate_size"]))
            _slate_reward[_info["gt_items"] != -1] = 1.0
        elif self._args["recsim_slate_reward_type"] == "last":
            _slate_reward = np.zeros((reward.shape[0], self._args["slate_size"] - 1))
            _mask = _info["gt_items"] != -1
            _slate_reward = np.hstack([_slate_reward, _mask[:, None].astype(np.float)])
        else:
            raise ValueError

        _info["slate_reward"] = _slate_reward

        # === Reshape the done
        _slate_done = np.zeros((reward.shape[0], self._args["slate_size"] - 1))
        _slate_done = np.hstack([_slate_done, done[:, None].astype(np.float)])
        _info["slate_done"] = _slate_done

        return next_obs, reward, done, _info


def wrap_env(env, args: dict):
    """ Launch all the wrappers for an env through this method! """
    # Update some end dependent params
    if_recsim = args.get("env_name", "recsim") == "recsim" or args.get("env_name").startswith("RecSim")
    args["if_recsim"] = if_recsim
    args["recsim_num_categories"] = args["recsim_num_categories"] if if_recsim else env.num_categories

    if_ppo_recsim = args.get("env_name").startswith('RecSim')

    # TODO: Order dependent implementation of wrappers... Make it order-independent clean implementation in the future!
    if if_recsim:
        env = ObsInfoWrapper(env=env)
        env = ItemEmbeddingWrapper(env=env)  # Get the ground truth item-embedding from RecSim env
    env = TimeLimit(env=env, max_episode_steps=args.get("max_episode_steps", 100))
    if not if_ppo_recsim:
        env = DoneWrapper(env=env, args=args)
    if if_recsim:
        if args.get("recsim_if_mdp", False):
            env = RecSimMDPWrapper(env=env, batch_size=args.get("batch_step_size", 100), if_ppo_recsim=if_ppo_recsim)
        else:
            env = HistorySequenceWrapper(env=env,
                                         batch_size=args.get("batch_step_size", 100),
                                         if_ppo_recsim=if_ppo_recsim)
    env = CompatibilityWrapper(env=env, args=args)
    env = NewItemWrapper(env=env,
                         num_categories=args["recsim_num_categories"],
                         num_candidates=args.get("num_candidates", 100),
                         num_candidates_sp=args.get("num_candidates_sp", 3),
                         num_candidates_correct_normal=args.get("num_candidates_correct_normal", 3),
                         resampling_method=args.get("recsim_resampling_method", "category_based"),
                         resampling_method_sp=args.get("recsim_special_item_sampling_method", "random"),
                         if_resampling_at_ts=args.get("if_resampling_at_ts", False),
                         if_special_items=args.get("recsim_if_special_items", False),
                         if_special_items_flg=args.get("recsim_if_special_items_flg", False),
                         if_recsim=if_recsim)
    env = VisualisationWrapper(env=env, args=args)
    env = ConsoleVisualisationWrapper(env=env, args=args)
    env = MetricsWrapper(env=env, args=args)
    if args["recsim_slate_reward"]:
        env = RewardReshapeWrapper(env=env, args=args)
    env.reset()  # To activate some stuff prior to the training
    return env
