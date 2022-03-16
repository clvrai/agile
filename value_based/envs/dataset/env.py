"""
## Env
- Needs to provide (obs, reward, candidate_list)
    - obs: concat of [user_attr, user_history]
    - reward: reward model will inference
    - candidate_list: this changes over time so we need the master to maintain this info
        - get_candidateList: timestamp -> list of available items

## Data
- User attributes: user_attr.npy
- Item attributes: item_attr.npy
- Click log-data: train/val/test_log.csv
    - cols: click_in_slate,hist_seq,item_id,slate,timestamp,user_id,negative_items
"""
import os
import gym
import numpy as np
import pandas as pd

from typing import Dict
from sklearn.model_selection import train_test_split

from value_based.envs.wrapper import wrap_env
from value_based.commons.test import TestCase as test
from value_based.embedding.base import BaseEmbedding
from value_based.commons.observation import ObservationFactory
from value_based.envs.dataset.user_model.base import UserModel
from value_based.envs.dataset.content_manager import ContentManager
from value_based.commons.args import ENV_RANDOM_SEED, USER_HISTORY_COL_NAME


class DatasetEnv(gym.Env):
    def __init__(self, dict_embedding: Dict[str, BaseEmbedding], dict_dfLog: dict = None, args: dict = None):
        """ OpenAI Gym based Env for Dataset in general

        Args:
            dict_embedding (dict):
            dict_dfLog (dict): dict of log data in DataFrame; num_sessions x dim
            args (dict): args
        """
        self._args = args
        self._rng = np.random.RandomState(ENV_RANDOM_SEED)
        self._if_eval = False
        self._if_eval_train_or_test = "test"
        self._dim_item = self._args.get("dim_item", 32)
        self._dim_user = self._args.get("dim_user", 32)
        self._user_feat = self._args.get("user_feat", None)
        self._item_feat = self._args.get("item_feat", None)
        self._batch_step_size = self._args.get("batch_step_size", 100)
        self._slate_size = self._args.get("slate_size", 5)
        self._history_size = self._args.get("history_size", 5)
        self._if_use_log = self._args.get("data_if_use_log", False)

        # used to sample from the log data
        self._reset_cursors()

        # Preprocess data
        self.dict_dfLog = dict_dfLog

        # Set the components
        self.dict_embedding = dict_embedding

        # Prepare the list of user_ids
        self.user_ids = list(range(self.dict_embedding["user"].shape[0]))

        # Prepare the user model
        self.dict_UMs = dict()
        for name in ["offline", "online"]:
            self._args["rm_weight_path"] = os.path.join(self._args["rm_weight_save_dir"], f"{name}.pkl")
            self.dict_UMs[name] = UserModel(item_embedding=self.dict_embedding["item"], args=self._args)

        # get the items for training/test stages
        self._items_dict = dict()
        self._split_train_test()
        self.cc = ContentManager(items_dict=self._items_dict, df_master=dict_dfLog["df_master"], args=args)
        if self._args["recsim_if_special_items"]:  # prepare the special items for each main category
            self._prep_special_items()
            self.cc.set_items_dict(items_dict=self._items_dict)
        self._metric_fn = self._cpr_metric_fn = None

    def _reset_cursors(self):
        self._prev, self._cur = {"offline": 0, "online": 0}, \
                                {"offline": self._batch_step_size, "online": self._batch_step_size}

    @property
    def num_categories(self):
        return self.cc.num_categories

    @property
    def items_dict(self):
        return self._items_dict

    @property
    def item_embedding(self):
        return self.dict_embedding["item"]

    def _split_train_test(self):
        if ("offline" in self.dict_dfLog) and ("online" in self.dict_dfLog):
            self._items_dict["train"] = self.dict_dfLog["offline"]["item_id"].unique().tolist()
            self._items_dict["test"] = self.dict_dfLog["online"]["item_id"].unique().tolist()
            self._items_dict["all"] = self._items_dict["train"] + self._items_dict["test"]

        else:
            if self._args.get("recsim_if_new_action_env", True):
                # Synthetically generate the train/test item sets
                print(self.dict_embedding["item"].shape[0], self._args.get("num_allItems"))
                assert self.dict_embedding["item"].shape[0] == self._args.get("num_allItems")
                list_item_ids = list(range(self.dict_embedding["item"].shape[0]))
                _train, _test = train_test_split(list_item_ids,
                                                 random_state=ENV_RANDOM_SEED,
                                                 train_size=self._args.get("num_trainItems"),
                                                 test_size=self._args.get("num_testItems"))
            else:
                list_item_ids = _train = _test = list(range(self.dict_embedding["item"].shape[0]))
            self._items_dict = {"all": sorted(list_item_ids), "train": sorted(_train), "test": sorted(_test)}

        print("Available Items: [train] {} [test] {}".format(len(self._items_dict["train"]),
                                                             len(self._items_dict["test"])))
        self._items_dict["new_items"] = list(set(self._items_dict["test"]) - set(self._items_dict["train"]))
        print("New items in test stage: {}".format(len(self._items_dict["new_items"])))

    def _prep_special_items(self):
        for key in ["train", "test"]:
            self._items_dict[f"{key}_sp"] = list()
            for category_id in self.cc.main_category_master_dict[key].keys():
                # Randomly select the special item for each main category
                itemId_sp = self._rng.choice(a=self.cc.main_category_master_dict[key][category_id], size=1)[0]

                # Replace one item with the special item
                # Note: train and test sets intersect so that we need to remove from both sets!!
                # self._items_dict["train"].remove(itemId_sp)
                # self._items_dict["train"].remove(itemId_sp)
                self._items_dict[key].remove(itemId_sp)
                self._items_dict[f"{key}_sp"].append(itemId_sp)
        self._items_dict["all_sp"] = self._items_dict["train_sp"] + self._items_dict["test_sp"]

    def step(self, action):
        """ OpenAI API based step method

        Args:
            action (np.ndarray): batch_step_size x slate_size

        Returns:
            next_obs (np.ndarray): batch_step_size x history_size x dim_state;
                updated history_seq with clicked item or sampled history_seq from the log data
            reward (np.ndarray): batch_step_size x 1; binary(click/non-click) col vector
            done (bool): bool value indicating whether an episode terminates or not
            info (dict): additional info
        """
        if self._if_use_log:
            # During evaluation, we only sample from the log data to evaluate the performance of agent
            flg = "online" if self._if_eval else "offline"

            # gt_items and slates belong to the obs at the last step in the log data
            gt_items = self.gt_items

            # sample from the log data
            self.obs, self.gt_items, done = self.sample_from_dataset(flg=flg)

            # Get the reward
            reward = np.asarray([float(gt_item in slate) for gt_item, slate in zip(gt_items, action)])
        else:
            _obs = self.obs.make_obs(dict_embedding=self.dict_embedding, if_separate=False, device=self._args["device"])

            if self._if_eval:
                if self._if_eval_train_or_test == "train":
                    key = "offline"
                else:
                    key = "online"
            else:
                key = "offline"
            reward, a_user = self.dict_UMs[key].predict(obs=_obs, slate=action)

            # Update the user history sequence
            self.obs.update_history_seq(a_user=a_user)
            done = False  # we only terminate an episode by the maximum timesteps(see TimeLimit wrapper)
            gt_items = a_user

        if len(reward.shape) == 1:
            reward = np.expand_dims(reward, 1)
        elif len(reward.shape) > 2:
            raise ValueError
        reward = reward.astype(np.float) * self._args.get("data_click_bonus", 1.0)
        return self.obs, reward, done, {"gt_items": gt_items}

    def reset(self):
        # used to sample from the log data
        self._reset_cursors()
        if ("offline" in self.dict_dfLog) and ("online" in self.dict_dfLog):
            flg = "online" if self._if_eval else "offline"
            sample = self._sample_from_dataset(flg=flg)
            user_id = sample["user_id"].values
        else:
            # In training without log data, we start with the zero-filled state representing a new user
            # We randomly sample the user attributes from the log data
            user_id = self._rng.choice(a=self.user_ids, size=self._batch_step_size)
        self.obs = ObservationFactory(batch_size=self._batch_step_size, user_id=user_id)
        return self.obs

    def _sample_from_dataset(self, flg: str = "offline"):
        """ Sample entries from the log data by batch_step_size """
        # Sample the entries from the log data using double cursor technique
        if self._cur[flg] >= self.dict_dfLog[flg].shape[0]:
            sample = self.dict_dfLog[flg].sample(min(self.dict_dfLog[flg].shape[0], self._batch_step_size))
        else:
            # when it reaches the bottom of the log data.
            sample = self.dict_dfLog[flg].iloc[self._prev[flg]: min(self.dict_dfLog[flg].shape[0], self._cur[flg])]
        return sample

    def sample_from_dataset(self, flg: str = "offline"):
        """ Sample entries from the log data by batch_step_size
            This is supposed to be used when evaluation or specifically we'd like to train on the log dataset
            We basically go through all the samples in the log data from top to bottom and when it reaches the bottom,
            an episode terminates.

        Returns:
            _obs_dict: a dict of user_feat(batch_step_size x dim_user_feat) and
                history_seq(batch_step_size x history_size x dim_item)
            gt_items: a 2D matrix; batch_step_size x 1
            slates: a 2D matrix; batch_step_size x slate_size
            done: a bool representing if an episode terminates
        """
        # Sample the entries from the log data using double cursor technique
        sample = self._sample_from_dataset(flg=flg)

        obs = ObservationFactory(batch_size=sample.shape[0], user_id=sample["user_id"].values)
        obs.load_history_seq(history_seq=sample[USER_HISTORY_COL_NAME].values)

        # get other stuffs
        gt_items = sample["item_id"].values  # batch_step_size x 1
        # when it reaches the bottom of the log data, an episode terminates
        done = self.dict_dfLog[flg].shape[0] <= self._cur[flg]

        # Update the cursors
        self._prev[flg], self._cur[flg] = self._cur[flg], self._cur[flg] + self._batch_step_size
        return obs, gt_items, done

    @property
    def if_eval(self):
        return self._if_eval

    @property
    def if_eval_train_or_test(self):
        return self._if_eval_train_or_test

    def set_if_use_log(self, flg: bool = True):
        self._if_use_log = flg

    def set_if_eval(self, flg: bool = True):
        self._if_eval = flg

    def set_if_eval_train_or_test(self, train_or_test: str):
        self._if_eval_train_or_test = train_or_test

    def set_metric_fn(self, _fn):
        self._metric_fn = _fn

    def set_cpr_metric_fn(self, _fn):
        self._cpr_metric_fn = _fn

    # def get_log_data(self):
    #     """ Sample the logged data and supposed to be stored in Experience Replay
    #
    #     References:
    #         - Pseudo Dyna-Q(L.Zou WSDM20); Algo 1 L.26
    #         - FeedRec(L.Zou KDD19); Algo 1 L.12
    #
    #     Returns:
    #         obs: batch_step_size x history_size x dim_obs
    #         action: batch_step_size x slate_size
    #         reward: batch_step_size x 1
    #         next_obs: batch_step_size x history_size x dim_obs
    #         done: bool
    #     """
    #     # Sample from log data and get the next obs
    #     obs_dict, gt_items, _ = self._sample_from_dataset(type="train")
    #     next_obs_dict = self._update_state(obs_dict=obs_dict, a_user=gt_items)
    #
    #     # Process the log
    #     obs, next_obs = _dict_to_numpy(_dict=obs_dict), _dict_to_numpy(_dict=next_obs_dict)
    #     action = slates[:, :self._slate_size]
    #     reward = np.asarray([float(gt_item in slate) for gt_item, slate in zip(gt_items, action)])
    #     reward = np.expand_dims(reward, 1)
    #     return obs, action, reward, next_obs, False


def launch_env(dict_embedding: dict, args: dict):
    """ Launch an env based on args """
    # Load the log data
    df_master = pd.read_csv(os.path.join(args["data_dir"], "item_category.csv"))
    dict_dfLog = {"df_master": df_master}
    dict_dfLog["offline"] = pd.read_csv(os.path.join(args["data_dir"], "offline_log_data.csv"))
    dict_dfLog["online"] = pd.read_csv(os.path.join(args["data_dir"], "online_log_data.csv"))
    print("Log data; offline: {} online: {}".format(dict_dfLog["offline"].shape, dict_dfLog["online"].shape))

    # Instantiate the components
    env = DatasetEnv(dict_embedding=dict_embedding, dict_dfLog=dict_dfLog, args=args)

    env = wrap_env(env, args=args)
    return env


class Test(test):
    def __init__(self):
        from value_based.commons.args import NUM_FAKE_ITEMS
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        self.args.env_name = "sample"
        self.args.agent_type = "random"
        self.args.num_allItems = NUM_FAKE_ITEMS
        self.args.num_candidates = 20
        self.args.no_click_mass = 3.0
        self.args.recsim_reward_characteristic = "intrasession_diversity"
        self.args.recsim_type_specificity = "cosine"
        self.args.item_embedding_type = "pretrained"
        self.args.user_embedding_type = "pretrained"
        self.args.item_embedding_path = "../../../data/sample/item_attr.npy"
        self.args.user_embedding_path = "../../../data/sample/user_attr.npy"
        self.args.data_category_cols = "genre/size"
        self.args.data_dir = "../../../data/sample/"
        self.args.rm_weight_save_dir = "../../../trained_weight/reward_model/sample/"
        self._prep()

    def test(self):
        from value_based.policy.agent import RandomAgent
        from value_based.envs.recsim.env_test_random import test_short

        print("=== test: DatasetEnv ===")
        agent = RandomAgent(obs_encoder=None, slate_encoder=None, args=vars(self.args))
        env = launch_env(dict_embedding=self.dict_embedding, args=vars(self.args))

        for epoch in range(5):
            ep_metrics, ts = test_short(env=env, agent=agent, if_eval=True)
            print("reward: {}, ep_metrics: {}, ts: {}".format(ep_metrics["ep_reward"], ep_metrics, ts))


if __name__ == '__main__':
    Test().test()
