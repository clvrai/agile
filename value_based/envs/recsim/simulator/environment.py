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
"""Class to represent the environment in the recommender system setting.

   Thus, it models things such as (1) the user's state, for example his/her
   interests and circumstances, (2) the documents available to suggest from and
   their properties, (3) simulates the selection of an item in the slate (or a
   no-op/quit), and (4) models the change in a user's state based on the slate
   presented and the document selected.

   The agent interacting with the environment is the recommender system.  The
   agent receives the state, which is an observation of the user's state and
   observations of the candidate documents. The agent then provides an action,
   which is a slate (an array of indices into the candidate set).

   The goal of the agent is to learn a recommendation policy: a policy that
   serves the user a slate (action) based on user and document features (state)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import six
import collections
import itertools
import numpy as np
from copy import deepcopy
from sklearn.model_selection import train_test_split

from value_based.envs.recsim import document
from value_based.envs.recsim.document import AbstractDocumentSampler
from value_based.commons.args import ENV_RANDOM_SEED


@six.add_metaclass(abc.ABCMeta)
class AbstractEnvironment(object):
    """Abstract class representing the recommender system environment.

    Attributes:
        user_model: An list or single instantiation of AbstractUserModel representing the user/users.
        document_sampler: An instantiation of AbstractDocumentSampler.
        num_allItems: An integer representing the size of the candidate_set.
        slate_size: An integer representing the slate size.
        candidate_set: An instantiation of CandidateSet.
        num_categories: An integer representing the number of document clusters.
    """

    def __init__(self,
                 user_model,
                 document_sampler,
                 num_items_dict,
                 slate_size,
                 resample_documents=True,
                 if_new_action_env=False,
                 if_debug=False):
        """Initializes a new simulation environment.

        Args:
            user_model: An instantiation of AbstractUserModel or list of such instantiations
            document_sampler: An instantiation of AbstractDocumentSampler
            num_allItems: An integer representing the size of the candidate_set
            slate_size: An integer representing the slate size
            resample_documents: A boolean indicating whether to resample the candidate set every step
            if_new_action_env: A boolean indicating whether separate the itemIds into train/test
        """

        # === Original Attributes ===
        self._user_model = user_model
        self._document_sampler = document_sampler
        self._args = self._document_sampler.feature_dist._args
        self._slate_size = slate_size
        self._resample_documents = resample_documents

        # === Additional Attributes ===
        self._if_debug = if_debug
        self._num_items_dict = num_items_dict
        self._if_new_action_env = if_new_action_env
        self._candidate_items_dict = dict()  # (key: train/test, value: list of itemIds)
        self._hard_constraint_rule_dict = dict()  # (key: train/test, value: list of categoryIds satisfying constraints)

        # Create a candidate set
        self._split_train_test()
        self._do_resample_documents()

    def _do_resample_documents(self):
        """Resample the doc features."""
        # Init the bucket of the item instances
        self._candidate_set = document.CandidateSet(args=self._args)
        self._document_sampler.feature_dist.generate_item_representations(
            list_trainItemIds=self._items_dict["train"], list_testItemIds=self._items_dict["test"], if_special=False
        )
        if self._args["recsim_if_special_items"]:
            self._document_sampler.feature_dist.generate_item_representations(
                list_trainItemIds=self._items_dict["train_sp"], list_testItemIds=self._items_dict["test_sp"],
                if_special=True
            )

        # Instantiate the underlying item features
        keys = ["all"]
        if self._args["recsim_if_special_items"]:
            keys += ["all_sp"]
        for key in keys:
            for itemId in self._items_dict[key]:
                """ Change the behaviour to instantiate the item features
                    ** If itemId is in the intersection of train/test sets then it's initialised as train!
                """
                if_special = key == "all_sp"
                _key = "train_sp" if if_special else "train"
                flg_trainTest = "train" if itemId in self._items_dict[_key] else "test"
                _doc = self._document_sampler.sample_document(doc_id=itemId,
                                                              flg_trainTest=flg_trainTest,
                                                              if_special=if_special)
                self._candidate_set.add_document(document=_doc)

    @property
    def items_dict(self):
        return self._items_dict

    @property
    def candidate_items_dict(self):
        """ This will be update from NewItemWrapper in wrapper.py """
        return self._candidate_items_dict

    def set_candidate_items_dict(self, candidate_items_dict: dict):
        """ Create the link reference from Wrapper's candidate set to this base Env class's candidate set """
        self._candidate_items_dict = candidate_items_dict

    def set_hard_constraint_rule_dict(self, flg: str):
        """ Reassign the categories of items in the candidate set
            Args:
                flg: str of either train or test
        """
        _category_mat = self._candidate_set.get_slate_category(slate=self.candidate_items_dict[flg])
        hard_constraint_rule = np.argwhere(np.min(_category_mat, axis=-1) > 0.0).ravel()
        self._hard_constraint_rule_dict[flg] = hard_constraint_rule

    def _split_train_test(self):
        """ Init the dict which contain the itemIds of training/test
            ** This _items_dict is used in NewItemWrapper to sample the candidate_list
        """
        _all = np.arange(self._num_items_dict["all"])
        if self._if_new_action_env:
            # Split the normal items into train and test
            _train, _test = train_test_split(_all,
                                             random_state=ENV_RANDOM_SEED,
                                             train_size=self._num_items_dict["train"],
                                             test_size=self._num_items_dict["test"])
            self._items_dict = {"all": sorted(_all), "train": sorted(_train), "test": sorted(_test)}

            if self._args["recsim_if_special_items"]:
                # Split the special items into train and test
                # Note: ItemIds for special items start after the normal items in both train and test sets
                _all_sp = np.arange(self._num_items_dict["all_sp"] * 2) + self._num_items_dict["all"]
                _train_sp, _test_sp = train_test_split(_all_sp,
                                                       random_state=ENV_RANDOM_SEED,
                                                       train_size=self._num_items_dict["train_sp"],
                                                       test_size=self._num_items_dict["test_sp"])
                self._items_dict.update(
                    {"all_sp": sorted(_all_sp), "train_sp": sorted(_train_sp), "test_sp": sorted(_test_sp)}
                )

        else:
            # Split the normal items into train and test
            self._items_dict = {"all": sorted(_all), "train": sorted(_all), "test": sorted(_all)}

            if self._args["recsim_if_special_items"]:
                # Split the special items into train and test
                # Note: ItemIds for special items start after the normal items
                _all_sp = np.arange(self._num_items_dict["all_sp"]) + self._num_items_dict["all"]
                self._items_dict.update(
                    {"all_sp": sorted(_all_sp), "train_sp": sorted(_all_sp), "test_sp": sorted(_all_sp)}
                )

        if self._if_debug:
            print("AbstractEnvironment>> items_dict: {}".format({k: len(v) for k, v in self._items_dict.items()}))

    @abc.abstractmethod
    def reset(self):
        """Resets the environment and return the first observation.

        Returns:
          user_obs: An array of floats representing observations of the user's
            current state
          doc_obs: An OrderedDict of document observations keyed by document ids
        """

    @abc.abstractmethod
    def reset_sampler(self):
        """Resets the relevant samplers of documents and user/users."""

    @property
    def num_allItems(self):
        return self._num_items_dict["all"]

    @property
    def slate_size(self):
        return self._slate_size

    @property
    def candidate_set(self):
        return self._candidate_set

    @property
    def user_model(self):
        return self._user_model

    @abc.abstractmethod
    def step(self, slate):
        """Executes the action, returns next state observation and reward.

        Args:
          slate: An integer array of size slate_size (or list of such arrays), where
          each element is an index into the set of current_documents presented.

        Returns:
          user_obs: A gym observation representing the user's next state
          doc_obs: A list of observations of the documents
          responses: A list of AbstractResponse objects for each item in the slate
          done: A boolean indicating whether the episode has terminated
        """


class SingleUserEnvironment(AbstractEnvironment):
    """Class to represent the environment with one user.

    Attributes:
      user_model: An instantiation of AbstractUserModel that represents a user.
      document_sampler: An instantiation of AbstractDocumentSampler.
      num_allItems: An integer representing the size of the candidate_set.
      slate_size: An integer representing the slate size.
      candidate_set: An instantiation of CandidateSet.
      num_categories: An integer representing the number of document clusters.
    """

    def reset(self):
        """Resets the environment and return the first observation.

        Returns:
          user_obs: An array of floats representing observations of the user's
            current state
          doc_obs: An OrderedDict of document observations keyed by document ids
        """
        self._user_model.reset()
        user_obs = self._user_model.create_observation()
        if self._resample_documents:
            self._do_resample_documents()
        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())
        return (user_obs, self._current_documents)

    def reset_sampler(self):
        """Resets the relevant samplers of documents and user/users."""
        self._document_sampler.reset_sampler()
        self._user_model.reset_sampler()

    def step(self, slate):
        """Executes the action, returns next state observation and reward.

        Args:
          slate: An integer array of size slate_size, where each element is an index
            into the set of current_documents presented

        Returns:
          user_obs: A gym observation representing the user's next state
          doc_obs: A list of observations of the documents
          responses: A list of AbstractResponse objects for each item in the slate
          done: A boolean indicating whether the episode has terminated
        """

        assert (len(slate) <= self._slate_size
                ), 'Received unexpectedly large slate size: expecting %s, got %s' % (
            self._slate_size, len(slate))

        # Get the documents associated with the slate
        doc_ids = list(self._current_documents)  # pytype: disable=attribute-error
        mapped_slate = [doc_ids[x] for x in slate]
        documents = self._candidate_set.get_documents(mapped_slate)
        # Simulate the user's response
        responses = self._user_model.simulate_response(documents)

        # Update the user's state.
        self._user_model.update_state(documents, responses)

        # Update the documents' state.
        self._document_sampler.update_state(documents, responses)

        # Obtain next user state observation.
        user_obs = self._user_model.create_observation()

        # Check if reaches a terminal state and return.
        done = self._user_model.is_terminal()

        # Optionally, recreate the candidate set to simulate candidate
        # generators for the next query.
        if self._resample_documents:
            self._do_resample_documents()

        # Create observation of candidate set.
        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())

        return (user_obs, self._current_documents, responses, done)


Environment = SingleUserEnvironment  # for backwards compatibility


class MultiUserEnvironment(AbstractEnvironment):
    """Class to represent environment with multiple users.

    Attributes:
        user_model: A list of AbstractUserModel instances that represent users.
        num_users: An integer representing the number of users.
        document_sampler: An instantiation of AbstractDocumentSampler.
        num_allItems: An integer representing the size of the candidate_set.
        slate_size: An integer representing the slate size.
        candidate_set: An instantiation of CandidateSet.
        num_categories: An integer representing the number of document clusters.
    """

    def __init__(self,
                 user_model: list,
                 document_sampler: AbstractDocumentSampler,
                 num_items_dict: dict,
                 slate_size: int,
                 resample_documents: bool = False,
                 if_new_action_env: bool = True):
        super(MultiUserEnvironment, self).__init__(user_model=user_model,
                                                   document_sampler=document_sampler,
                                                   num_items_dict=num_items_dict,
                                                   slate_size=slate_size,
                                                   resample_documents=resample_documents,
                                                   if_new_action_env=if_new_action_env)
        self._if_eval = False
        self._if_eval_train_or_test = "test"
        self._activeUsers = self.user_model
        self._activeIds = list(range(len(self._activeUsers)))
        self._active_user_mask = np.asarray([True] * len(self._activeUsers))
        self._category_info = self._document_sampler.feature_dist.category_info
        self._if_use_subcategory = self._document_sampler.feature_dist.if_use_subcategory
        self._itemCategories_dict = self._candidate_set.get_categories()
        self._category_mat = np.eye(self._category_info["main"])

    @property
    def if_eval(self):
        return self._if_eval

    @property
    def if_eval_train_or_test(self):
        return self._if_eval_train_or_test

    def set_if_eval(self, flg: bool):
        """ Used to define the sampling dist of user-interest """
        self._if_eval = flg

    def set_if_eval_train_or_test(self, train_or_test: str):
        """ Used to define the sampling dist of user-interest """
        self._if_eval_train_or_test = train_or_test

    @property
    def active_users(self):
        """ Used to aggregate the responses of active users in recsim_gym.py """
        return self._activeUsers

    @property
    def active_userIds(self):
        """ Used to collect the logged data in rollout.py """
        return self._activeIds

    @property
    def active_user_mask(self):
        return self._active_user_mask

    @property
    def categories(self):
        return self._itemCategories_dict

    @property
    def category_master_dict(self):
        return self._document_sampler.category_master_dict

    @property
    def main_category_master_dict(self):
        return self._document_sampler.main_category_master_dict

    @property
    def id2category_dict(self):
        return self._document_sampler.id2category_dict

    def set_metric_fn(self, _fn):
        """ not a good design pattern but temp way to bypass a metric computing method from Wrapper class to this
            lower level env implementation.
        """
        [user.set_metric_fn(_fn=_fn) for user in self.user_model]

    def set_cpr_metric_fn(self, _fn):
        """ not a good design pattern but temp way to bypass a metric computing method from Wrapper class to this
            lower level env implementation.
        """
        [user.set_cpr_metric_fn(_fn=_fn) for user in self.user_model]

    def set_pairing_bonus_fn(self, _fn):
        """ not a good design pattern but temp way to bypass a metric computing method from Wrapper class to this
            lower level env implementation.
        """
        [user.set_pairing_bonus_fn(_fn=_fn) for user in self.user_model]

    def reset(self):
        """Resets the environment and return the first observation.

        Returns:
          user_obs: An array of floats representing observations of the user's
            current state
          doc_obs: An OrderedDict of document observations keyed by document ids
        """
        # Set all the users active
        self._activeUsers = self.user_model
        self._activeIds = list(range(len(self._activeUsers)))
        self._active_user_mask = np.asarray([True] * len(self._activeUsers))

        for user_model in self._activeUsers:
            # flg_trainTest is used to make sure that we sample users from the same distribution as items(train/test)
            flg_trainTest = "test" if self._if_eval else "train"
            user_model.reset(flg_trainTest=flg_trainTest)
        user_obs = [user_model.create_observation() for user_model in self._activeUsers]

        # this is unnecessary since it's dealt in Wrapper class for New-action Env
        # if self._resample_documents:
        #     self._do_resample_documents()

        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())
        return (user_obs, self._current_documents)

    def reset_sampler(self):
        self._document_sampler.reset_sampler()
        for user_model in self.user_model:
            user_model.reset_sampler()

    def step(self, slates):
        """Executes the action, returns next state observation and reward.

        Args:
            slates: A list of slates, where each slate is an integer array of size
                slate_size, where each element is an index into the set of
                current_documents presented

        Returns:
            user_obs: A list of gym observation representing active users' next state
            doc_obs: A list of observations of the documents
            responses: A list of AbstractResponse objects for each item in the slate
            done: A boolean indicating whether the episode has terminated
        """
        if_ppo_recsim = self._document_sampler.feature_dist._args.get('env_name').startswith('RecSim')
        active_user_obs = []  # Accumulate each user's obses to served documents.
        active_user_done = []  # Accumulate each user's dones to served documents.
        active_user_responses = []  # Accumulate each user's responses to served documents.
        all_documents = []  # Accumulate documents served to each user.
        recommendable_category = self._hard_constraint_rule_dict["test" if self._if_eval else "train"]
        if if_ppo_recsim:
            assert len(self._activeIds) == 1
            slates = [slates]
        for user_id, user_model, slate in zip(self._activeIds, self._activeUsers, slates):
            # Get the documents associated with the slate
            documents = self._candidate_set.get_documents(slate)
            if not if_ppo_recsim:
                done = user_model.is_terminal()  # check if a user hasn't run out the time-budget at the previous time-step
            else:
                done = False

            # If a user is active
            if not done:
                # Get the categories of items in slate; num_categories x num_subcategories or num_categories x 1
                category_mat = self._candidate_set.get_slate_category(slate=slate)

                # Array of category of items in slate: (slate_size)-size array or slate_size x 2(main/sub categories)
                category_vec = np.asarray([self._candidate_set.get_fullCategory_vec(itemId=i) for i in slate])

                # Simulate the user's response using the pre-defined UserChoiceModel
                # list of IEvResponse objects, one for each item
                responses = user_model.simulate_response(documents=documents,
                                                         pred_slate=slate,
                                                         user_id=user_id,
                                                         category_mat=category_mat,
                                                         category_vec=category_vec,
                                                         recommendable_category=recommendable_category)

                # Update the user's state(ie., user-interests vector and scalar of time-budget) based on a clicked item
                user_model.update_state(documents, responses)

                # Obtain next user state observation.
                active_user_obs.append(user_model.create_observation())

                # Record the responses of active users
                active_user_responses.append(responses)
            else:
                # Record the responses of users who just turned inactive
                active_user_responses.append(user_model.create_empty_response(slate_size=len(slate)))

            # Other things to log
            if if_ppo_recsim:
                done = user_model.is_terminal()  # check if a user hasn't run out the time-budget at the CURRENT time-step
            active_user_done.append(done)
            all_documents.append(documents)

        def flatten(list_):
            return list(itertools.chain(*list_))

        # Update the documents' state.; NOTE: In Interest Evolution env, nothing is implemented in update_state
        self._document_sampler.update_state(flatten(all_documents), flatten(active_user_responses))

        info = {
            # "prev_activeUsers": deepcopy(self._activeUsers),
            "prev_activeIds": deepcopy(self._activeIds),
            "prev_active_user_mask": deepcopy(self._active_user_mask)
        }
        mask = np.asarray(active_user_done, dtype=np.bool)
        # Check if users reach a terminal state.
        done = all(mask)

        # These info about Active users are used in the higher level implementation of Env
        self._active_user_mask = ~mask
        self._activeUsers = list(itertools.compress(data=self._activeUsers, selectors=~mask))
        self._activeIds = list(itertools.compress(data=self._activeIds, selectors=~mask))
        info.update({"cur_active_user_mask": self._active_user_mask, "cur_activeIds": deepcopy(self._activeIds)})

        # Optionally, recreate the candidate set to simulate candidate generators for the next query.
        if self._resample_documents:
            self._do_resample_documents()

        # Create observation of candidate set.
        self._current_documents = collections.OrderedDict(self._candidate_set.create_observation())

        return (active_user_obs, self._current_documents, active_user_responses, done, info)
