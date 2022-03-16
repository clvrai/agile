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
"""Abstract classes that encode a user's state and dynamics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import numpy as np
import six

from value_based.commons.args import CPR_BONUS, SKIP_TOKEN, ENV_RANDOM_SEED, CPR_BONUS2
from value_based.envs.recsim.user import AbstractUserState
from value_based.commons.utils import softmax
from value_based.commons.test import TestCase as test


def adaptive_no_click_mass(scores, no_click_mass, alpha, _method):
    """ Compute the adaptive no-click-mass """
    if _method == "min_max":
        no_click_mass = min(scores) + alpha * (max(scores) - min(scores))
    elif _method == "slate_size":
        no_click_mass /= scores.shape[0]
    elif _method == "None":
        pass
    else:
        raise ValueError
    return no_click_mass


@six.add_metaclass(abc.ABCMeta)
class AbstractChoiceModel(object):
    """Abstract class to represent the user choice model. Each user has a choice model.
    """
    _scores = None
    _score_no_click = None
    _item_label = None

    @abc.abstractmethod
    def score_items(self, user_state: AbstractUserState = None, doc_obs: list = None, **kwargs):
        """Computes unnormalised scores of documents in the slate given user state.

        Args:
            user_state (AbstractUserState): An instance of AbstractUserState.
            doc_obs (list): (slate_size)-sized list of dim_item
        Attributes:
            scores: A numpy array that stores the scores of all documents.
            score_no_click: A float that represents the score for the action of picking no document.
        """

    @property
    def all_scores(self):
        return np.append(self._scores, self._score_no_click)

    @property
    def scores(self):
        return self._scores

    @property
    def score_no_click(self):
        return self._score_no_click

    @abc.abstractmethod
    def choose_item(self):
        """Returns selected index of document in the slate.

        Returns:
            selected_index: a integer indicating which item was chosen, or None if none were selected.
        """

    def reset_sampler(self):
        self._rng = np.random.RandomState(ENV_RANDOM_SEED)


class NormalizableChoiceModel(AbstractChoiceModel):
    """A normalizable choice model.

    Attributes:
        min_normalizer: A float (<= 0) used to offset the scores to be positive.
            Specifically, if the scores have negative elements, then they do not
            form a valid probability distribution for sampling. Subtracting the
            least expected element is one heuristic for normalization.
        no_click_mass: An optional float indicating the mass given to a no click option
    """

    def __init__(self, choice_features):
        self._rng = np.random.RandomState(ENV_RANDOM_SEED)
        self._min_normalizer = choice_features.get('min_normalizer', 0.0)
        self._no_click_mass = choice_features.get('no_click_mass', 0.0)
        self._no_click_mass_alpha = choice_features.get('no_click_mass_alpha', 0.3)
        self._no_click_mass_method = choice_features.get('no_click_mass_method', "None")
        self._if_noClickMass_include = choice_features.get("if_noClickMass_include", False)

    def set_no_click_mass(self, no_click_mass: float = 0.0):
        self._no_click_mass = no_click_mass

    @staticmethod
    def _score_items_helper(user_state, doc_obs, **kwargs):
        """ Dot product based scoring method

        Args:
            user_state (AbstractUserState): An instance of AbstractUserState.
            doc_obs (np.ndarray): (slate_size)-sized list of dim_item

        Returns:
            scores (np.ndarray): (slate_size)-sized array
        """
        scores = np.array([])
        for doc in doc_obs:
            scores = np.append(scores, user_state.score_document(doc, cosine=False))
        return scores

    def choose_item(self):
        """ Picks either an item or skip given the computed scores of items in a slate

        Returns:
            selected_index (int or None): itemId or None representing skip
        """
        # update no-click-mass
        self._score_no_click = adaptive_no_click_mass(scores=self._scores,
                                                      no_click_mass=self._score_no_click,
                                                      alpha=self._no_click_mass_alpha,
                                                      _method=self._no_click_mass_method)

        if self._if_noClickMass_include:
            # Normalise the scores including no_click_mass altogether; click probabilities are less controllable
            all_scores = np.append(self._scores, self._score_no_click)
        else:
            # Normalise the item scores and append no_click_mass; click probabilities are more controllable
            all_scores = np.append(self._scores, self._no_click_mass)

        if self._item_label is None:
            self._all_probs = softmax(all_scores)
        else:
            # likelihood of items with item-label=False will be 0
            _all_probs = np.zeros_like(all_scores)
            self._item_label = np.append(arr=self._item_label, values=True)  # for no-click-mass
            _all_probs[self._item_label] = softmax(all_scores[self._item_label])
            self._all_probs = _all_probs
        selected_index = self._rng.choice(a=np.arange(len(self._all_probs)), p=self._all_probs)  # np.arange stars at 0!
        if selected_index == len(self._all_probs) - 1:
            selected_index = SKIP_TOKEN
        return selected_index

    @property
    def all_probs(self):
        return self._all_probs


class MultinomialLogitChoiceModel(NormalizableChoiceModel):
    """ A multinomial logit choice model.
        Samples item x in scores according to p(x) = exp(x) / Sum_{y in scores} exp(y)
    """

    def score_items(self, user_state: AbstractUserState = None, doc_obs: list = None, **kwargs):
        """ Scores the items as well as the skip option
        Args:
            user_state (AbstractUserState): An instance of AbstractUserState.
            doc_obs (list): (slate_size)-sized list of dim_item
        """
        logits = self._score_items_helper(user_state, np.asarray(doc_obs))
        if self._if_noClickMass_include:
            logits = np.append(logits, self._no_click_mass)
        # Use softmax scores instead of exponential scores to avoid overflow.
        if self._if_noClickMass_include:
            self._scores = logits[:-1]
            self._score_no_click = logits[-1]
        else:
            scores = softmax(logits)
            self._scores = scores
            self._score_no_click = self._no_click_mass


class MultinomialProportionalChoiceModel(NormalizableChoiceModel):
    """ A multinomial proportional choice function.
        Samples item x in scores according to p(x) = x - min_normalizer / sum(x - min_normalizer)
    """

    def score_items(self, user_state: AbstractUserState = None, doc_obs: list = None, **kwargs):
        """ Scores the items as well as the skip option
        Args:
            user_state (AbstractUserState): An instance of AbstractUserState.
            doc_obs (list): (slate_size)-sized list of dim_item
        """
        scores = self._score_items_helper(user_state, np.asarray(doc_obs))
        if self._if_noClickMass_include:
            scores = np.append(scores, self._no_click_mass)
        scores = scores - self._min_normalizer
        scores /= np.sum(scores)
        if self._if_noClickMass_include:
            self._scores = scores[:-1]
            self._score_no_click = scores[-1]
        else:
            self._scores = scores
            self._score_no_click = self._no_click_mass


class CascadeChoiceModel(NormalizableChoiceModel):
    """The base class for cascade choice models.

    Attributes:
        attention_prob: The probability of examining a document i given document i - 1 not clicked.
        score_scaling: A multiplicative factor to convert score of doc i to the click probability of examined doc i.

    Raises:
        ValueError: if either attention_prob or base_attention_prob is invalid.
    """

    def __init__(self, choice_features):
        super(CascadeChoiceModel, self).__init__(choice_features=choice_features)
        self._min_normalizer = choice_features.get('min_normalizer', 0.0)
        self._attention_prob = choice_features.get('attention_prob', 0.9)
        self._score_scaling = choice_features.get('score_scaling')
        if self._attention_prob < 0.0 or self._attention_prob > 1.0:
            raise ValueError('attention_prob must be in [0,1].')
        if self._score_scaling < 0.0:
            raise ValueError('score_scaling must be positive.')

    def _positional_normalization(self, scores):
        """Computes the click probability of each document in _scores.

        The probability to click item i conditioned on unclicked item i - 1 is:
            attention_prob * score_scaling * score(i)
        We also compute the probability of not clicking any items in _score_no_click
        Because they are already probabilities, the normlaization in choose_item
        is no-op but we utilize random choice there.

        Args:
            scores: normalizable scores.
        """
        self._score_no_click = self._no_click_mass

        if self._if_noClickMass_include:
            # Original implementation!!
            for i in range(len(scores)):
                s = self._score_scaling * scores[i]
                # assert s <= 1.0, 'score_scaling cannot convert score %f into a probability' % scores[i]
                scores[i] = self._score_no_click * self._attention_prob * s
                self._score_no_click *= (1.0 - self._attention_prob * s)
        else:
            for i in range(len(scores)):
                s = self._score_scaling * scores[i]
                # assert s <= 1.0, 'score_scaling cannot convert score %f into a probability' % scores[i]
                scores[i] = (self._attention_prob ** i) * s
        self._scores = scores


class ExponentialCascadeChoiceModel(CascadeChoiceModel):
    """An exponential cascade choice model.

    Clicks the item at position i according to p(i) = attention_prob * score_scaling * exp(score(i))
    by going through the slate in order, and stopping once an item has been clicked.
    """

    def score_items(self, user_state: AbstractUserState = None, doc_obs: list = None, **kwargs):
        """ Scores the items as well as the skip option
        Args:
            user_state (AbstractUserState): An instance of AbstractUserState.
            doc_obs (list): (slate_size)-sized list of dim_item
        """
        scores = self._score_items_helper(user_state, np.asarray(doc_obs))
        scores = np.exp(scores)
        self._positional_normalization(scores)


class ProportionalCascadeChoiceModel(CascadeChoiceModel):
    """A proportional cascade choice model.

    Clicks the item at position i according to attention_prob * score_scaling * (score(i) - min_normalizer)
    by going through the slate in order, and stopping once an item has been clicked.
    """

    def __init__(self, choice_features):
        super(ProportionalCascadeChoiceModel, self).__init__(choice_features)

    def score_items(self, user_state: AbstractUserState = None, doc_obs: list = None, **kwargs):
        """ Scores the items as well as the skip option
        Args:
            user_state (AbstractUserState): An instance of AbstractUserState.
            doc_obs (list): (slate_size)-sized list of dim_item
        """
        scores = self._score_items_helper(user_state, np.asarray(doc_obs))
        scores = scores - self._min_normalizer
        # assert not scores[scores < 0.0], 'Normalized scores have non-positive elements.'
        self._positional_normalization(scores)


class SlateDependentChoiceModel(CascadeChoiceModel):
    """The class of user choice model which is dependent on the presented slate."""

    def __init__(self, choice_features: dict):
        super(SlateDependentChoiceModel, self).__init__(choice_features=choice_features)
        self._base_choiceModel = choice_features.get("base_SlateChoiceModel", "ProportionalCascadeChoiceModel")
        # Control the impact of metric over the slate
        self._metric_alpha = choice_features.get("choice_model_metric_alpha", 2.0)
        self._user_alpha = choice_features.get("choice_model_user_alpha", 1.0)
        self._cpr_alpha = choice_features.get("choice_model_cpr_alpha", 0.1)
        self._how_to_apply_metric = choice_features.get("choice_model_how_to_apply_metric", "add")
        self._slate_alpha = choice_features.get("choice_model_slate_alpha", 0.5)
        self._type_slateAggFn = choice_features.get("type_slateAggFn", "mean")
        if choice_features.get("choice_model_type_constraint", "None") != "None":
            _name = choice_features.get("choice_model_type_constraint", "soft-mask").split("-")
            self._type_constraint, self._mask_prob = _name
            self._mask_prob = self._mask_prob == "mask"
        else:
            self._type_constraint = self._mask_prob = "None", False

    def score_items(self, user_state: AbstractUserState = None, doc_obs: list = None, **kwargs):
        """ Scores the items as well as the skip option
        Args:
            user_state (AbstractUserState): An instance of AbstractUserState
            doc_obs (list): (slate_size)-sized list of dim_item
        """
        if user_state is not None:
            # Get the score for each item in a slate
            scores = self._score_items_helper(user_state, np.asarray(doc_obs))  # array of slate_size
            scores *= self._user_alpha
        else:
            # This will be used from DatasetEnv
            assert "base_scores" in kwargs, "You need to provide either user_state or base_scores"
            scores = kwargs["base_scores"]

        if ("metric" in kwargs) and (kwargs["metric"] is not None):
            """ Note
            Diversity will push the scores of all items up equally so that, overall, the click probs of items
            against the no_click_mass will increase accordingly.
            """
            if self._how_to_apply_metric == "add":
                scores += kwargs["metric"] * self._metric_alpha
            elif self._how_to_apply_metric == "multiply":
                scores *= (1 + (kwargs["metric"] * self._metric_alpha))
            else:
                raise ValueError

        if ("cpr_metric" in kwargs) and (kwargs["cpr_metric"] is not None):
            # print(kwargs["cpr_metric"] * self._cpr_alpha)
            scores += kwargs["cpr_metric"] * self._cpr_alpha

        if ("item_label" in kwargs) and (kwargs["item_label"] is not None):
            # from pudb import set_trace; set_trace()
            bonus_vec = np.zeros(kwargs["item_label"].shape[0])

            if type(kwargs["item_label"][0]) != bool:
                # Transform the binary label into the booleans
                kwargs["item_label"] = kwargs["item_label"] == 1

            if self._mask_prob:
                self._item_label = kwargs["item_label"]

            if self._type_constraint.lower() == "soft":
                # Discount the scores of bad items and raise the scores of good items

                """
                item_label, score > 0.0, CPR_BONUS
                T, T, T
                T, F. F
                F, T, F
                F, F, T
                """
                _mask = ~(kwargs["item_label"]) ^ (scores > 0.0)  # XNOR Logic
                bonus_vec[_mask] = CPR_BONUS
                bonus_vec[~_mask] = 1 / CPR_BONUS
                scores *= bonus_vec
            elif self._type_constraint.lower() == "soft2":
                # Raise the scores of good items
                _mask = ~(kwargs["item_label"]) ^ (scores > 0.0)  # XNOR Logic
                bonus_vec[_mask] = CPR_BONUS
                bonus_vec[~_mask] = 1 / CPR_BONUS
                bonus_vec[~kwargs["item_label"]] = 1.0  # Replace the discounting factor of bad items with 1!!
                scores *= bonus_vec
            elif self._type_constraint.lower() == "add":
                # Add the bonus constant to the correct pairings in the slate
                bonus_vec[kwargs["item_label"]] = CPR_BONUS2
                scores += bonus_vec

        if self._base_choiceModel == "ProportionalCascadeChoiceModel":
            scores -= self._min_normalizer
            self._positional_normalization(scores)
        elif self._base_choiceModel == "ExponentialCascadeChoiceModel":
            scores = np.exp(scores)  # array of slate_size
            self._positional_normalization(scores)  # array of slate_size
        elif self._base_choiceModel == "MultinomialProportionalChoiceModel":
            if self._if_noClickMass_include:
                scores = np.append(scores, self._no_click_mass)
            scores -= self._min_normalizer
            if self._if_noClickMass_include:
                self._scores = scores[:-1]
                self._score_no_click = scores[-1]
            else:
                self._scores = scores
                self._score_no_click = self._no_click_mass

        elif self._base_choiceModel == "MultinomialLogitChoiceModel":
            if self._if_noClickMass_include:
                scores = np.append(scores, self._no_click_mass)
            # Use softmax scores instead of exponential scores to avoid overflow.
            if self._if_noClickMass_include:
                self._scores = scores[:-1]
                self._score_no_click = scores[-1]
            else:
                scores = softmax(scores)
                self._scores = scores
                self._score_no_click = self._no_click_mass
        elif self._base_choiceModel == "None":
            self._scores = scores
            self._score_no_click = self._no_click_mass
        else:
            raise ValueError


def choose_UserChoiceModel(_type="MultinomialLogitChoiceModel"):
    if _type == "MultinomialLogitChoiceModel":
        return MultinomialLogitChoiceModel
    elif _type == "MultinomialProportionalChoiceModel":
        return MultinomialProportionalChoiceModel
    elif _type == "ExponentialCascadeChoiceModel":
        return ExponentialCascadeChoiceModel
    elif _type == "ProportionalCascadeChoiceModel":
        return ProportionalCascadeChoiceModel
    elif _type == "SlateDependentChoiceModel":
        return SlateDependentChoiceModel
    else:
        raise ValueError


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        self._prep()

    def test(self):
        from environments.interest_evolution import UtilityModelUserSampler

        print("=== Test ===")

        # === Hyper-params
        choice_features = {
            'score_scaling': 0.05,
            'attention_prob': 0.9,
            'no_click_mass': 1.0,
            'min_normalizer': 0.0,
            "choice_model_user_alpha": 1.0,
            "choice_model_slate_alpha": 0.5,
            "choice_model_metric_alpha": 0.5,
            # "choice_model_metric_alpha": 1.0,
            # "choice_model_how_to_apply_metric": "add",
            "choice_model_how_to_apply_metric": "multiply",
            "type_slateAggFn": "mean",
            "if_noClickMass_include": True,
            "base_SlateChoiceModel": "MultinomialProportionalChoiceModel"
        }

        for type_itemFeatures in [
            # "discrete",
            "continuous"
        ]:
            for choice_model_ctor in [
                # MultinomialLogitChoiceModel,
                # MultinomialProportionalChoiceModel,
                # ExponentialCascadeChoiceModel,
                # ProportionalCascadeChoiceModel,
                SlateDependentChoiceModel
            ]:
                print(f"\n=== choice_model: {choice_model_ctor}, type_itemFeatures: {type_itemFeatures}")

                # User-interest vector over 3 categories
                feat_sample_fn = lambda: np.asarray([0.1, 0.5, 0.9])
                user_sampler = UtilityModelUserSampler(feat_sample_fn=feat_sample_fn, seed=self.args.random_seed)
                choice_model = choice_model_ctor(choice_features=choice_features)

                # Get a user
                user_state = user_sampler.sample_user(flg_trainTest="train")

                # Get slates
                bad_slate = [[0.9, 0.5, 0.1], [0.9, 0.5, 0.1]]
                good_slate = [[0.2, 0.4, 0.8], [0.2, 0.4, 0.8]]
                perfect_slate = [[0.1, 0.5, 0.9], [0.1, 0.5, 0.9]]

                choice_model.score_items(user_state=user_state, doc_obs=bad_slate, metric=0.1)
                print(f"[bad]     scores: {choice_model.all_scores} selected item: {choice_model.choose_item()}")

                choice_model.score_items(user_state=user_state, doc_obs=good_slate, metric=0.1)
                print(f"[good]    scores: {choice_model.all_scores} selected item: {choice_model.choose_item()}")

                choice_model.score_items(user_state=user_state, doc_obs=perfect_slate, metric=0.1)
                print(f"[perfect] scores: {choice_model.all_scores} selected item: {choice_model.choose_item()}")


if __name__ == '__main__':
    Test().test()
