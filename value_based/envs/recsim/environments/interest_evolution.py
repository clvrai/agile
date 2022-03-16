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
"""Classes to represent the interest evolution documents and users."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from gym import spaces

from value_based.envs.recsim import choice_model, document, user, utils, distribution
from value_based.envs.recsim.simulator import environment, recsim_gym
from value_based.envs.wrapper import wrap_env
from value_based.commons.args import (
    MIN_QUALITY_SCORE, MAX_QUALITY_SCORE, MAX_VIDEO_LENGTH, MEAN_VIDEO_LENGTH, MIN_VIDEO_LENGTH, SIGMA_VIDEO_LENGTH,
    SIGMA_QUALITY_SCORE, ENV_RANDOM_SEED, SKIP_TOKEN
)
from value_based.commons.utils import scipy_truncated_normal


class IEvResponse(user.AbstractResponse):
    """Class to represent a user's response to a video.

    Attributes:
        clicked: A boolean indicating whether the video was clicked.
        watch_time: A float for fraction of the video watched.
        liked: A boolean indicating whether the video was liked.
        quality: A float indicating the quality of the video.
        category_id: A integer representing the cluster ID of the video.
    """

    def __init__(self, clicked=False, watch_time=0.0, liked=False, quality=0.0, category_id=0.0):
        """Creates a new user response for a video.

        Args:
            clicked: A boolean indicating whether the video was clicked
            watch_time: A float for fraction of the video watched
            liked: A boolean indicating whether the video was liked
            quality: A float for document quality
            category_id: a integer for the cluster ID of the document.
        """
        self.clicked = clicked
        self.watch_time = watch_time
        self.liked = liked
        self.quality = quality
        self.category_id = category_id

    def create_observation(self):
        return {
            'click': int(self.clicked),
            'watch_time': np.array(self.watch_time),
            'liked': int(self.liked),
            'quality': np.array(self.quality),
            'category_id': int(self.category_id),
        }

    @classmethod
    def response_space(cls):
        """
        `clicked` feature range is [0, 1]
        `watch_time` feature range is [0, MAX_VIDEO_LENGTH]
        `liked` feature range is [0, 1]
        `quality`: the quality of the document and range is specified by [MIN_QUALITY_SCORE, MAX_QUALITY_SCORE].
        `category_id`: the cluster the document belongs to and its range is.  [0, IEvVideo.NUM_FEATURES].
        """
        return spaces.Dict({
            'click': spaces.Discrete(2),
            'watch_time': spaces.Box(low=0.0, high=MAX_VIDEO_LENGTH, shape=tuple(), dtype=np.float32),
            'liked': spaces.Discrete(2),
            'quality': spaces.Box(low=MIN_QUALITY_SCORE, high=MAX_QUALITY_SCORE, shape=tuple(), dtype=np.float32),
            'category_id': spaces.Discrete(1)
        })


class IEvUserState(user.AbstractUserState):
    """Class to represent interest evolution users."""

    def __init__(self,
                 user_interests,
                 user_satisfaction,
                 time_budget=None,
                 score_scaling=None,
                 attention_prob=None,
                 no_click_mass=None,
                 base_SlateChoiceModel=None,
                 keep_interact_prob=None,
                 min_doc_utility=None,
                 user_update_alpha=None,
                 watched_videos=None,
                 impressed_videos=None,
                 liked_videos=None,
                 skip_penalty=None,
                 min_normalizer=None,
                 choice_model_user_alpha=None,
                 choice_model_cpr_alpha=None,
                 choice_model_slate_alpha=None,
                 choice_model_metric_alpha=None,
                 no_click_mass_alpha=None,
                 no_click_mass_method=None,
                 choice_model_how_to_apply_metric=None,
                 if_noClickMass_include=None,
                 type_slateAggFn=None,
                 user_quality_factor=None,
                 document_quality_factor=None):
        """Initializes a new user."""

        # Only user_interests is required, since it is needed to create an
        # observation. It is the responsibility of the designer to make sure any
        # other variables that are needed in the user choice/transition model are
        # also provided.

        ## User features
        #######################

        # The user's interests (1 = very interested, -1 = disgust)
        # Another option could be to represent in [0,1] e.g. by dirichlet
        self.user_interests = user_interests
        self.user_satisfaction = user_satisfaction

        # Amount of time in minutes this user has left in session.
        self.time_budget = time_budget

        # Probability of interacting with another element on the same slate
        self.keep_interact_prob = keep_interact_prob

        # Min utility to interact with a document
        self.min_doc_utility = min_doc_utility

        # Convenience wrapper
        self.choice_features = {
            # === Original Features ===
            'score_scaling': score_scaling,
            # Factor of attention to give for subsequent items on slate
            # Item i on a slate will get attention (attention_prob)^i
            'attention_prob': attention_prob,
            # Mass that user does not click on any item in the slate
            'no_click_mass': no_click_mass,
            # If using the multinomial proportion model with negative scores, this
            # negative value will be subtracted from all scores to make a valid
            # distribution for sampling.
            'min_normalizer': min_normalizer,

            # === Additional Features ===
            # Config of Slate Dependent User Choice Model
            "base_SlateChoiceModel": base_SlateChoiceModel,
            "choice_model_user_alpha": choice_model_user_alpha,
            "choice_model_cpr_alpha": choice_model_cpr_alpha,
            "choice_model_slate_alpha": choice_model_slate_alpha,
            "choice_model_metric_alpha": choice_model_metric_alpha,
            "no_click_mass_method": no_click_mass_method,
            "no_click_mass_alpha": no_click_mass_alpha,
            "choice_model_how_to_apply_metric": choice_model_how_to_apply_metric,
            "if_noClickMass_include": if_noClickMass_include,
            "type_slateAggFn": type_slateAggFn,
        }

        ## Transition model parameters
        ##############################

        # Step size for updating user interests based on watched videos (small!)
        # We may want to have different values for different interests
        # to represent how malleable those interests are (e.g. strong dislikes may
        # be less malleable).
        self.user_update_alpha = user_update_alpha

        # A step penalty applied when no item is selected (e.g. the time wasted
        # looking through a slate but not clicking, and any loss of interest)
        self.skip_penalty = skip_penalty

        # How much to weigh the user quality when updating budget
        self.user_quality_factor = user_quality_factor
        # How much to weigh the document quality when updating budget
        self.document_quality_factor = document_quality_factor

        # Observable user features (these are just examples for now)
        ###########################

        # Video IDs of videos that have been watched
        self.watched_videos = watched_videos

        # Video IDs of videos that have been impressed
        self.impressed_videos = impressed_videos

        # Video IDs of liked videos
        self.liked_videos = liked_videos

    def score_document(self, doc_obs, cosine=False):
        if self.user_interests.shape != doc_obs.shape:
            raise ValueError('User and document feature dimension mismatch!')
        if cosine:
            cosine_similarity = np.dot(self.user_interests, doc_obs) /\
                (np.linalg.norm(self.user_interests) * np.linalg.norm(doc_obs))
            return cosine_similarity
        else:
            return np.dot(self.user_interests, doc_obs)

    def create_observation(self):
        """Return an observation of this user's observable state."""
        return self.user_interests

    # @classmethod
    # def observation_space(cls):
    #     # return spaces.Box(shape=(RECSIM_DIM_USER,), dtype=np.float32, low=-1.0, high=1.0)
    #     return spaces.Box(shape=(1,), dtype=np.float32, low=-1.0, high=1.0)

    def get_state(self):
        """Return this user's internal state."""
        return {
            "user_interests": self.user_interests,
            "user_satisfaction": self.user_satisfaction,
            "time_budget": self.time_budget,
            "keep_interact_prob": self.keep_interact_prob,
            "min_doc_utility": self.min_doc_utility,
            "choice_features": self.choice_features,
            "user_update_alpha": self.user_update_alpha,
            "skip_penalty": self.skip_penalty,
            "user_quality_factor": self.user_quality_factor,
            "document_quality_factor": self.document_quality_factor,
            "watched_videos": self.watched_videos,
            "impressed_videos": self.impressed_videos,
            "liked_videos": self.liked_videos,
        }


class UtilityModelUserSampler(user.AbstractUserSampler):
    """Class that samples users for utility model experiment."""

    def __init__(self,
                 feat_sample_fn,
                 user_state_ctor=IEvUserState,
                 document_quality_factor=1.0,
                 no_click_mass=1.0,
                 base_SlateChoiceModel="",
                 min_normalizer=0.0,
                 choice_model_user_alpha=1.0,
                 choice_model_cpr_alpha=0.1,
                 choice_model_slate_alpha=0.5,
                 choice_model_metric_alpha=2.0,
                 no_click_mass_alpha=0.2,
                 no_click_mass_method="None",
                 choice_model_how_to_apply_metric="add",
                 if_noClickMass_include=False,
                 type_slateAggFn="mean",
                 time_budget=200.0,
                 skip_penalty=0.5,
                 score_scaling=0.05,  # for CascadingUserChoiceModel
                 attention_prob=0.9,  # for CascadingUserChoiceModel
                 user_quality_factor=0.0,
                 num_categories=20,
                 **kwargs):
        """Creates a new user state sampler."""
        self._no_click_mass = no_click_mass
        self._min_normalizer = min_normalizer
        self._document_quality_factor = document_quality_factor
        self._time_budget = time_budget
        self._skip_penalty = skip_penalty
        self._score_scaling = score_scaling
        self._attention_prob = attention_prob
        self._user_quality_factor = user_quality_factor

        # === Additional Features of User State to define User Choice Model ===
        self._num_categories = num_categories
        self._base_SlateChoiceModel = base_SlateChoiceModel
        self._choice_model_user_alpha = choice_model_user_alpha
        self._choice_model_cpr_alpha = choice_model_cpr_alpha
        self._choice_model_slate_alpha = choice_model_slate_alpha
        self._choice_model_metric_alpha = choice_model_metric_alpha
        self._no_click_mass_method = no_click_mass_method
        self._no_click_mass_alpha = no_click_mass_alpha
        self._choice_model_how_to_apply_metric = choice_model_how_to_apply_metric
        self._if_noClickMass_include = if_noClickMass_include
        self._type_slateAggFn = type_slateAggFn
        self._feat_sample_fn = feat_sample_fn

        super(UtilityModelUserSampler, self).__init__(user_state_ctor=user_state_ctor, **kwargs)

    def sample_user(self, flg_trainTest: str):
        features = {}

        # Interests/Satisfaction are distributed uniformly randomly
        features['user_interests'] = self._feat_sample_fn()
        features['user_satisfaction'] = self._rng.uniform(low=-1.0, high=1.0, size=self._num_categories)
        features['time_budget'] = self._time_budget
        features['no_click_mass'] = self._no_click_mass
        features['skip_penalty'] = self._skip_penalty
        features['score_scaling'] = self._score_scaling
        features['attention_prob'] = self._attention_prob
        features['min_normalizer'] = self._min_normalizer
        features['user_quality_factor'] = self._user_quality_factor
        features['document_quality_factor'] = self._document_quality_factor
        features['base_SlateChoiceModel'] = self._base_SlateChoiceModel
        features['choice_model_user_alpha'] = self._choice_model_user_alpha
        features['choice_model_cpr_alpha'] = self._choice_model_cpr_alpha
        features['choice_model_slate_alpha'] = self._choice_model_slate_alpha
        features['choice_model_metric_alpha'] = self._choice_model_metric_alpha
        features['no_click_mass_alpha'] = self._no_click_mass_alpha
        features['no_click_mass_method'] = self._no_click_mass_method
        features['choice_model_how_to_apply_metric'] = self._choice_model_how_to_apply_metric
        features['if_noClickMass_include'] = self._if_noClickMass_include
        features['type_slateAggFn'] = self._type_slateAggFn

        # Fraction of video length we can extend (or cut) budget by
        # Maybe this should be a parameter that varies by user?
        alpha = 0.9
        # In our setup, utility is just doc_quality as user_quality_factor is 0.
        # doc_quality is distributed normally ~ N([-3,3], 0.1) for a 3 sigma range
        # of [-3.3,3.3]. Therefore, we normalize doc_quality by 3.4 (adding a little
        # extra in case) to get in [-1,1].
        utility_range = 1.0 / 3.4
        features['user_update_alpha'] = alpha * utility_range
        return self._user_state_ctor(**features)


class IEvUserModel(user.AbstractUserModel):
    """Class to model an interest evolution user.

    Assumes the user state contains:
        - user_interests
        - time_budget
        - no_click_mass
    """

    def __init__(self,
                 slate_size,
                 feat_sample_fn,
                 choice_model_ctor=None,
                 response_model_ctor=IEvResponse,
                 user_state_ctor=IEvUserState,
                 no_click_mass=1.0,
                 base_SlateChoiceModel="",
                 user_model_metric_if_vector=True,
                 user_model_type_constraint="None",
                 choice_model_type_constraint="soft",
                 choice_model_user_alpha=1.0,
                 choice_model_cpr_alpha=0.1,
                 choice_model_slate_alpha=0.5,
                 choice_model_metric_alpha=2.0,
                 no_click_mass_method="None",
                 no_click_mass_alpha=0.2,
                 choice_model_how_to_apply_metric="add",
                 if_noClickMass_include=False,
                 type_slateAggFn="mean",
                 seed=ENV_RANDOM_SEED,
                 alpha_x_intercept=1.0,
                 alpha_y_intercept=0.3,
                 time_budget=200.0,
                 skip_penalty=0.5):
        """Initializes a new user model.

        Args:
            slate_size: An integer representing the size of the slate
            choice_model_ctor: A constructor function to create user choice model.
            response_model_ctor: A constructor function to create response. The function should take
                a string of doc ID as input and returns a IEvResponse object.
            user_state_ctor: A constructor to create user state
            no_click_mass: A float that will be passed to compute probability of no click.
            seed: A integer used as the seed of the choice model.
            alpha_x_intercept: A float for the x intercept of the line used to compute interests update factor.
            alpha_y_intercept: A float for the y intercept of the line used to compute interests update factor.

        Raises:
            Exception: if choice_model_ctor is not specified.
        """
        super(IEvUserModel, self).__init__(
            response_model_ctor=response_model_ctor,
            user_sampler=UtilityModelUserSampler(user_state_ctor=user_state_ctor,
                                                 no_click_mass=no_click_mass,
                                                 base_SlateChoiceModel=base_SlateChoiceModel,
                                                 choice_model_user_alpha=choice_model_user_alpha,
                                                 choice_model_cpr_alpha=choice_model_cpr_alpha,
                                                 choice_model_slate_alpha=choice_model_slate_alpha,
                                                 choice_model_metric_alpha=choice_model_metric_alpha,
                                                 no_click_mass_alpha=no_click_mass_alpha,
                                                 no_click_mass_method=no_click_mass_method,
                                                 choice_model_how_to_apply_metric=choice_model_how_to_apply_metric,
                                                 if_noClickMass_include=if_noClickMass_include,
                                                 type_slateAggFn=type_slateAggFn,
                                                 feat_sample_fn=feat_sample_fn,
                                                 time_budget=time_budget,
                                                 skip_penalty=skip_penalty,
                                                 seed=seed),
            slate_size=slate_size
        )
        if choice_model_ctor is None:
            raise Exception('A choice model needs to be specified!')

        # TODO: move args for choice model that we added and not exist in the original RecSim here
        #  so that we can just update the choice_features directly!
        self._user_state.choice_features.update(
            {"choice_model_type_constraint": choice_model_type_constraint}
        )
        self.choice_model = choice_model_ctor(choice_features=self._user_state.choice_features)

        self._alpha_x_intercept = alpha_x_intercept
        self._alpha_y_intercept = alpha_y_intercept

        self._metric_if_vector = user_model_metric_if_vector
        self._type_constraint = user_model_type_constraint
        self._metric_fn = None  # this is supposed to be set from the higher level, e.g., Wrapper!
        self._cpr_metric_fn = None  # this is supposed to be set from the higher level, e.g., Wrapper!
        self._pairing_bonus_fn = None  # this is supposed to be set from the higher level, e.g., Wrapper!
        self._base_category_mat = np.eye(len(self._user_state.create_observation()))

    def is_terminal(self):
        """Returns a boolean indicating if the session is over."""
        return self._user_state.time_budget <= 0.0

    def _update_user_characteristic(self, doc_vec, user_vec):
        """ Updated to make the user's characteristic similar to the clicked item's one

        Args:
            doc_vec (np.ndarray):
            user_vec (np.ndarray):

        Returns:
            user_vec (np.ndarray):
        """
        target = doc_vec - user_vec
        mask = doc_vec
        # Step size should vary based on interest.
        alpha = (-self._alpha_y_intercept / self._alpha_x_intercept) * np.absolute(user_vec) + self._alpha_y_intercept

        update = alpha * mask * target
        positive_update_prob = np.dot((user_vec + 1.0) / 2, mask)
        flip = self._user_sampler._rng.rand(1)
        if flip < positive_update_prob:
            user_vec += update
        else:
            user_vec -= update
        user_vec = np.clip(user_vec, -1.0, 1.0)
        return user_vec

    def update_state(self, slate_documents, responses):
        """Updates the user state based on responses to the slate.

        This function assumes only 1 response per slate. If a video is watched, we
        update the user's interests some small step size alpha based on the
        user's interest in that topic. The update is either towards the
        video's features or away, and is determined stochastically by the user's
        interest in that document.

        Args:
            slate_documents: a list of IEvVideos representing the slate
            responses: a list of IEvResponses representing the user's response to each video in the slate.
        """

        user_state = self._user_state

        for doc, response in zip(slate_documents, responses):
            if response.clicked:
                # Update user interests
                user_state.user_interests = self._update_user_characteristic(doc_vec=doc.features,
                                                                             user_vec=user_state.user_interests)

                # Update user satisfaction
                # TODO: should this be fixed?
                user_state.user_satisfaction = self._update_user_characteristic(doc_vec=doc.quality,
                                                                                user_vec=user_state.user_satisfaction)

                # Update budget
                _satisfaction = np.dot(user_state.user_satisfaction, doc.quality)  # dot product based satisfaction
                user_state.time_budget -= response.watch_time  # Consuming an item reduces the remaining time-budget
                # good recommendation entertains a user so that the user gets to interact with an agent more
                user_state.time_budget += user_state.user_update_alpha * _satisfaction

                # TODO: Change here to accommodate multiple clicks in a slate!
                return

        # TODO: Change here to accommodate multiple clicks in a slate!
        # Step penalty if no selection
        user_state.time_budget -= user_state.skip_penalty

    def simulate_response(self, documents, **kwargs):
        """ Simulates the user's response to a slate of documents with choice model.
            Used in step API of Env classes in environment.py

        Args:
            documents: a list of IEvVideo objects
            kwargs: variable dict of other key arguments

        Returns:
            responses: a list of IEvResponse objects, one for each document
        """
        # List of empty responses
        responses = self.create_empty_response(slate_size=len(documents))

        # Update the attributes of UserResponse Class
        # TODO: we can extend here to add more attributes in the response!
        for i, response in enumerate(responses):
            response.quality = documents[i].quality
            response.category_id = documents[i].category_id

        # Get item embedding of items in a slate
        doc_obs = [doc.create_observation() for doc in documents]

        # Compute the metrics to be used in User Choice Model later
        assert ("pred_slate" in kwargs) and ("user_id" in kwargs)

        info = {
            "user_id": kwargs["user_id"],
            "category_mat": kwargs["category_mat"],  # num_categories x num_subcategories or num_categories x 1
            "category_vec": kwargs["category_vec"],  # (slate_size)-size array or slate_size x 2(main/sub category)
            "recsim_metric_if_vector": self._metric_if_vector,
        }

        # Compute the metric
        if self._metric_fn is not None:
            _metric = self._metric_fn(gt_item=None, pred_slate=kwargs["pred_slate"], info=info)
        else:
            _metric = None

        # Use either the hard constraint or CPR score
        """ Note
        - When user the hard constraint, we give bonus to only items that satisfy the constraint
        - When user the CPR score, we compute the CPR scores for all the items in slate
        """
        item_label = None  # used in the user choice model
        _cpr_metric = None  # used in the user choice model
        _bonus_pairing = None  # used in the user choice model
        if self._type_constraint.lower() != "none":
            recommendable_category = None
            item_label = list()
            if self._type_constraint.lower() == "user":
                recommendable_category = [np.argmax(self._user_state.create_observation())]
                # recommendable_category_vec = self._base_category_mat[np.argmax(self._user_state.create_observation())]
            elif self._type_constraint.lower() == "candidate":
                recommendable_category = kwargs["recommendable_category"]
                # recommendable_category_vec = self._base_category_mat[kwargs["recommendable_category"]]
            for item_category in kwargs["category_vec"]:
                # Get the main category of an item
                item_category = item_category if len(item_category) == 1 else item_category[0]

                # Evaluate if the item satisfies the constraint
                item_label.append(item_category in recommendable_category)
            item_label = np.asarray(item_label)

        # Compute CPR score
        if self._cpr_metric_fn is not None:
            _cpr_metric = self._cpr_metric_fn(gt_item=None, pred_slate=kwargs["pred_slate"], info=info)

        # Compute the pairing bonus
        if self._pairing_bonus_fn is not None:
            assert item_label is None
            _bonus_pairing = self._pairing_bonus_fn(action=kwargs["pred_slate"], if_pairing_cnt=False)
            item_label = _bonus_pairing  # we update the item-label here

        # Metric is used to calibrate the scores of items in a slate
        self.choice_model.score_items(user_state=self._user_state,
                                      doc_obs=doc_obs,
                                      metric=_metric,
                                      cpr_metric=_cpr_metric,
                                      item_label=item_label)

        # TODO: We can modify here to be able to accommodate multiple user feedbacks!
        # Pick an item or skip based on the computed scores
        selected_index = self.choice_model.choose_item()

        # If it's skip;
        if selected_index is SKIP_TOKEN:
            return responses

        # Generate the click based response
        self._generate_click_response(documents[selected_index], responses[selected_index])

        return responses

    def _generate_click_response(self, doc, response):
        """Generates a response to a clicked document.

        Right now we assume watch_time is a fixed value that is the minium value of
        time_budget and video_length. In the future, we may want to try more
        variations of watch_time definition.

        Args:
            doc: an IEvVideo object
            response: am IEvResponse for the document
        Updates:
            response, with whether the document was clicked, liked, and how much of it was watched
        """
        user_state = self._user_state
        response.clicked = True
        response.watch_time = min(user_state.time_budget, doc.video_length)

    def set_metric_fn(self, _fn):
        self._metric_fn = _fn

    def set_cpr_metric_fn(self, _fn):
        self._cpr_metric_fn = _fn

    def set_pairing_bonus_fn(self, _fn):
        self._pairing_bonus_fn = _fn

    def reset_sampler(self):
        self.choice_model.reset_sampler()
        self._user_sampler.reset_sampler()


class IEvVideo(document.AbstractDocument):
    """Class to represent a interest evolution Video.

    Attributes:
        features: A numpy array that stores video features.
        category_id: An integer that represents.
        video_length : A float for video length.
        quality: a float the represents document quality.
    """

    def __init__(self, doc_id, features, category_id, video_length=None, quality=None):
        """Generates a random set of features for this interest evolution Video."""
        # Document features (i.e. distribution over topics)
        self.features = features

        # Cluster ID
        self.category_id = category_id

        # Length of video
        self.video_length = video_length

        # Document quality (i.e. trashiness/nutritiousness)
        self.quality = quality

        # doc_id is an integer representing the unique ID of this document
        super(IEvVideo, self).__init__(doc_id)

    def create_observation(self):
        """Returns observable properties of this document as a float array."""
        return self.features

    def create_state(self):
        """Returns all the properties of this document as a float array."""
        return np.append(self.features, [self.video_length, self.quality])

    # @classmethod
    # def observation_space(cls):
    #     # return spaces.Box(shape=(RECSIM_DIM_ITEM,), dtype=np.float32, low=-1.0, high=1.0)
    #     return spaces.Box(shape=(1,), dtype=np.float32, low=-1.0, high=1.0)


class UtilityModelVideoSampler(document.AbstractDocumentSampler):
    """Class that samples videos for utility model experiment."""

    def __init__(self,
                 doc_ctor=IEvVideo,
                 feature_dist: distribution.Distribution = distribution.Distribution,  # TODO: dirty.... fix this.
                 num_categories: int = 20,
                 **kwargs):
        """Creates a new utility model video sampler.

        Args:
            doc_ctor: A class/constructor for the type of videos that will be sampled by this sampler.
            min_utility: A float for the min utility score.
            max_utility: A float for the max utility score.
            video_length: A float for the video_length in minutes.
            **kwargs: other keyword parameters for the video sampler.
        """
        super(UtilityModelVideoSampler, self).__init__(doc_ctor=doc_ctor, **kwargs)
        self._doc_count = 0
        self._feature_dist = feature_dist
        self._num_categories = num_categories

    @property
    def feature_dist(self):
        return self._feature_dist

    @property
    def category_master_dict(self):
        return self._feature_dist.category_master_dict

    @property
    def main_category_master_dict(self):
        return self._feature_dist.main_category_master_dict

    @property
    def id2category_dict(self):
        return self._feature_dist.id2category_dict

    def sample_document(self, doc_id: int, flg_trainTest: str = "train", if_special: bool = False):
        doc_features = {}
        doc_features['doc_id'] = doc_id

        # Get the doc features
        doc_features['features'], doc_features['category_id'] = self._feature_dist.sample(itemId=doc_id,
                                                                                          flg=flg_trainTest,
                                                                                          if_special=if_special)

        # Sample a video length (in minutes) from the truncated normal dist
        doc_features['video_length'] = scipy_truncated_normal(_min=MIN_VIDEO_LENGTH,
                                                              _max=MAX_VIDEO_LENGTH,
                                                              _mu=MEAN_VIDEO_LENGTH,
                                                              _sigma=SIGMA_VIDEO_LENGTH,
                                                              size=1)

        # Sample Quality(aka item satisfaction) from the normal distribution weakly conditioned on the item embedding
        quality_mean = np.mean(doc_features["features"])

        # Sample from the normal distribution
        doc_features['quality'] = self._rng.normal(loc=quality_mean,
                                                   scale=SIGMA_QUALITY_SCORE,
                                                   size=self._num_categories)
        return self._doc_ctor(**doc_features)


def clicked_watchtime_reward(responses):
    """Calculates the total clicked watchtime from a list of responses.

    Args:
      responses: A list of IEvResponse objects

    Returns:
      reward: A float representing the total watch time from the responses
    """
    reward = 0.0
    for response in responses:
        if response.clicked:
            reward += response.watch_time
    return reward


def clicked_watchtime_reward_multiuser(responses):
    """Calculates the total clicked watchtime from a list of responses for each user

    Args:
      responses: A nested list of IEvResponse objects for each user

    Returns:
      reward: An array with the size of num_users contains floats representing the total watch time from the responses
              for each user
    """
    num_users = len(responses)
    reward = np.zeros(shape=num_users)
    for _user_id in range(num_users):
        for response in responses[_user_id]:
            if response.clicked:
                reward[_user_id] += response.watch_time
    return reward


def total_clicks_reward(responses):
    """Calculates the total number of clicks from a list of responses.

    Args:
       responses: A list of IEvResponse objects

    Returns:
      reward: A float representing the total clicks from the responses
    """
    reward = 0.0
    for r in responses:
        reward += r.clicked
    return reward


def total_clicks_reward_multiuser(responses):
    """Calculates the total number of clicks from a list of responses.

    Args:
       responses: A list of IEvResponse objects

    Returns:
      reward: n-sized array with floats representing the total clicks from the responses
    """
    num_users = len(responses)
    reward = np.zeros(shape=num_users)
    for _user_id in range(num_users):
        for response in responses[_user_id]:
            reward[_user_id] += response.clicked
    return reward


def launch_user_choice_model(args: dict):
    """Launch a user model and returns the instantiated user choice model"""
    feature_dist = distribution.Distribution(args=args)

    choice_model_ctor = choice_model.choose_UserChoiceModel(
        _type=args.get("type_UserChoiceModel", "SlateDependentChoiceModel")
    )
    user_model = IEvUserModel(slate_size=args.get("slate_size", 5),
                              feat_sample_fn=feature_dist.make_user_sampling_dist(),
                              choice_model_ctor=choice_model_ctor,
                              response_model_ctor=IEvResponse,
                              user_state_ctor=IEvUserState,
                              no_click_mass=args.get("no_click_mass", 3.0),
                              base_SlateChoiceModel=args.get("recsim_base_SlateChoiceModel",
                                                             "ProportionalCascadeChoiceModel"),
                              user_model_metric_if_vector=args.get("recsim_user_model_metric_if_vector", False),
                              user_model_type_constraint=args.get("recsim_user_model_type_constraint", "None"),
                              choice_model_type_constraint=args.get("recsim_choice_model_type_constraint",
                                                                        "None"),
                              choice_model_user_alpha=args.get("recsim_choice_model_user_alpha", 1.0),
                              choice_model_cpr_alpha=args.get("recsim_choice_model_cpr_alpha", 0.1),
                              choice_model_slate_alpha=args.get("recsim_choice_model_slate_alpha", 0.5),
                              choice_model_metric_alpha=args.get("recsim_choice_model_metric_alpha", 2.0),
                              no_click_mass_alpha=args.get("recsim_no_click_mass_alpha", 0.2),
                              no_click_mass_method=args.get("recsim_no_click_mass_method", "None"),
                              choice_model_how_to_apply_metric=args.get("recsim_choice_model_how_to_apply_metric",
                                                                        "add"),
                              if_noClickMass_include=args.get("recsim_if_noClickMass_include", False),
                              type_slateAggFn=args.get("type_slateAggFn", "mean"),
                              time_budget=args.get("recsim_time_budget", 200.0),
                              skip_penalty=args.get("recsim_skip_penalty", 10.0),
                              seed=ENV_RANDOM_SEED)
    return user_model.choice_model


def create_multiuser_environment(args: dict):
    """Creates an interest evolution environment for multiple users."""
    feature_dist = distribution.Distribution(args=args)

    user_models = list()
    choice_model_ctor = choice_model.choose_UserChoiceModel(
        _type=args.get("type_UserChoiceModel", "MultinomialLogitChoiceModel")
    )
    for _ in range(args.get("batch_step_size", 100)):
        user_model = IEvUserModel(slate_size=args.get("slate_size", 5),
                                  feat_sample_fn=feature_dist.make_user_sampling_dist(),
                                  choice_model_ctor=choice_model_ctor,
                                  response_model_ctor=IEvResponse,
                                  user_state_ctor=IEvUserState,
                                  no_click_mass=args.get("no_click_mass", 3.0),
                                  base_SlateChoiceModel=args.get("recsim_base_SlateChoiceModel",
                                                                 "ProportionalCascadeChoiceModel"),
                                  user_model_metric_if_vector=args.get("recsim_user_model_metric_if_vector", False),
                                  user_model_type_constraint=args.get("recsim_user_model_type_constraint",
                                                                          "None"),
                                  choice_model_type_constraint=args.get("recsim_choice_model_type_constraint",
                                                                            "None"),
                                  choice_model_user_alpha=args.get("recsim_choice_model_user_alpha", 1.0),
                                  choice_model_cpr_alpha=args.get("recsim_choice_model_cpr_alpha", 0.1),
                                  choice_model_slate_alpha=args.get("recsim_choice_model_slate_alpha", 0.5),
                                  choice_model_metric_alpha=args.get("recsim_choice_model_metric_alpha", 2.0),
                                  no_click_mass_alpha=args.get("recsim_no_click_mass_alpha", 0.2),
                                  no_click_mass_method=args.get("recsim_no_click_mass_method", "None"),
                                  choice_model_how_to_apply_metric=args.get("recsim_choice_model_how_to_apply_metric",
                                                                            "add"),
                                  if_noClickMass_include=args.get("recsim_if_noClickMass_include", False),
                                  type_slateAggFn=args.get("type_slateAggFn", "mean"),
                                  time_budget=args.get("recsim_time_budget", 200.0),
                                  skip_penalty=args.get("recsim_skip_penalty", 10.0),
                                  seed=ENV_RANDOM_SEED)
        user_models.append(user_model)

    document_sampler = UtilityModelVideoSampler(doc_ctor=IEvVideo, feature_dist=feature_dist, seed=ENV_RANDOM_SEED)
    """ Resampling documents generates the features of new items infinitely.
        So, it is difficult to associate with specific itemIds and causes us to loose the track of all the itemIds
        that have been used during training.
        Thus, when we sample from ReplyBuffer, it's difficult to recover the candidate set at that time.
    """
    _num_items_dict = {
        "all": args["num_allItems"], "train": args["num_trainItems"], "test": args["num_testItems"]
    }
    if args["recsim_if_special_items"]:
        _num_items_dict.update({
            "all_sp": args["num_allItems_sp"], "train_sp": args["num_trainItems_sp"],
            "test_sp": args["num_testItems_sp"]
        })
    ievenv = environment.MultiUserEnvironment(user_model=user_models,
                                              document_sampler=document_sampler,
                                              num_items_dict=_num_items_dict,
                                              slate_size=args.get("slate_size", 5),
                                              # resample_documents=args.get("recsim_resample_documents", False),
                                              resample_documents=False,
                                              if_new_action_env=args.get("recsim_if_new_action_env", True))

    if args.get("recsim_type_base_reward", "click") == "watchtime":
        reward_aggregator = clicked_watchtime_reward_multiuser
    elif args.get("recsim_type_base_reward", "click") == "click":
        reward_aggregator = total_clicks_reward_multiuser
    else:
        raise ValueError

    env = recsim_gym.RecSimGymEnv(raw_environment=ievenv,
                                  reward_aggregator=reward_aggregator,
                                  metrics_aggregator=utils.aggregate_video_cluster_metrics_multiuser,
                                  metrics_writer=utils.write_video_cluster_metrics_multiuser)
    env = wrap_env(env=env, args=args)
    return env
