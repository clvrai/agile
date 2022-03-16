import numpy as np

from value_based.commons.utils import softmax
from value_based.envs.dataset.user_model.reward_model import launch_RewardModel
from value_based.envs.recsim.environments.interest_evolution import launch_user_choice_model
from value_based.commons.args import SKIP_TOKEN, ENV_RANDOM_SEED
from value_based.embedding.base import BaseEmbedding
from value_based.commons.test import TestCase as test
from value_based.commons.utils import min_max_scale


SCORE_MIN = 0.7


class UserModel(object):
    def __init__(self, item_embedding: BaseEmbedding, args: dict):
        self._args = args
        self._rng = np.random.RandomState(ENV_RANDOM_SEED)
        self._device = self._args.get("device", "cpu")
        self._batch_step_size = args.get("batch_step_size", 16)
        self._slate_size = self._args.get("slate_size", 10)
        self._history_size = self._args.get("history_size", 5)

        self.item_embedding = item_embedding
        self.reward_model = launch_RewardModel(args=self._args)
        self.choice_model = launch_user_choice_model(args=self._args)

    def predict(self, obs: np.ndarray, slate: np.ndarray, metrics: dict = None):
        """ Estimate a reward and user's response

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs(dim_user + dim_item)
            slate(np.ndarray): batch_size x slate_size

        Returns:
            reward: (batch_size)-size array
            a_user: (batch_size)-size array
        """
        # get the item-embedding for items in a slate; batch_step_size x slate_size x dim_item
        item_embedding = self.item_embedding.get(index=slate, if_np=True)

        # get the scores for items in a slate; batch_step_size x slate_size
        score = self.reward_model.compute_score(obs=obs, item_embedding=item_embedding)
        score = score.ravel()

        # get the user action as index in slate; (batch_step_size)-size array
        # a_user_ind = list()
        # for i in range(self._batch_step_size):
        #     # get the metrics for a user
        #     _metrics = dict()
        #     if metrics["metric"] is not None:
        #         _metrics["metric"] = metrics["metric"][i]
        #     if metrics["cpr_metric"] is not None:
        #         _metrics["cpr_metric"] = metrics["cpr_metric"][i]
        #     self.choice_model.score_items(base_scores=np.ones(self._args["slate_size"]) * score[i], **_metrics)
        #     _a_user_ind = self.choice_model.choose_item()
        #     a_user_ind.append(_a_user_ind)
        # # Recover the itemId in an index-based user action and Compute the reward based on the user action
        # a_user, reward = list(), list()
        # for _slate, _a_user_ind in zip(slate, a_user_ind):
        #     if _a_user_ind != SKIP_TOKEN:
        #         itemId = _slate[_a_user_ind]
        #         reward.append(1)
        #     else:
        #         itemId = SKIP_TOKEN
        #         reward.append(0)
        #     a_user.append(itemId)

        a_user, reward = list(), list()
        for i in range(self._batch_step_size):
            # get the metrics for a user
            # if self._rng.random() < score[i]:
            _score = score[i]
            if _score > SCORE_MIN:  # If it's too high then rescale the predicted score
                _score = min_max_scale(x=score[i], _min=SCORE_MIN, _max=1.0)
            if self._rng.random() < _score:
                _a_user_ind = self._rng.choice(a=slate[i, :], size=1)[0]
                reward.append(1)
            else:
                _a_user_ind = SKIP_TOKEN
                reward.append(0)
            a_user.append(_a_user_ind)
        return np.asarray(reward), np.asarray(a_user)


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        self.args.env_name = "sample"
        self.args.type_UserChoiceModel = "SlateDependentChoiceModel"
        self.args.recsim_reward_characteristic = "specificity"
        self._prep()

    def test(self):
        for model_name in [
            # "None",
            "sample/offline.pkl",
        ]:
            if model_name:
                self.args.rm_weight_path = "../../../../trained_weight/reward_model/" + model_name

            self.user_model = UserModel(item_embedding=self.dict_embedding["item"], args=vars(self.args))
            print("=== Model: {} ===".format(model_name))
            print("=== Test: predict ===")
            self._predict()

    def _predict(self):
        metrics = {"metric": np.random.randn(self.args.batch_step_size),
                   "cpr_metric": np.random.randn(self.args.batch_step_size)}
        obses = self.obses.make_obs(dict_embedding=self.dict_embedding)
        reward, a_user = self.user_model.predict(obs=obses, slate=self.actions, metrics=metrics)
        print(reward.shape, a_user.shape)
        print(reward, a_user)


if __name__ == '__main__':
    Test().test()
