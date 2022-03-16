""" Metrics class to manage all the metrics for experiments """
import numpy as np
from copy import deepcopy
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from value_based.embedding.base import BaseEmbedding
from value_based.commons.test import TestCase
from value_based.commons.metrics_constants import get_minmax
from value_based.commons.utils import min_max_scale

# original one
# METRIC_NAMES = ["hit_rate", "mrr", "ndcg", "intrasession_diversity", "coverage", "shannon_entropy", "specificity"]
# new one
METRIC_NAMES = ["hit_rate", "mrr", "ndcg", "intrasession_diversity", "coverage", "shannon_entropy", "specificity",
                "distance_to_user", "user_stats", "cpr_score"]


class Metrics(object):
    def __init__(self, item_embedding: BaseEmbedding = None, metric_names=METRIC_NAMES, args: dict = {}):
        self._metric_names = metric_names
        self._args = args
        self._batch_step_size = self._args.get("batch_step_size", 32)
        self._slate_size = self._args.get("slate_size", 5)
        self._env_name = self._args.get("env_name", "recsim")
        self.item_embedding = item_embedding

        # Metrics is computed for each user then we average the metrics over users to get the metrics in an episode
        self._bucket = {_metric_name: np.zeros(shape=self._batch_step_size) for _metric_name in self._metric_names}
        self._bucket_prev = deepcopy(self._bucket)
        self._session_cnt = np.zeros(shape=self._batch_step_size)

        # Keep track the occurrence of items in an episode; num_users x num_items
        self._occurrence_counter_dict = {}  # key: itemId, value: occurrence
        self._rec_count_table = np.zeros((self._batch_step_size, self.item_embedding.shape[0]))
        self._click_item_counter = np.zeros(self.item_embedding.shape[0])

        # Init Action Buffer
        self._action_buffer = {k: list() for k in range(self._batch_step_size)}

        # Get the caps of metrics
        self._metric_caps = get_minmax(slate_size=self._slate_size,
                                       num_categories=self._args.get("recsim_num_categories"))

    def count_occurrence(self, pred_slate, _id):
        """ Count the occurrence of items in a slate to compute the diversity related metrics later """
        for item in pred_slate:
            self._rec_count_table[_id][item] += 1.0
            if item not in self._occurrence_counter_dict.keys():
                self._occurrence_counter_dict[item] = 1.0
            else:
                self._occurrence_counter_dict[item] += 1.0

    @property
    def session_cnt(self):
        """ To compute the episode reward per user accurately! """
        return self._session_cnt

    @property
    def occurrence_counter_dict(self):
        return self._occurrence_counter_dict

    @property
    def rec_count_table(self):
        return self._rec_count_table

    @property
    def click_item_counter(self):
        return self._click_item_counter

    def hit_rate(self, **kwargs):
        """ Same as Recall

        Returns:
            this returns either 1 or 0
        """
        if isinstance(kwargs["gt_item"], np.int64):
            if kwargs["gt_item"] in kwargs["pred_slate"]:
                return 1
            return 0
        elif kwargs["gt_item"] == -1:  # -1 is not included in np.int so that we need this!
            return 0
        elif isinstance(eval(kwargs["gt_item"]), list):
            for i in eval(kwargs["gt_item"]):
                if i in kwargs["pred_slate"]:
                    return 1
            return 0
        else:
            raise NotImplementedError

    def mrr(self, **kwargs):
        """ Mean Reciprocal Rank """
        if kwargs["gt_item"] in kwargs["pred_slate"]:
            _rank = np.where(kwargs["pred_slate"] == kwargs["gt_item"])[0]
            try:
                ret = float(1.0 / int(_rank + 1) * 1.0)
            except:
                print(f"Error: Metrics>> gt_item: {kwargs['gt_item']}, pred_slate: {kwargs['pred_slate']}")
                raise ValueError
            return ret
        return 0.0

    def cg(self, **kwargs):
        """ Cumulative Gain """
        return np.sum(kwargs["rel_list"])

    def dcg(self, **kwargs):
        """ Discounted Cumulative Gain """
        _res = 0.0
        for i in range(len(kwargs["rel_list"])):
            _res += kwargs["rel_list"][i] / np.log(i + 1)
        return _res

    def ndcg(self, **kwargs):
        """ Normalised Discounted Cumulative Gain """
        if kwargs["gt_item"] in kwargs["pred_slate"]:
            index = np.where(kwargs["pred_slate"] == kwargs["gt_item"])[0]
            return np.asscalar(np.reciprocal(np.log2(index + 2)))
        return 0.0

    def intersession_diversity(self):
        _sorted = dict(sorted(self._occurrence_counter_dict.items(), key=lambda item: item[1]))
        num_items = len(_sorted.keys())
        index = np.asarray(list(_sorted.keys()))
        _sorted = np.asarray(list(_sorted.values()))
        gini = 2 * np.sum((num_items + 1 - index) / (num_items + 1) * _sorted / np.sum(_sorted))
        return gini

    def intrasession_diversity(self, **kwargs):
        """ Intra-session Diversity; item-similarity based diversity using Cosine similarity
            Range is [SimilarItems, DissimilarItems] -> [0.0, 2.0]

        References
            - See Eq(9) in D2RL, L.Yong et al., 2019(https://arxiv.org/pdf/1903.07826.pdf)
                - they used the Jaccard similarity but we use the Cosine similarity
        """
        if self.item_embedding is None: return 0.0
        _embedding = self.item_embedding.get(index=kwargs["pred_slate"], if_np=True)  # slate_size x dim_item
        _similarity = cosine_similarity(_embedding, _embedding)  # slate_size x slate_size

        if kwargs["info"].get("recsim_metric_if_vector", False):
            # Aggregate the pairwise similarities for each item; (slate_size)-size array
            _similarity = _similarity.sum(axis=-1)
            _similarity -= 1.0  # Subtract the cosine similarity to oneself which is 1.0
            _diversity = 1.0 / _similarity
        else:
            _similarity = _similarity.sum()  # Aggregate the pairwise similarities
            _similarity -= self._slate_size  # Subtract the diagonal elements; cosine similarity to oneself is 1.0
            _similarity /= 2.0  # because it's the symmetric matrix

            # Inverse the similarity to get the diversity
            if self._slate_size == 1:
                _diversity = 1 - (2 / self._slate_size * _similarity)
            else:
                _diversity = 1 - (2 / (self._slate_size * (self._slate_size - 1)) * _similarity)

        if not kwargs.get("if_specificity", False):
            _diversity = min_max_scale(x=_diversity,
                                       _min=self._metric_caps["min_diversity"],
                                       _max=self._metric_caps["max_diversity"])
        return _diversity

    def shannon_entropy(self, **kwargs):
        """ Shannon Entropy of items based on category """
        # One-hot vector of item category; batch_step_size x slate_size x num_categories
        if "category_mat" not in kwargs["info"]: return 0.0
        if len(kwargs["info"]["category_mat"].shape) == 3:
            # Get the category of items presented to an active user
            category_mat = kwargs["info"]["category_mat"][kwargs["info"]["index"], ...]  # slate_size x num_categories
        elif len(kwargs["info"]["category_mat"].shape) == 2:
            category_mat = kwargs["info"]["category_mat"]  # num_categories x num_subcategories(or 1)
        else:
            raise ValueError()
        category_mat = np.sum(category_mat, axis=-1)
        p = category_mat / self._slate_size  # (num_categories)-size array
        _mask = p == 0.0
        p[_mask] = 10 ** (-10)  # Avoid the warning in np.log

        if kwargs["info"].get("recsim_metric_if_vector", False):
            _entropy = -p * np.log2(p)
            _entropy[_mask] = 0.0
            if self._args.get("recsim_if_use_subcategory", False):
                # category_vec(slate_size x 2) -> [col1: main-category, col2: sub-category]
                _entropy = _entropy[kwargs["info"]["category_vec"][:, 0]]  # (slate_size)-size array
            else:
                # category_vec: (slate_size)-size array
                _entropy = _entropy[kwargs["info"]["category_vec"]]  # (slate_size)-size array
        else:
            _entropy = -np.sum(p * np.log2(p))  # scalar

        if not kwargs.get("if_specificity", False):
            _entropy = min_max_scale(x=_entropy,
                                     _min=self._metric_caps["min_entropy"],
                                     _max=self._metric_caps["max_entropy"])
        return _entropy

    def specificity(self, **kwargs):
        """ Inverse of Diversity """
        if self._args.get("recsim_type_specificity", "entropy") == "entropy":
            min_spec, max_spec = self._metric_caps["min_spec_ent"], self._metric_caps["max_spec_ent"]
            _diversity = self.shannon_entropy(**kwargs, if_specificity=True)
        elif self._args.get("recsim_type_specificity", "entropy") == "cosine":
            min_spec, max_spec = self._metric_caps["min_spec_div"], self._metric_caps["max_spec_div"]
            _diversity = self.intrasession_diversity(**kwargs, if_specificity=True)
        else:
            raise ValueError

        if type(_diversity) == np.ndarray:
            _mask = _diversity == 0.0
            _diversity[_mask] = 10 ** (-10)  # Avoid the warning in np.log
        elif type(_diversity) == np.float64:
            _diversity = 10 ** (-10) if _diversity == 0.0 else _diversity  # Avoid the warning in np.log
        elif type(_diversity) == float:
            _diversity = 10 ** (-10) if _diversity == 0.0 else _diversity  # Avoid the warning in np.log

        _specificity = 1.0 / _diversity  # Specificity is the inverse of the diversity
        _specificity = min_max_scale(x=_specificity, _min=min_spec, _max=max_spec)
        return _specificity

    def gini_index(self, **kwargs):
        """ Gini Index of items based on category """
        # One-hot vector of item category; batch_step_size x slate_size x num_categories
        if len(kwargs["info"]["category_mat"].shape) == 3:
            # Get the category of items presented to an active user
            category_mat = kwargs["info"]["category_mat"][kwargs["info"]["index"], ...]  # slate_size x num_categories
        elif len(kwargs["info"]["category_mat"].shape) == 2:
            category_mat = kwargs["info"]["category_mat"]  # num_categories x num_subcategories(or 1)
        else:
            raise ValueError("Metrics[shannon-entropy]>>")
        category_mat = np.sum(category_mat, axis=-1)
        p = category_mat / self._slate_size
        p[p == 0.0] = 10 ** (-10)  # Avoid the warning in np.log
        _gini = 1 - np.sum(np.square(p))
        return _gini

    def novelty(self):
        pass
        # _popularity = n_item_interactions / n_total_interaction_across_all_items
        # _novelty = - np.log2(_popularity)
        # return _novelty

    def coverage(self, **kwargs):
        return np.sum(self._rec_count_table[kwargs["user_id"]] > 0.0) / float(self.item_embedding.shape[0])

    def _other_minors(self, **kwargs):
        # Result buffer
        result = {}

        # Get the distance of slate to a user in the item embedding space
        result.update(self.distance_to_user(**kwargs))

        # Collect user stats
        result.update(self.user_stats(**kwargs))

        # Compute the CPR score
        result.update(**self.cpr_score(**kwargs))
        return result

    def distance_to_user(self, **kwargs):
        """ Compute the distances for each user to the given slate """
        # Store the results
        result = {}
        if kwargs["info"]["user_state"] is not None:
            _user_interest = kwargs["info"]["user_state"]["user_interests"][kwargs["user_id"]]
            _slate = self.item_embedding.get(index=kwargs["pred_slate"], if_np=True)

            # Euclidean distance
            _dist_euc = euclidean_distances(X=_user_interest[None, :], Y=_slate).ravel()

            # Dot Product
            _dist_dp = [np.dot(_user_interest, _item) for _item in _slate]

            # NOTE: Open below when needed
            # result["distance_to_user_euclidean_mean"] = np.mean(_dist_euc)
            result["distance_to_user_euclidean_min"] = np.min(_dist_euc)  # important one
            # result["distance_to_user_euclidean_max"] = np.max(_dist_euc)
            # result["distance_to_user_dot_product_mean"] = np.mean(_dist_dp)
            # result["distance_to_user_dot_product_min"] = np.min(_dist_dp)
            result["distance_to_user_dot_product_max"] = np.max(_dist_dp)  # important one
        else:
            # NOTE: Open below when needed
            # result["distance_to_user_euclidean_mean"] = 0.0
            result["distance_to_user_euclidean_min"] = 0.0
            # result["distance_to_user_euclidean_max"] = 0.0
            # result["distance_to_user_dot_product_mean"] = 0.0
            # result["distance_to_user_dot_product_min"] = 0.0
            result["distance_to_user_dot_product_max"] = 0.0
        return result

    def user_stats(self, **kwargs):
        if kwargs["info"]["user_state"] is not None:
            return {"time_budget": kwargs["info"]["user_state"]["time_budget"][kwargs["user_id"]]}
        else:
            return {"time_budget": 0.0}

    def cpr_score(self, **kwargs):
        """ Complementary Product Recommendation Score """
        # Make the temp kwargs to modify in this method!
        _kwargs = deepcopy(kwargs)

        # batch_step_size x num_categories x num_subcategories or num_categories x num_subcategories
        category_mat = _kwargs["info"]["category_mat"]
        if len(category_mat.shape) == 3:  # if batch_step_size x num_categories x num_subcategories
            category_mat = category_mat[_kwargs["info"]["index"]]  # num_categories x num_subcategories

        # Specificity of main-category(= Purity of items in slate based on main category)
        _kwargs["info"]["category_mat"] = category_mat
        main_score = self.specificity(**_kwargs)  # either scalar or slate_size-size vector

        # Compute the score of subcategories for each main-category by aggregating the corresponding subcategories
        sub_scores = list()
        for vec_subcategory in category_mat:  # for each main category
            # Reshape the row representing the count of the occurrence of subcategories in a main category
            _kwargs["info"]["category_mat"] = vec_subcategory[:, None]
            _kwargs["info"]["recsim_metric_if_vector"] = False

            # Diversity of sub-category(= Impurity of items in slate based on subcategory)
            if self._args.get("recsim_type_cpr_diversity_subcategory", "entropy") == "entropy":
                _sub_score = self.shannon_entropy(**_kwargs)
            elif self._args.get("recsim_type_cpr_diversity_subcategory", "entropy") == "cosine":
                _sub_score = self.intrasession_diversity(**_kwargs)
            else:
                raise ValueError
            sub_scores.append(_sub_score)
        sub_scores = np.asarray(sub_scores)

        if kwargs["info"].get("recsim_metric_if_vector", False):
            """ Note
            Since we compute the subcategory score for each main-category by aggregating the sub-categories for each
            main-category, we access those scores by the item's main-category
            """
            # category_vec(slate_size x 2) -> [col1: main-category, col2: sub-category]
            if self._args.get("recsim_cpr_op", "add") == "add":
                cpr_score = main_score + sub_scores[_kwargs["info"]["category_vec"][:, 0]]
            elif self._args.get("recsim_cpr_op", "add") == "multiply":
                cpr_score = main_score * sub_scores[_kwargs["info"]["category_vec"][:, 0]]
            elif self._args.get("recsim_cpr_op", "add") == "subtract":
                # Since diversity_sub >> specificity_main is empirically known, we can do this!
                cpr_score = sub_scores[_kwargs["info"]["category_vec"][:, 0]] - main_score
            else:
                raise ValueError
        else:
            if self._args.get("recsim_cpr_op", "add") == "add":
                cpr_score = main_score + np.sum(sub_scores)
            elif self._args.get("recsim_cpr_op", "add") == "multiply":
                cpr_score = main_score * np.sum(sub_scores)
            elif self._args.get("recsim_cpr_op", "add") == "subtract":
                # Since diversity_sub >> specificity_main is empirically known, we can do this!
                cpr_score = np.sum(sub_scores) - main_score
            else:
                raise ValueError

        if kwargs["info"].get("visualise_metric", False):
            return {"cpr_score": cpr_score, "diversity_sub": np.sum(sub_scores)}
        else:
            return cpr_score

    def update_metrics(self, gt_items: np.ndarray, pred_slates: np.ndarray, info: dict):
        """ Compute all the metrics and store them in the dict(ie., self._bucket)

        Args:
            gt_items (np.ndarray): (batch_step_size)-size array
            pred_slates (np.ndarray): batch_step_size x slate_size
            info (dict)
        """
        gt_items, pred_slates = np.asarray(gt_items, dtype=np.int64), np.asarray(pred_slates, dtype=np.int64)
        assert len(info["active_userIds"]) == gt_items.shape[0] == pred_slates.shape[0]

        # Increment only the ones of Active Users
        self._session_cnt[np.asarray(info["active_userIds"], dtype=np.int64)] += 1.0
        for _index, (_user_id, gt_item, pred_slate) in enumerate(zip(info["active_userIds"], gt_items, pred_slates)):
            # Record action
            self._action_buffer[_user_id].append(pred_slate.tolist())

            # Set the index of a line of the log
            info["index"] = _index

            # Compute the metrics
            for _name in self._metric_names:
                _metric = eval("self." + _name.lower())(gt_item=gt_item,
                                                        pred_slate=pred_slate,
                                                        user_id=_user_id,
                                                        info=info)
                if type(_metric) is not dict:
                    self._bucket[_name][_user_id] += _metric
                else:
                    for _name, value in _metric.items():
                        if _name not in self._bucket:
                            self._bucket[_name] = np.zeros(shape=self._batch_step_size)
                        self._bucket[_name][_user_id] += value

            # Update the recommended items in a slate for each user
            self.count_occurrence(pred_slate=pred_slate, _id=_user_id)

            # Register the clicked items across users
            self._click_item_counter[gt_item] += 1

    def get_metrics(self):
        """ Average the metrics over number of sessions and return

        Returns:
            _metrics (dict): dict of metrics
        """
        # Intra-session metrics
        if all(self._session_cnt == 0):  # edge case
            return {k: 0.0 for k, _ in self._bucket.items()}

        """ This metric computation bit is tricky, I guess, so that be careful!! """
        # === Get the average of all the metrics
        # Average over time-steps; np.divide is to deal with nan generated from division by zero warning in numpy
        result = {
            _name: np.divide(_value,
                             self._session_cnt,
                             out=np.zeros_like(self._session_cnt),
                             where=self._session_cnt != 0)
            for _name, _value in self._bucket.items()
        }
        result["num_clicks"] = self._bucket["hit_rate"]
        result = {k: np.sum(v) / self._batch_step_size for k, v in result.items()}  # Average over users

        # Inter-session Diversity; occurrence based diversity
        if "intrasession_diversity" in self._metric_names:
            result["intersession_diversity"] = self.intersession_diversity()
        return result

    def get_metrics_as_reward(self):
        # Get the step-wise metrics as rewards
        _metrics = {
            _name: _value[:, None] - _value_prev[:, None]
            for (_name, _value), (_, _value_prev) in zip(self._bucket.items(), self._bucket_prev.items())
        }
        # Update the bucket
        self._bucket_prev = deepcopy(self._bucket)
        return _metrics

    def reset(self):
        """ Reset the metrics """
        self._session_cnt = np.zeros(shape=self._batch_step_size)
        self._bucket = {_metric_name: np.zeros(shape=self._batch_step_size) for _metric_name in self._metric_names}
        self._bucket_prev = deepcopy(self._bucket)
        self._action_buffer = {k: list() for k in range(self._batch_step_size)}

    def set_item_embedding(self, item_embedding: BaseEmbedding):
        """ Set the item-embedding to compute metrics

        Args:
            item_embedding (BaseEmbedding):
        """
        self.item_embedding = item_embedding

    def metric_fn_factory(self, metric_name: str = "intrasession_diversity"):
        """ The produced metric_fn will be used to compute the metric for computing reward not for visualisation! """
        if metric_name != "None":
            # Get the base function to compute a metric
            __fn = eval("self." + metric_name.lower())

            # Base function to provide a metric
            def _fn(**kwargs):
                assert "user_id" in kwargs["info"]
                pred_slate = kwargs.get("pred_slate", None)
                gt_item = kwargs.get("gt_item", None)
                info = kwargs.get("info", None)
                info["visualise_metric"] = False
                return __fn(gt_item=gt_item, pred_slate=pred_slate, user_id=info["user_id"], info=info)
        else:
            _fn = None
        return _fn


class Test(TestCase):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        # self.args.env_name = "ml-100k"
        self.args.env_name = "recsim"
        self._prep()

    def test(self):
        print("=== test ===")
        self.args.batch_step_size = 2

        # test case;
        # num_users = 2 but one of them became inactive after t = 2
        active_userIds = np.asarray([
            [0, 1],  # t = 1
            [0, 1],  # t = 2
            [0],  # t = 3
        ])
        gt_items = np.asarray([
            [0, 1],  # t = 1
            [-1, 3],  # t = 2
            [4],  # t = 3
        ])
        # slate_size = 5
        pred_slates = np.asarray([
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],  # t = 1
            [[0, 1, 2, 3, 4], [0, 1, 2, 3, 4]],  # t = 2
            [[0, 1, 2, 3, 4]],  # t = 3
        ])

        # for test purpose... and not proper way to get the categories
        category_mat = np.eye(self.args.recsim_num_categories)
        metrics = Metrics(item_embedding=self.item_embedding, args=vars(self.args))
        # single_metric_fn = metrics.metric_fn_factory()
        single_metric_fn = metrics.metric_fn_factory(metric_name="shannon_entropy")
        for _active_userIds, _gt_items, _pred_slates in zip(active_userIds, gt_items, pred_slates):
            _categories = np.asarray([category_mat[_slate] for _slate in _pred_slates])
            info = {
                "active_userIds": _active_userIds,
                "categories": _categories,
                "user_state": {
                    "user_interests": np.random.randn(len(_active_userIds), self.args.recsim_num_categories),
                    "time_budget": np.ones(len(_active_userIds)) * 0.2
                }
            }
            metrics.update_metrics(gt_items=_gt_items, pred_slates=_pred_slates, info=info)
            for _id, (_gt_item, _pred_slate) in enumerate(zip(_gt_items, _pred_slates)):
                _info = {
                    "user_id": _id,
                    "categories": _categories
                }
                single_metric = single_metric_fn(gt_item=_gt_item, pred_slate=_pred_slate, info=_info)
                print(single_metric)
            print(metrics.get_metrics())

        print("=== Reset test ===")
        metrics.reset()
        print(metrics.get_metrics())

        print("=== Check click counts ===")
        print(metrics.click_item_counter.shape, metrics.click_item_counter.sum())


if __name__ == '__main__':
    Test().test()
