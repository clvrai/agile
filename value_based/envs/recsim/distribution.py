import numpy as np
from random import shuffle
from copy import deepcopy
from sklearn.metrics.pairwise import euclidean_distances

from value_based.commons.args import ENV_RANDOM_SEED, TRAIN_ITEM_COV_COEFF, TEST_ITEM_COV_COEFF, USER_COV_COEFF


def _helper(_dict: dict):
    """ ref: https://stackoverflow.com/a/15751984/9246727 """
    v = {}
    for key, value in sorted(_dict.items()):
        v.setdefault(value, []).append(key)
    return v


def _helper2(_dict: dict):
    """ (categoryId, itemId_list) -> (main_categoryId, itemId_list) """
    a = {k: v for k, v in _dict.items()}
    _a = dict()
    for k, v in sorted(a.items()):
        if type(k) == int:
            key = k
        else:
            key = k.split("_")[0]
        if int(key) in _a:
            _a[int(key)] += v
        else:
            _a[int(key)] = v
    return _a


class Distribution(object):
    """Supposed to be used in DocumentSampler to instantiate the doc features."""

    def __init__(self, args: dict = {}):
        self._args = args
        self._seed = ENV_RANDOM_SEED
        self._num_allItems = self._args.get("num_allItems", 500)
        self._num_trainItems = self._args.get("num_trainItems", 300)
        self._num_testItems = self._args.get("num_testItems", 200)
        self._sampling_method = self._args.get("recsim_itemFeat_samplingMethod", "rejection_sampling")
        self._sampling_dist = self._args.get("recsim_itemFeat_samplingDist", "GMM")
        self._distance_threshold = self._args.get("recsim_rejectionSampling_distance", 3.0)
        self._if_orthogonal_categories = self._args.get("recsim_if_orthogonal_categories", False)
        self._if_use_subcategory = self._args.get("recsim_if_use_subcategory", False)
        self._subcategory_alpha = self._args.get("recsim_subcategory_alpha", 0.15)
        self._if_debug = self._args.get("if_debug", False)

        # Instantiate the empty components
        self._id2category_dict = dict()
        self._category_master_dict = dict()
        self._main_category_master_dict = dict()
        self._trainItems_dict, self._testItems_dict = dict(), dict()

        if self._if_debug:
            print(f"Distribution>> num_allItems: {self._num_allItems}, "
                  f"num_trainItems: {self._num_trainItems}, num_testItems: {self._num_testItems}")

        self.reset_sampler()

        # === Prep GMM
        self._num_categories = self._args.get("recsim_num_categories", 10)  # Centroids in GMM
        self._num_subcategories = self._args.get("recsim_num_subcategories", 3)  # Centroids in GMM
        self._distribute_centroids()

    def generate_item_representations(self, list_trainItemIds: list, list_testItemIds: list, if_special: bool = False):
        if if_special:
            key_train, key_test = "train_sp", "test_sp"
        else:
            key_train, key_test = "train", "test"

        if self._args.get("recsim_if_new_action_env", True):
            # Assign ids to categories; (k, v): (itemId, categoryId)
            self._id2category_dict[key_train] = self._get_category_master(list_itemIds=list_trainItemIds)
            self._id2category_dict[key_test] = self._get_category_master(list_itemIds=list_testItemIds)

            # Recreate the dict as the category master; (k, v): (categoryId, id_list)
            self._category_master_dict[key_train] = _helper(_dict=self._id2category_dict[key_train])
            self._category_master_dict[key_test] = _helper(_dict=self._id2category_dict[key_test])

            # Dict; (key: categoryId, value: list of itemIds)
            self._main_category_master_dict[key_train] = _helper2(_dict=self._category_master_dict[key_train])
            self._main_category_master_dict[key_test] = _helper2(_dict=self._category_master_dict[key_test])
        else:  # if not new action env, then both train and test item sets must be the same
            # Assign ids to categories; (k, v): (itemId, categoryId)
            self._id2category_dict[key_train] = self._id2category_dict[key_test] = self._get_category_master(
                list_itemIds=list_trainItemIds)

            # Recreate the dict as the category master; (k, v): (categoryId, id_list)
            self._category_master_dict[key_train] = self._category_master_dict[key_test] = _helper(
                _dict=self._id2category_dict[key_train])

            # Dict; (key: categoryId, value: list of itemIds)
            self._main_category_master_dict[key_train] = self._main_category_master_dict[key_test] = _helper2(
                _dict=self._category_master_dict[key_train])

        # ==== Prep to instantiate the class
        if self._sampling_method == "rejection_sampling":
            self._do_rejection_sampling(if_special=if_special)

    @property
    def category_info(self):
        return {"main": self._num_categories, "sub": self._num_subcategories}

    @property
    def if_use_subcategory(self):
        return self._if_use_subcategory

    @property
    def category_master_dict(self):
        return self._category_master_dict

    @property
    def main_category_master_dict(self):
        return self._main_category_master_dict

    @property
    def id2category_dict(self):
        return self._id2category_dict

    def _distribute_centroids(self):
        """ Prepare the centroids of GMM """
        if not self._if_use_subcategory:
            if self._if_orthogonal_categories:
                # num_categories x num_categories
                self._centroids = np.eye(self._num_categories)
            else:
                # num_categories x num_categories
                self._centroids = self._rng.random((self._num_categories, self._num_categories))
        else:
            centroids = dict()  # (key, value): ({_category}_{_subcategory}, itemId)
            _centroids = np.eye(self._num_categories)
            for _category in range(self._num_categories):
                for _subcategory in range(self._num_subcategories):
                    # _c = _centroids[_category] + self._rng.uniform(low=-0.6, high=0.6, size=self._num_categories)
                    _c = _centroids[_category] + self._rng.uniform(low=-0.5, high=0.5, size=self._num_categories)
                    # _c = _centroids[_category] + self._rng.uniform(low=-0.4, high=0.4, size=self._num_categories)
                    # _c = _centroids[_category] + self._rng.uniform(low=-0.3, high=0.3, size=self._num_categories)
                    # _c = _centroids[_category] + self._rng.normal(size=self._num_categories) * self._subcategory_alpha
                    centroids[f"{_category}_{_subcategory}"] = _c
            self._centroids = centroids

    def _do_rejection_sampling(self, if_special: bool = False):
        """ Rejection Sampling """
        if if_special:
            key = "sp"
        else:
            key = "normal"
        if self._sampling_dist == "uniform":
            raise ValueError
        elif self._sampling_dist == "GMM":
            # (k: item_id, v: (category_id, embedding)}
            self._trainItems_dict[key], self._testItems_dict[key] = self._rejection_sampling_GMM(if_special=if_special)
        else:
            raise ValueError

        if not self._args.get("recsim_if_new_action_env", True):
            # if not new action env, then both train and test item sets must be the same
            self._trainItems_dict = deepcopy(self._testItems_dict)

    def _get_category_master(self, list_itemIds: list):
        """ Prepare the dictionary based master table of id and category """
        self._rng.shuffle(list_itemIds)
        _split = np.array_split(ary=list_itemIds, indices_or_sections=self._num_categories)

        if self._if_use_subcategory:
            # Dict: (itemId, {category_id}_{subcategory_id})
            _id2category = dict()
            for _category, __id2category in enumerate(_split):
                ___id2category = np.array_split(ary=__id2category, indices_or_sections=self._num_subcategories)
                for _subcategory, __ids in enumerate(___id2category):
                    for __id in __ids:
                        _id2category[__id] = f"{_category}_{_subcategory}"
        else:
            # Dict: (itemId, category_id)
            _id2category = {_id: _category for _category, _ids in enumerate(_split) for _id in _ids}
        return _id2category

    def reset_sampler(self):
        self._rng = np.random.RandomState(self._seed)

    def sample(self, itemId: int, flg: str, if_special: bool = False):
        """ Returns the sampled feature vector and category_id

        Args:
            flg (str): train or test

        Returns:
            features (np.ndarray):
            category_id (int):
        """
        features, category_id = None, None
        if self._sampling_method == "normal":
            # Features are a vector of real values
            if flg == "train":
                features = self._rng.uniform(low=self._args.get("recsim_itemDist_train_low", -0.6),
                                             high=self._args.get("recsim_itemDist_train_high", 0.6),
                                             size=self._num_categories)
            elif flg == "test":
                # Uniformly sample from two ranges
                left_or_right = self._rng.random() > 0.5
                if left_or_right:
                    features = self._rng.uniform(low=self._args.get("recsim_itemDist_test_low", -1.0),
                                                 high=self._args.get("recsim_itemDist_train_low", -0.6),
                                                 size=self._num_categories)
                else:
                    features = self._rng.uniform(low=self._args.get("recsim_itemDist_train_high", 0.6),
                                                 high=self._args.get("recsim_itemDist_test_high", 1.0),
                                                 size=self._num_categories)
        elif self._sampling_method == "rejection_sampling":
            if if_special:
                key = "sp"
            else:
                key = "normal"
            if flg == "train":
                category_id, features = self._trainItems_dict[key][itemId]  # (k: item_id, v: (category_id, embedding)}
            elif flg == "test":
                category_id, features = self._testItems_dict[key][itemId]  # (k: item_id, v: (category_id, embedding)}
        else:
            raise ValueError
        return features, category_id

    def _rejection_sampling_uniform(self):
        """ Rejection Sampling of features

        Returns:
            item_feats_dict (dict): dict of lists of features vectors
        """
        if self._if_debug:
            print("\nDistribution>> ====== START: Rejection Sampling ======")

        # Uniformly sample train items first!
        train_items = self._rng.uniform(low=-1.0, high=1.0, size=(self._num_trainItems, self._num_categories))

        # Rejection Sampling for test items
        test_items = None
        done = False
        cnt = 0
        while not done:  # Repeat till we finish collecting the specific no. of samples
            # Uniformly sample features; num_testItems x dim_item
            _test_items = self._rng.uniform(low=-1.0, high=1.0, size=(self._num_testItems, self._num_categories))

            # Compute the pairwise distance b/w all the train/test items -> num_trainItems x num_testItems
            dist = euclidean_distances(X=train_items, Y=_test_items)

            # Check if test items are enough away from all the training samples in the item feature space
            _cnt = np.sum(dist > self._distance_threshold, axis=0)

            # Get the samples that satisfied the above condition
            mask = (_cnt == self._num_trainItems)

            # Fill the test_items bucket
            if test_items is None:
                test_items = _test_items[mask, :]
            else:
                test_items = np.vstack([test_items, _test_items[mask, :]])

            if self._if_debug:
                print(f"Distribution>> [Iter: {cnt}] test_items: {test_items.shape}")

            # check if we can finish
            done = test_items.shape[0] >= self._num_testItems
            cnt += 1

        # Process the shape of test_items
        if test_items.shape[0] > self._num_testItems:
            test_items = test_items[:self._num_testItems, :]
        assert test_items.shape[0] == self._num_testItems

        if self._if_debug:
            print(f"Distribution>> Train: {train_items.shape}, Test: {test_items.shape}")
            print("Distribution>> ====== END: Rejection Sampling ======\n")
        item_feats_dict = {"train": train_items.tolist(), "test": test_items.tolist()}
        category_ids_dict = {
            "train": self._rng.randint(low=0, high=self._num_categories, size=self._num_trainItems).tolist(),
            "test": self._rng.randint(low=0, high=self._num_categories, size=self._num_trainItems).tolist()
        }
        return item_feats_dict, category_ids_dict

    def _rejection_sampling_GMM(self, if_special: bool = False):
        """ Rejection Sampling of features

        Returns:
            (k: item_id, v: (category_id, embedding)}
        """
        # Sample from GMM;
        if if_special:
            key_train = "train_sp"
            key_test = "test_sp"
        else:
            key_train = "train"
            key_test = "test"

        trainItems_dict = {}  # (k: item_id, v: (category_id, embedding)}
        _trainItems_dict_temp = {}  # (k: category_id, v: embedding}
        for _category_id, _list in self._category_master_dict[key_train].items():  # (k, v): (categoryId, id_list)
            # We use the averaged representation over items in a category for special items, ie., just centroids!!
            _emb = self._sample_from_GMM(centroid=_category_id,
                                         num_samples=len(_list),
                                         cov_coeff=0.0 if if_special else TRAIN_ITEM_COV_COEFF)
            _trainItems_dict_temp[_category_id] = _emb.copy()
            _emb = _emb.tolist()
            for _itemid in _list:
                trainItems_dict[_itemid] = (_category_id, _emb.pop(0))

        # Rejection Sampling for test items; (k: item_id, v: (category_id, embedding)}

        testItems_dict = {}
        for _category_id, _list in sorted(self._category_master_dict[key_test].items()):
            """ === Rejection Sampling for a category === """
            if self._if_debug: print(f"\nDistribution>> === CATEGORY: {_category_id}")
            _test_items, done, cnt = None, None, 0
            while not done:  # Repeat till we finish collecting the specific no. of samples
                # Sample features from GMM based on the category; num_testSamples x num_categories
                # We use the averaged representation over items in a category for special items, ie., just centroids!!
                _samples = self._sample_from_GMM(centroid=_category_id,
                                                 num_samples=len(_list),
                                                 cov_coeff=0.0 if if_special else TEST_ITEM_COV_COEFF)

                if not if_special:  # for normal items we use the rejection sampling
                    # Compute the pairwise distance b/w all the train/test items in the category
                    dist = euclidean_distances(X=_trainItems_dict_temp[_category_id],
                                               Y=_samples)  # num_trainSamples x num_testSamples

                    # Check if test items are different enough from all the training samples in the item feature space
                    _cnt = np.sum(dist > self._distance_threshold, axis=0)  # (num_trainSamples)-size array

                    # Get the samples that satisfied the above condition
                    mask = (_cnt == len(self._category_master_dict["train"][_category_id]))

                    # Fill the test_items bucket
                    if _test_items is None:
                        _test_items = _samples[mask, :]
                    else:
                        _test_items = np.vstack([_test_items, _samples[mask, :]])

                    if self._if_debug:
                        print(f"Distribution>> [Iter: {cnt}] mean dist: {np.mean(dist): .5f} accepted samples: {sum(mask)} "
                              f"test_items: {_test_items.shape}")

                    # check if we can finish
                    done = _test_items.shape[0] >= len(_samples)
                else:  # for special items we don't use the rejection sampling
                    _test_items = _samples
                    done = True
                cnt += 1

            """ === After Rejection Sampling for a category === """
            del _trainItems_dict_temp[_category_id]
            # Process the shape of test_items
            if _test_items.shape[0] > len(_list):
                _test_items = _test_items[:len(_list), :]
            assert _test_items.shape[0] == len(_list)

            # (k: item_id, v: (category_id, embedding)}
            _test_items = _test_items.tolist()
            for _itemid in _list:
                testItems_dict[_itemid] = (_category_id, _test_items.pop(0))

        return trainItems_dict, testItems_dict

    def _sample_from_GMM(self, centroid, num_samples: int = 100, cov_coeff: float = 0.2):
        """ Sample from GMM

        Args:
            centroid (int or str): Key to access the centroid in centroids(dict)
            num_samples (int):
            cov_coeff (float):

        Returns:
            feat_mat (np.ndarray): num_samples x num_categories
        """
        # Get the mean vector
        if self._if_use_subcategory:
            # _centroids: (key, value): ({_category}_{_subcategory}, itemId)
            mu = self._centroids[centroid]
        else:
            mu = self._centroids[centroid, :]

        cov_mat = np.eye(self._num_categories, ) * cov_coeff
        feat_mat = self._rng.multivariate_normal(mean=mu, cov=cov_mat, size=num_samples)
        return feat_mat

    def make_user_sampling_dist(self):
        def _fn():
            # === Method 1: Sample a user who has the particular preference over one category
            # Sample a category for a user to prefer
            category_id = self._rng.choice(a=self._num_categories, size=1)[0]
            if self._if_use_subcategory:
                # Sample a user embedding by averaging the interest over sub-categories under the sampled main category
                _vec = list()
                for _sub_category in range(self._num_subcategories):
                    _category_id = f"{category_id}_{_sub_category}"
                    __vec = self._sample_from_GMM(centroid=_category_id, num_samples=1, cov_coeff=USER_COV_COEFF)[0]
                    _vec.append(__vec)
                _vec = np.mean(_vec, axis=0)
            else:
                # sample a weight vector from GMM
                _vec = self._sample_from_GMM(centroid=category_id, num_samples=1, cov_coeff=USER_COV_COEFF)[0]
            return _vec

        return _fn


def test():
    print("=== test ===")

    # hyper-params
    args = {
        "recsim_if_new_action_env": False,
        "recsim_num_categories": 10,
        "recsim_num_subcategories": 10,
        "recsim_rejectionSampling_distance": 0.2,
        "num_allItems": 1000,
        "num_trainItems": 500,
        "num_testItems": 500,
        "if_debug": True,
        # "recsim_if_use_subcategory": True,
        "recsim_if_use_subcategory": False,
    }

    list_trainItemIds = list(range(args["num_trainItems"]))
    list_testItemIds = list(range(args["num_trainItems"], args["num_testItems"] + args["num_trainItems"]))

    for recsim_itemFeat_samplingMethod, recsim_itemFeat_samplingDist in [
        # ("normal", None),
        ("rejection_sampling", "GMM"),
        # ("rejection_sampling", "uniform"),
    ]:
        print(f"=== recsim_itemFeat_samplingMethod: {recsim_itemFeat_samplingMethod}, "
              f"recsim_itemFeat_samplingDist: {recsim_itemFeat_samplingDist}")
        args["recsim_itemFeat_samplingMethod"] = recsim_itemFeat_samplingMethod
        args["recsim_itemFeat_samplingDist"] = recsim_itemFeat_samplingDist

        dist = Distribution(args=args)
        dist.generate_item_representations(list_trainItemIds=list_trainItemIds, list_testItemIds=list_testItemIds)

        print("=== Test: user sampling method ===")
        fn = dist.make_user_sampling_dist()
        print(f"User Interest Vec: {fn()}")

        for train_test in [
            "train",
            "test"
        ]:
            print(train_test)
            for i in range(args[f"num_{train_test}Items"]):
                doc_feature, category_id = dist.sample(flg=train_test)
                # print(f"itemId: {i}, category: {category_id}, feature: {doc_feature}")


def test_GMM():
    # hyper-params
    args = {
        "random_seed": 1,
        "recsim_num_categories": 20,
        "recsim_rejectionSampling_distance": 0.2,
        "recsim_itemFeat_samplingMethod": "rejection_sampling",
        "recsim_itemFeat_samplingDist": "GMM",
        "num_allItems": 1000,
        "num_trainItems": 500,
        "num_testItems": 500,
        "if_debug": True
    }
    dist = Distribution(args=args)

    for centroid in range(args["recsim_num_categories"]):
        result = dist._sample_from_GMM(centroid=centroid, num_samples=args["num_trainItems"])
        print(f"centroid: {centroid}, mean of sampled features: {result.mean(axis=0)}")


if __name__ == '__main__':
    # test_GMM()
    test()
