""" All miscellaneous util fns """
import os
import json
import datetime
import scipy.stats
import numpy as np
import pandas as pd
from math import ceil

from value_based.commons.args import ENV_RANDOM_SEED


def create_directory_name(dir_name):
    """ To safely save the outcomes of an experiment """
    _time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    _dir_name = os.path.join(dir_name, _time)
    # make the directory if it doesn't exist yet
    if not os.path.exists(_dir_name):
        os.makedirs(_dir_name)
    return _dir_name


def create_log_file(dir_name, args):
    """ To save the args of an experiment """
    with open(os.path.join(dir_name, "args.json"), "w") as file:
        json.dump(vars(args), file)


def logging(*msg):
    print("{}>".format(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')), *msg)


def float_round(num, n_decimal=0):
    """ Ref: https://stackoverflow.com/a/9232628/9246727 """
    return ceil(float(num) * (10 ** n_decimal)) / float(10 ** n_decimal)


def running_mean(_arr, window_size=5):
    """ For visualisation """
    cumsum = np.cumsum(np.insert(_arr, 0, 0))
    return (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)


def ema(values, alpha: float):
    values = np.array(values)
    return pd.DataFrame(values).ewm(alpha=alpha).mean().values.ravel()


def _test_logging():
    print("=== test logging ===")
    logging("a"
            "b"
            "c")


def mean_dict(_list_dict: list):
    result = {}
    for d in _list_dict:
        for k in d.keys():
            result[k] = result.get(k, 0) + d[k]

    for k, v in result.items():
        result[k] = float(v) / float(len(_list_dict))
    return result


def scipy_truncated_normal(_min, _max, _mu, _sigma, size=1):
    X = scipy.stats.truncnorm((_min - _mu) / _sigma, (_max - _mu) / _sigma, loc=_mu, scale=_sigma)
    X.random_state = np.random.RandomState(ENV_RANDOM_SEED)  # To fix the behaviour of Env
    res = X.rvs(size=size)
    if size == 1:
        res = res[0]
    return res


def softmax(_vec):
    """Computes the softmax of a vector."""
    normalized_vector = np.array(_vec) - np.max(_vec)  # For numerical stability
    return np.exp(normalized_vector) / np.sum(np.exp(normalized_vector))


def min_max_scale(x: float, _min: float, _max: float):
    """ https://stackoverflow.com/a/48178963/9246727 """
    x = np.minimum(x, _max)  # avoid the overflow in the specificity
    return (x - _min) / (_max - _min)


def _test_mean_dict():
    print("=== _test_mean_dict ===")
    _k = 11
    _list_dict = [dict(a=i * 1, b=i * 2, c=i * 3) for i in range(1, _k)]
    res = mean_dict(_list_dict=_list_dict)
    print(res)

    _list_dict = [{'hit_rate': 0.0, 'mrr': 0.0, 'ndcg': 0.0} for _ in range(10)]
    res = mean_dict(_list_dict=_list_dict)
    print(res)

    _list_dict = [{'hit_rate': 0.5, 'mrr': 0.5, 'ndcg': 0.5} for _ in range(1)]
    res = mean_dict(_list_dict=_list_dict)
    print(res)


def _test_ema():
    values = [1, 2, 3, 4, 5]
    print(ema(values, alpha=0.1))


if __name__ == '__main__':
    _test_logging()
    _test_mean_dict()
    _test_ema()
