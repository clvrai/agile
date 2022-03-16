""" Metrics class to manage all the metrics for experiments """
import numpy as np


def intrasession_diversity(_similarity, slate_size):
    _similarity = _similarity.sum()  # Aggregate the pairwise similarities
    _similarity -= slate_size  # Subtract the diagonal elements; cosine similarity to oneself is 1.0
    _similarity /= 2.0  # because it's the symmetric matrix
    _diversity = 1 - (2 / (slate_size * (slate_size - 1)) * _similarity)
    return _diversity


def shannon_entropy(p):
    _mask = p == 0.0
    p[_mask] = 10 ** (-10)  # Avoid the warning in np.log
    _entropy = -np.sum(p * np.log2(p))  # scalar
    return _entropy


def get_minmax(slate_size: int = 5, num_categories: int = 10):
    # Most similar items in slate
    p = np.zeros(num_categories)
    p[0] = 1.0  # items from the same single category
    min_entropy = shannon_entropy(p=p)  # Theoretically, it's the best entropy such that the slate has only one category
    similarity = np.ones((slate_size, slate_size))  # items have the same repre.
    min_diversity = intrasession_diversity(_similarity=similarity, slate_size=slate_size)

    # Most ACHIEVABLE dissimilar items in slate
    # NOTE: we only can fill the category within slate-zie
    p = np.array_split(ary=np.ones(slate_size), indices_or_sections=num_categories)
    p = np.asarray([len(i) for i in p]).astype(np.float32)
    p /= np.sum(p)
    max_entropy = shannon_entropy(p=p)
    similarity = (- np.ones((slate_size, slate_size))) + (2 * np.eye(slate_size))
    max_diversity = intrasession_diversity(_similarity=similarity, slate_size=slate_size)

    # Compute the minmax of specificity
    # We use the second worst case for the cap of the specificity
    p = np.zeros(num_categories)
    p[0] = slate_size - 1  # items from the same single category
    p[1] = 1  # items from the same single category
    p /= np.sum(p)
    _min_entropy = shannon_entropy(p=p)
    similarity = np.ones((slate_size, slate_size)) * 0.95  # items have the same repre.
    _min_diversity = intrasession_diversity(_similarity=similarity, slate_size=slate_size)

    # max_spec_ent = np.minimum(1 / _min_entropy, max_entropy) * 2  # Avoid the overflow
    # max_spec_div = np.minimum(1 / _min_diversity, max_diversity) * 2  # Avoid the overflow
    # min_spec_ent = np.minimum(1 / max_entropy, _min_entropy)  # Avoid the overflow
    # min_spec_div = np.minimum(1 / max_diversity, _min_diversity)  # Avoid the overflow

    max_spec_ent = 1.0 / _min_entropy
    max_spec_div = 1.0 / _min_diversity
    min_spec_ent = 1.0 / max_entropy
    min_spec_div = 1.0 / max_diversity
    print("=== CHECK MIN_MAX OF METRICS ===")
    print(f"[Min] Ent: {min_entropy}, Div: {min_diversity}, Spec_ent: {min_spec_ent}, Spec_div: {min_spec_div}")
    print(f"[Max] Ent: {max_entropy}, Div: {max_diversity}, Spec_ent: {max_spec_ent}, Spec_div: {max_spec_div}")

    return {
        "min_entropy": min_entropy, "min_diversity": min_diversity, "min_spec_ent": min_spec_ent,
        "min_spec_div": min_spec_div, "max_entropy": max_entropy, "max_diversity": max_diversity,
        "max_spec_ent": max_spec_ent, "max_spec_div": max_spec_div
    }


if __name__ == '__main__':
    slate_size = 5
    num_categories = 10

    get_minmax(slate_size=slate_size, num_categories=num_categories)
