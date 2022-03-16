import numpy as np

from value_based.envs.recsim.distribution import Distribution
from value_based.commons.plot_embedding import plot_embedding

if __name__ == '__main__':
    # hyper-params
    args = {
        "recsim_num_categories": 10,
        "recsim_num_subcategories": 2,
        "recsim_rejectionSampling_distance": 0.0,
        "recsim_itemFeat_samplingMethod": "rejection_sampling",
        "recsim_itemFeat_samplingDist": "GMM",
        "num_allItems": 500,
        "num_trainItems": 300,
        "num_testItems": 200,
        "if_debug": False,
        "recsim_if_use_subcategory": True
    }
    list_trainItemIds = list(range(args["num_trainItems"]))
    list_testItemIds = list(range(args["num_trainItems"], args["num_testItems"] + args["num_trainItems"]))
    dist = Distribution(args=args)
    dist.generate_item_representations(list_trainItemIds=list_trainItemIds, list_testItemIds=list_testItemIds)

    # Get the labels
    label_dict = {"labels": []}
    embedding = list()
    # for centroid in range(args["recsim_num_categories"]):
    for centroid in sorted(list(dist.category_master_dict["train"].keys())):
        result = dist._sample_from_GMM(centroid=centroid, num_samples=args["num_trainItems"], cov_coeff=0.1)
        label_dict["labels"] += [centroid] * result.shape[0]
        embedding.append(result)
        print(f"centroid: {centroid}, mean of sampled features: {result.mean(axis=0)}")

    label_dict["desc"] = {i: i for i in set(label_dict["labels"])}
    label_dict["labels"] = np.asarray(label_dict["labels"])
    embedding = np.asarray(embedding).reshape(
        (args["recsim_num_categories"] * args["recsim_num_subcategories"] * args["num_trainItems"],
         args["recsim_num_categories"])
    )

    plot_embedding(embedding=embedding,
                   label_dict=label_dict,
                   save_dir="../images/test",
                   file_name=f"GMM_subcategory",
                   num_neighbours=args["recsim_num_categories"])
