import os
import torch
import numpy as np
from typing import Dict
from argparse import Namespace

from value_based.embedding.base import BaseEmbedding
from value_based.commons.seeds import set_randomSeed


def prep_input(dict_embedding: Dict[str, BaseEmbedding], args: Namespace, if_get_all: bool = False):
    if if_get_all:
        deep_feat = dict_embedding["deep"].get_all(if_np=False)
        wide_feat = dict_embedding["wide"].get_all(if_np=False)
    else:
        deep_feat = dict_embedding["deep"].sample(num_samples=args.batch_size, if_np=False)
        wide_feat = dict_embedding["wide"].sample(num_samples=args.batch_size, if_np=False)
    return {"in_deep": deep_feat, "in_wide": wide_feat}


def train(model, dict_embedding: Dict[str, BaseEmbedding], args: Namespace):
    print("=== Train Model ===")
    model.train()
    for epoch in range(args.num_epochs):
        _in = prep_input(dict_embedding=dict_embedding, args=args)
        out = model(_in)
        loss = model.loss_function(x=out)

        if ((epoch + 1) % 500) == 0: print(f"epoch: {epoch} loss: {loss}")

    return model


def main(args):
    # Set the random seed
    set_randomSeed(seed=args.random_seed)

    # make the directory if it doesn't exist yet
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    embedding = list()
    for name in ["offline", "online"]:
        deep_embedding = BaseEmbedding(num_embeddings=args.num_allItems,
                                       dim_embed=args.dim_deep,
                                       embed_type="pretrained",
                                       embed_path=os.path.join(args.data_dir, f"{name}_deep_attr.npy"),
                                       if_debug=args.if_debug,
                                       device=args.device)
        wide_embedding = BaseEmbedding(num_embeddings=args.num_allItems,
                                       dim_embed=args.dim_wide,
                                       embed_type="pretrained",
                                       embed_path=os.path.join(args.data_dir, f"{name}_wide_attr.npy"),
                                       if_debug=args.if_debug,
                                       device=args.device)
        dict_embedding = {"deep": deep_embedding, "wide": wide_embedding}

        if name == "offline":
            from value_based.embedding.wide_and_deep import WideAndDeepVAE
            model = WideAndDeepVAE(in_wide_dim=args.dim_wide, in_deep_dim=args.dim_deep)
            args.name = name
            model = train(model=model, dict_embedding=dict_embedding, args=args)

            # Get the item embedding
            _in = prep_input(dict_embedding=dict_embedding, args=args, if_get_all=True)
            _, _embedding = model(_in, return_embedding=True)
            print(f"Pretrained Embedding: {_embedding.shape}, Mean: {_embedding.mean()}, Std: {_embedding.std()}")
            embedding.append(_embedding)
        else:
            # Get the item embedding
            _in = prep_input(dict_embedding=dict_embedding, args=args, if_get_all=True)
            _, _embedding = model(_in, return_embedding=True)
            print(f"Pretrained Embedding: {_embedding.shape}, Mean: {_embedding.mean()}, Std: {_embedding.std()}")
            embedding.append(_embedding)
    embedding = np.vstack(embedding)
    print(f"Pretrained Embedding: {embedding.shape}, Mean: {embedding.mean()}, Std: {embedding.std()}")
    np.save(file=os.path.join(args.save_dir, f"vae_item_embedding"), arr=embedding)
    torch.save(model.state_dict(), os.path.join(args.save_dir, "vae_weight.pkl"))


if __name__ == '__main__':
    from value_based.commons.args import get_all_args, add_args

    # Get the hyper-params
    args = get_all_args()

    # =========== DEBUG =======================
    # args.device = "cpu"
    # args.num_epochs = 3
    # args.if_debug = False
    # args.env_name = "sample"
    # args.data_dir = "./data/sample/"
    # args.batch_size = 64
    # args.save_dir = "./trained_weight/reward_model/sample/"
    # args = add_args(args=args)
    # =========== DEBUG =======================

    print(args)

    main(args=args)
