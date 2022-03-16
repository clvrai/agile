from argparse import Namespace
from typing import Dict

from value_based.embedding.base import BaseEmbedding
from value_based.policy.agent import RandomAgent
from value_based.policy.dqn import DQN
from value_based.policy.gcdqn import GCDQN
from value_based.encoder.obs_encoder import BasicObsEncoder, SequentialObsEncoder
from value_based.encoder.slate_encoder import BasicSlateEncoder, SequentialSlateEncoder
from value_based.encoder.act_encoder import BasicActEncoder, DenseActEncoder
from value_based.envs.recsim.environments import interest_evolution
from value_based.envs.dataset.env import launch_env as launch_dataset_env


def launch_embedding(args: Namespace, no_cuda=False):
    # === Launch item_embedding
    item_embedding = BaseEmbedding(num_embeddings=args.num_allItems,
                                   dim_embed=args.dim_item,
                                   embed_type=args.item_embedding_type,
                                   embed_path=args.item_embedding_path,
                                   if_debug=args.if_debug,
                                   device='cpu' if no_cuda else args.device)

    # === Launch user_embedding
    if hasattr(args, "num_allUsers"):
        user_embedding = BaseEmbedding(num_embeddings=args.num_allUsers,
                                       dim_embed=args.dim_user,
                                       embed_type=args.user_embedding_type,
                                       embed_path=args.user_embedding_path,
                                       if_debug=args.if_debug,
                                       device='cpu' if no_cuda else args.device)
    else:
        user_embedding = None

    dict_embedding = {"item": item_embedding, "user": user_embedding}
    return dict_embedding


def launch_env(dict_embedding: Dict[str, BaseEmbedding], args: Namespace):
    if args.env_name.lower() == "recsim" or args.env_name.startswith('RecSim'):
        env = interest_evolution.create_multiuser_environment(args=vars(args))
        if args.if_debug:
            print("launch_env>> Will Load Embedding: {}".format(env.item_embedding.shape))
        dict_embedding["item"].load(embedding=env.item_embedding)
    elif args.env_name.lower() in ["ml-100k", "sample"]:
        env = launch_dataset_env(dict_embedding=dict_embedding, args=vars(args))
    else:
        raise ValueError
    return env


def launch_encoders(item_embedding: BaseEmbedding, args: Namespace):
    # Obs encoder
    if args.obs_encoder_type.lower() == "basic":
        obs_encoder = BasicObsEncoder(args=vars(args))
    elif args.obs_encoder_type.lower().startswith("sequential"):
        obs_encoder = SequentialObsEncoder(args=vars(args))
    else:
        obs_encoder = None

    # Slate encoder
    if args.slate_encoder_type.lower() == "basic":
        slate_encoder = BasicSlateEncoder(item_embedding=item_embedding, args=vars(args))
    elif args.slate_encoder_type.lower().startswith("sequential"):
        slate_encoder = SequentialSlateEncoder(item_embedding=item_embedding, args=vars(args))
    else:
        slate_encoder = None

    # Action encoder
    if args.act_encoder_type.lower() == "basic":
        act_encoder = BasicActEncoder(args=vars(args))
    elif args.act_encoder_type.lower() == "dense":
        act_encoder = DenseActEncoder(args=vars(args))
    else:
        act_encoder = None
    dict_encoder = {"obs_encoder": obs_encoder, "slate_encoder": slate_encoder, "act_encoder": act_encoder}
    return dict_encoder


def launch_agent(dict_embedding: Dict[str, BaseEmbedding], encoders_dict: dict, args: Namespace):
    if args.agent_type.lower() == "gcdqn":
        agent = GCDQN(obs_encoder=encoders_dict["obs_encoder"],
                      dict_embedding=dict_embedding,
                      act_encoder=encoders_dict["act_encoder"],
                      slate_encoder=encoders_dict["slate_encoder"], args=vars(args))
    elif args.agent_type.lower() == "dqn":
        agent = DQN(obs_encoder=encoders_dict["obs_encoder"],
                    slate_encoder=encoders_dict["slate_encoder"],
                    act_encoder=encoders_dict["act_encoder"] if args.act_encoder_type != "None" else None,
                    dict_embedding=dict_embedding,
                    args=vars(args))
    elif args.agent_type.lower() == "random":
        agent = RandomAgent(obs_encoder=encoders_dict["obs_encoder"],
                            slate_encoder=encoders_dict["slate_encoder"], args=vars(args))
    else:
        raise ValueError
    return agent
