import matplotlib

matplotlib.use("Agg")

import os
import copy
import numpy as np
import pandas as pd
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

from argparse import Namespace
from scipy.special import softmax
from sklearn.preprocessing import minmax_scale

from value_based.commons.utils import logging
from value_based.commons.plot_embedding import plot_embedding
from value_based.policy.agent import Test

ATTENTION_WEIGHT_MAGNIFY = 50
NUM_NODES_TO_VISUALISE = 1


def _visualise_state_embed_user_interests(user_interests: np.ndarray, state_embed: np.ndarray, args: dict):
    """
    Args:
        user_interests (np.ndarray): batch_step_size x dim_user(num_categories)
        state_embed (np.ndarray): batch_step_size x dim_state(num_categories)
    """
    logging("=== visualise_state_embed_user_interests ===")
    # visualise the user state and user_embed
    embedding = np.vstack([user_interests, state_embed])
    # logging(user_interests.shape, state_embed.shape, embedding.shape)

    # Get the labels
    label_dict = dict()
    label_dict["labels"] = np.zeros(embedding.shape[0]).astype(np.int)
    label_dict["labels"][:user_interests.shape[0]] = 0
    label_dict["labels"][user_interests.shape[0]:] = 1
    label_dict["desc"] = {"user_interests": 0, "state_embed": 1}

    plot_embedding(embedding=embedding,
                   label_dict=label_dict,
                   save_dir="./images/",
                   file_name=f"state_embed_Ep{args.get('epoch', 0)}",
                   num_neighbours=args["recsim_num_categories"])


def _visualise_gnn_embed_item_embedding(item_embed: np.ndarray, gnn_embed: np.ndarray, args: dict):
    """

    Args:
        item_embed (np.ndarray): num_candidates x dim_item(num_categories)
        gnn_embed (np.ndarray): batch_step_size x dim_node(num_categories) or
                                batch_step_size x num_candidates x dim_node(num_categories)
    """
    # logging("=== visualise_gnn_embed_item_embedding ===")
    # visualise the item embedding and gnn_embed
    embedding = np.vstack([item_embed, gnn_embed])
    # logging(item_embed.shape, gnn_embed.shape, embedding.shape)

    # Get the labels
    label_dict = dict()
    label_dict["labels"] = np.zeros(embedding.shape[0]).astype(np.int)
    label_dict["labels"][:item_embed.shape[0]] = 0
    label_dict["labels"][item_embed.shape[0]:] = 1
    label_dict["desc"] = {"item_embed": 0, "gnn_embed": 1}

    plot_embedding(embedding=embedding,
                   label_dict=label_dict,
                   save_dir="./images/",
                   file_name=f"gnn_embed_Ep{args.get('epoch', 0)}",
                   num_neighbours=args["recsim_num_categories"])


def _visualise_slate_embed_item_embedding(item_embed: np.ndarray, slate_embed: np.ndarray, args: dict):
    """
    Args:
        item_embed (np.ndarray): num_candidates x dim_item(num_categories)
        slate_embed (np.ndarray): batch_step_size x slate_size x dim_slate(num_categories)
    """
    # logging("=== visualise_slate_embed_item_embedding ===")
    # visualise the item embedding and gnn_embed

    for _user_id in range(slate_embed.shape[0]):
        embedding = np.vstack([item_embed, slate_embed[:, _user_id, :]])
        # logging(item_embed.shape, slate_embed.shape, embedding.shape)

        # Get the labels
        label_dict = dict()
        label_dict["labels"] = np.zeros(embedding.shape[0]).astype(np.int)
        label_dict["labels"][:item_embed.shape[0]] = 0
        label_dict["labels"][item_embed.shape[0]:] = 1
        label_dict["desc"] = {"item_embed": 0, "slate_embed": 1}

        plot_embedding(embedding=embedding,
                       label_dict=label_dict,
                       save_dir="./images/",
                       file_name=f"slate_embed_user{_user_id}_Ep{args.get('epoch', 0)}",
                       num_neighbours=args["recsim_num_categories"])
        return None


def _visualise_attention_weight(inputs: dict, args: dict):
    attention_map_image_list = list()
    # Pick the major main item category of the candidate set
    main_category = np.argmax(np.bincount(inputs["category"]))  # int
    category_mask = inputs["category"] == main_category  # (num_candidate)-size array
    num_candidates = len(category_mask)  # int

    # Select the user who has the highest variance in attention weights in the items falling in the major category
    _embedding = inputs["attention_weight"][..., category_mask, :]
    for _ in range(len(_embedding.shape) - 1):
        _embedding = np.var(_embedding, axis=-1)
    selected_user = np.argmax(_embedding)
    del _embedding

    # Extract from the batch of users
    embedding = inputs["attention_weight"][selected_user, ...]
    action = inputs["action_ind"][selected_user, :]  # action (np.ndarray): batch_size x slate_size

    # select the user based on the variance in attention weight
    if len(embedding.shape) == 2:  # num_nodes x num_nodes
        labels = np.asarray([f"sl{i}" if i in action else f"c{i}" for i in range(num_candidates)])
        _path_list = _visualise_networkx(labels=labels,
                                         embedding=embedding,
                                         category=inputs["category"],
                                         selected_node=action,
                                         save_dir="./images/",
                                         file_name=f"TS{args.get('ts', 0)}")
        attention_map_image_list += _path_list
    elif len(embedding.shape) == 3:  # intra_slate_step x num_nodes x num_nodes
        labels = np.asarray([
            f"sp{i}" if inputs["special_item_flg_vec"][i] else f"c{i}" for i in range(num_candidates)
        ])
        for t in range(action.shape[-1]):
            _labels = labels.copy()
            _labels = _labels.tolist()
            if _labels[action[t]].startswith("sp"):
                _labels[action[t]] = f"slp{action[t]}"
            else:
                _labels[action[t]] = f"sl{action[t]}"
            _labels = np.asarray(_labels)
            _embedding = embedding[t, ...].copy()  # num_nodes x num_nodes
            _category = inputs["category"].copy()  # (num_nodes)-sized array
            _category_mask = category_mask.copy()  # (num_nodes)-sized array

            if t > 0:
                # Remove the information about the previously selected items in the intermediate-slate
                _embedding = np.delete(_embedding, action[:t], axis=0)  # delete rows
                _embedding = np.delete(_embedding, action[:t], axis=1)  # delete cols
                _category = np.delete(_category, action[:t], axis=0)  # delete entries in the array
                _category_mask = np.delete(_category_mask, action[:t], axis=0)  # delete entries in the array
                _labels = np.delete(_labels, action[:t], axis=0)  # delete entries in the array

            try:
                selected_node = _labels.tolist().index(f"sl{action[t]}")
            except:
                selected_node = _labels.tolist().index(f"slp{action[t]}")

            # rgb_values = sns.color_palette("Set2", args["recsim_num_categories"])
            # rgb_values = sns.color_palette("Paired", args["recsim_num_categories"])
            rgb_values = sns.color_palette("husl", args["recsim_num_categories"])
            rgb_values = [rgb_values[__category] for __category in sorted(list(set(_category.tolist())))]
            _path_list = _visualise_networkx(labels=_labels,
                                             embedding=_embedding,
                                             category=_category,
                                             selected_node=selected_node,
                                             rgb_values=rgb_values,
                                             save_dir="./images/",
                                             file_name=f"eval_{args['train_test_flg']}_TS{args.get('ts', 0)}_t{t}")
            attention_map_image_list += _path_list
    return attention_map_image_list


def _visualise_networkx(labels: np.ndarray,
                        embedding: np.ndarray,
                        category: np.ndarray,
                        selected_node: int,
                        rgb_values: list,
                        save_dir: str,
                        file_name: str):
    """
    Args:
        embedding (np.ndarray): num_nodes x num_nodes (num_senders x num_receivers)
        category (np.ndarray): num_nodes-array
    """
    attention_map_image_list = list()

    # Create a dir if it doesn't exist
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # Prep the attention weights
    np.fill_diagonal(a=embedding, val=0.0)  # replace the diagonal elem(ie, self attention) with zero
    embedding *= ATTENTION_WEIGHT_MAGNIFY  # Magnify the edge values for visualisation purpose
    _embedding = embedding.copy()

    mask = np.eye(_embedding.shape[0])[selected_node].astype(np.bool)
    _embedding[~mask, :] = int(0)

    # Visualise the attention map
    df = pd.DataFrame(_embedding, index=labels, columns=labels)
    _path = __visualise(df=df,
                        category=category,
                        selected_column=df.columns[selected_node],
                        plot_labels=dict(zip(labels, labels)),
                        rgb_values=rgb_values,
                        file_name=f"{file_name}_{selected_node}",
                        title=f"{file_name}_{selected_node}",
                        save_dir=save_dir)
    attention_map_image_list.append(_path)
    return attention_map_image_list


def __visualise(df: pd.DataFrame, category, file_name, save_dir, rgb_values=None, mapping=None, selected_column=None,
                plot_labels=None, env='recsim', title=None):
    G = nx.from_pandas_adjacency(df)
    G = G.to_directed()
    pos = nx.circular_layout(G)

    if mapping is None:
        mapping = set(category)

    if rgb_values is None:
        rgb_values = sns.color_palette("Set2", len(mapping))
    color_map = dict(zip(mapping, rgb_values))
    node_color = [color_map[_category] for _category in category]

    # Get weights from graph
    if selected_column is not None:
        [G.remove_edge(s, e) for s, e in copy.deepcopy(G).edges() if e == selected_column]

    if env.lower() in ["create", "gw"]:
        weights = np.array([G[s][e]["weight"] for s, e in G.edges()])
        if weights is not None and len(weights) > 0:
            weights = 2 * weights / weights.max()
            [G.remove_edge(s, e) for i, (s, e) in enumerate(copy.deepcopy(G).edges()) if weights[i] < 0.0001]

        weights = np.array([G[s][e]["weight"] for s, e in G.edges()])
        if weights is not None and len(weights) > 0:
            weights = 2 * weights / weights.max()
    else:  # recsim / sample
        weights = np.array([G[s][e]["weight"] for s, e in G.edges()])
        if weights is not None and len(weights) > 0 and np.std(weights) > 0.0:
            weights = minmax_scale(X=weights.T).T
            weights *= 1.5
            # weights = 2 * weights / weights.max()
            [G.remove_edge(s, e) for i, (s, e) in enumerate(copy.deepcopy(G).edges()) if weights[i] < 0.0001]
            # weights = softmax(x=weights, axis=-1)
        else:
            # the original logic to normalsie the weights
            weights = np.array([G[s][e]["weight"] for s, e in G.edges()])
            if weights is not None and len(weights) > 0:
                weights = 2 * weights / weights.max()
                [G.remove_edge(s, e) for i, (s, e) in enumerate(copy.deepcopy(G).edges()) if weights[i] < 0.0001]

            weights = np.array([G[s][e]["weight"] for s, e in G.edges()])
            if weights is not None and len(weights) > 0:
                weights = 2 * weights / weights.max()

    options = {
        "node_color": node_color,
        "edge_color": "#7678ed",
        "width": weights,
        "edge_cmap": plt.cm.Blues,
        "arrowsize": 10,
        "arrowstyle": "-|>",
        "node_size": 500,
        "font_size": 10 if env == 'create' else 12,
        "node_shape": 'o',
        "labels": plot_labels,
        "font_weight": 'regular',
    }

    # make empty plot with correct color and label for each group
    for _category in mapping:
        plt.scatter([], [], color=color_map[_category], label=_category)

    nx.draw(G, pos, **options)
    G.clear()

    legend_title_dict = {
        'create': 'Activator Type',
        'recsim': 'Item Type',
        'sample': 'Item Type',
        'gw': 'Skill',
    }

    plt.legend(loc=(1.04, 0), title=legend_title_dict[env])
    if not env in ['create']:
        plt.title(title if title is not None else file_name.split("_")[-1])
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    plt.savefig(os.path.join(save_dir, "{}.png".format(file_name)), bbox_inches="tight")
    save_image = os.path.join(save_dir, "{}.png".format(file_name))
    # plt.savefig(os.path.join(save_dir, "{}.pdf".format(file_name)), bbox_inches="tight")
    # save_image = os.path.join(save_dir, "{}.pdf".format(file_name))
    plt.clf()
    return save_image


def visualise_agent(env, inputs: dict, args: dict):
    # logging("=== Check the input ===")
    # for k, v in inputs.items():
    #     if v is None:
    #         logging(f"{k}: skipped")
    #     else:
    #         logging(f"{k}: {v.shape}")
    #
    # # Get the item embedding
    # item_embed = env.item_embedding.get(index=env.items_dict["test"], if_np=True)
    #
    # # plot state_embed
    # user_states = env.get_state()
    # _visualise_state_embed_user_interests(user_interests=user_states["user_interests"],
    #                                       state_embed=inputs["state_embed"],
    #                                       args=args)
    # if inputs["gnn_embed"] is not None:
    #     # plot gnn_embed
    #     _visualise_gnn_embed_item_embedding(item_embed=item_embed, gnn_embed=inputs["gnn_embed"], args=args)
    #
    # if inputs["slate_embed"] is not None:
    #     # plot slate_embed
    #     _visualise_slate_embed_item_embedding(item_embed=item_embed, slate_embed=inputs["slate_embed"], args=args)

    attention_map_image_list = None
    if inputs["attention_weight"] is not None:
        # plot the attention vector
        attention_map_image_list = _visualise_attention_weight(inputs=inputs, args=args)
    return attention_map_image_list


def test(agent, env, args: Namespace):
    agent.set_if_visualise(flg=True)
    env.set_if_eval_train_or_test(train_or_test=args.train_test_flg)
    obs = env.reset()
    env.set_if_eval(flg=True)
    for ts in range(5):
        action = agent.select_action(obs=obs, candidate_list=env.items_dict["test"], epsilon=0.0)
        obs, reward, done, info = env.step(action)
        if ts == 4:
            args.ts = ts
            data = agent.get_out_info()
            data["category"] = env.get_mainCategory_of_items(arr_items=env.items_dict["test"])
            data["special_item_flg_vec"] = env.get_special_item_flg_vec(arr_items=env.items_dict["test"])
            visualise_agent(env=env, inputs=data, args=vars(args))


if __name__ == '__main__':
    from value_based.commons.launcher import launch_agent, launch_encoders, launch_embedding, launch_env
    from value_based.commons.args import add_args, get_all_args
    from value_based.commons.args_agents import params_dict, check_param
    from value_based.commons.seeds import set_randomSeed

    # Get the basic args
    args = get_all_args()
    args.slate_size = 3
    args.num_candidates = 20
    args.recsim_resampling_method = "skewed"
    args.recsim_if_special_items = True
    set_randomSeed(seed=2021)

    # main test part
    for agent_name in [
        # "CDQN",
        # "GCDQN",
        "AGILE",
        # "RCDQN",
        # "R3CDQN"
    ]:
        for _params_dict in params_dict[agent_name]:
            if not check_param(_params_dict=_params_dict): continue
            logging(f"\n=== params {_params_dict} ===")

            # Update the hyper-params with the test specific ones
            args = Test.update_args_from_dict(args=args, _dict=_params_dict)
            args = add_args(args=args)
            args.slate_encoder_type = "sequential-rnn"
            args.obs_encoder_type = "None"
            args.train_test_flg = "test"
            dict_embedding = launch_embedding(args=args)
            encoders_dict = launch_encoders(item_embedding=dict_embedding["item"], args=args)
            agent = launch_agent(dict_embedding=dict_embedding, encoders_dict=encoders_dict, args=args)
            env = launch_env(dict_embedding=dict_embedding, args=args)
            args.epoch = 0
            test(agent=agent, env=env, args=args)
            asdf
