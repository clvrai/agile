import os
import torch
import numpy as np
import pandas as pd
from typing import Dict
from argparse import Namespace
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix

from value_based.commons.args import HISTORY_SIZE
from value_based.commons.launcher import launch_embedding
from value_based.embedding.base import BaseEmbedding
from value_based.envs.dataset.user_model.reward_model import launch_RewardModel, BaseRewardModel
from value_based.envs.dataset.user_model.reward_model_WandD import launch_RewardModel as launch_RewardModel2
from value_based.commons.seeds import set_randomSeed
from value_based.commons.utils import logging


def compute_metrics(pred: np.ndarray, y_true: np.ndarray):
    # y_pred = np.argmax(pred, axis=-1)  # for multi-class classification!
    y_pred = pred.astype(np.float32).ravel()
    y_pred = (y_pred > 0.5).astype(np.float32)
    correct_results_sum = float(sum(y_pred == y_true))
    acc = correct_results_sum / y_true.shape[0]
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average='macro')
    prec = precision_score(y_true=y_true, y_pred=y_pred, average='macro')
    recall = recall_score(y_true=y_true, y_pred=y_pred, average='macro')
    metrics = {"accuracy": acc, "f1": f1, "precision": prec, "recall": recall}
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    return metrics, conf_mat


def prep_input(data: dict, dict_embedding: Dict[str, BaseEmbedding], _name: str, args: Namespace):
    if args.rm_reward_model_type == 1:
        if _name == "all":
            item_embedding = dict_embedding["item"].get_all(if_np=False)
            return item_embedding
        user_history_feat = dict_embedding["item"].get(index=data[f"{_name}_history_seq"][args._idx], if_np=True)
        obs = np.concatenate([data[f"{_name}_user_feat"][args._idx], user_history_feat], axis=-1)
        item_embedding = dict_embedding["item"].get(index=data[f"{_name}_slate"][args._idx], if_np=False)
        y = data[f"{_name}_label"][args._idx]
        return obs, item_embedding, y
    elif args.rm_reward_model_type == 2:
        if _name == "all":
            deep_feat = dict_embedding["deep"].get_all(if_np=False)
            wide_feat = dict_embedding["wide"].get_all(if_np=False)
            item_embedding = {"in_deep": deep_feat, "in_wide": wide_feat}
            return item_embedding
        else:
            deep_feat = dict_embedding["deep"].get(index=data[f"{_name}_history_seq"][args._idx], if_np=False)
            wide_feat = dict_embedding["wide"].get(index=data[f"{_name}_history_seq"][args._idx], if_np=False)
            user_history_feat = {"in_deep": deep_feat, "in_wide": wide_feat}
            deep_feat = dict_embedding["deep"].get(index=data[f"{_name}_slate"][args._idx], if_np=False)
            wide_feat = dict_embedding["wide"].get(index=data[f"{_name}_slate"][args._idx], if_np=False)
            item_embedding = {"in_deep": deep_feat, "in_wide": wide_feat}
            user_feat = torch.tensor(data[f"{_name}_user_feat"][args._idx], device=args.device)
            y = data[f"{_name}_label"][args._idx]
            return user_feat, user_history_feat, item_embedding, y
    elif args.rm_reward_model_type == 3:
        user_history_feat = dict_embedding["item"].get(index=data[f"{_name}_history_seq"][args._idx], if_np=True)
        user_feat = data[f"{_name}_user_feat"][args._idx]
        item_embedding = dict_embedding["item"].get(index=data[f"{_name}_slate"][args._idx], if_np=False)
        y = data[f"{_name}_label"][args._idx]
        return [user_feat, user_history_feat], item_embedding, y


def train(model: BaseRewardModel, data: dict, dict_embedding: Dict[str, BaseEmbedding], args: Namespace):
    logging("=== Train Model ===")
    model.train()
    train_idx = np.array_split(ary=np.arange(data["train_history_seq"].shape[0]),
                               indices_or_sections=data["train_history_seq"].shape[0] // args.batch_size)
    test_idx = np.array_split(ary=np.arange(data["test_history_seq"].shape[0]),
                              indices_or_sections=data["test_history_seq"].shape[0] // args.batch_size)
    for epoch in range(args.num_epochs):
        # === One Training Epoch ===
        loss = list()
        for _idx in train_idx:
            args._idx = _idx
            inputs = prep_input(data=data, dict_embedding=dict_embedding, _name="train", args=args)
            _loss = model.update(*inputs)
            loss.append(_loss)

        # === After one epoch ===
        logging(f"[Train] epoch: {epoch} loss: {np.mean(loss):.3f}")

        # === Evaluation ===
        model.eval()
        with torch.no_grad():
            # Sample from the data
            train_metrics_list = list()
            test_metrics_list = list()
            from value_based.commons.utils import mean_dict
            for flg in ["train", "test"]:
                for _idx in eval(f"{flg}_idx"):
                    args._idx = _idx
                    inputs = prep_input(data=data, dict_embedding=dict_embedding, _name=flg, args=args)
                    pred = model.compute_score(*inputs[:-1])
                    _metrics, _conf_mat = compute_metrics(pred=pred, y_true=inputs[-1])
                    if flg == "train":
                        conf_mat = _conf_mat
                        train_metrics_list.append(_metrics)
                    else:
                        conf_mat += _conf_mat
                        test_metrics_list.append(_metrics)

        logging("=== CONFUSION MATRIX ===")
        print(conf_mat)

        train_metrics = mean_dict(_list_dict=train_metrics_list)
        test_metrics = mean_dict(_list_dict=test_metrics_list)
        logging(f"[Eval: Train] epoch: {epoch} | {train_metrics}")
        logging(f"[Eval: Test] epoch: {epoch} | {test_metrics}")
        model.train()

        """ === Save Model === """
        if (epoch > 0) and (epoch % int(args.num_epochs * 0.2)) == 0:
            state = model.state_dict()
            state["epoch"] = epoch
            rm_weight_path = f"{args.rm_weight_path}_ep{epoch}.pkl"
            torch.save(state, rm_weight_path)
            logging("Model is saved in {}".format(rm_weight_path))

    if (args.rm_reward_model_type == 2) and args.if_offline:
        # Get the item embedding
        item_embedding = prep_input(data=data, dict_embedding=dict_embedding, _name="all", args=args)
        embedding = model.get_item_embedding(item_embedding=item_embedding)
        np.save(file=os.path.join(args.save_dir, "item_embedding"), arr=embedding)
        logging(f"Pretrained Embedding: {embedding.shape}, Mean: {embedding.mean()}, Std: {embedding.std()}")


def get_data(df_log: pd.DataFrame, dict_embedding: Dict[str, BaseEmbedding]):
    # Get label: index of clicked item in a slate
    history_seq, label, slate = list(), list(), list()
    for row in df_log.iterrows():
        _slate = eval(row[1]["slate"])
        _history_seq = eval(row[1]["hist_seq"])
        _clickedItem = int(row[1]["item_id"])
        # label.append(_slate.index(_clickedItem) if _clickedItem in _slate else -1)  # multiclass classification task!
        # label.append(1 if _clickedItem in _slate else 0)
        label.append(float(row[1]["click_in_slate"]))
        slate.append(_slate)
        history_seq.append(_history_seq)
    label, slate, history_seq = np.asarray(label), np.asarray(slate), np.asarray(history_seq)

    # get user attributes
    user_id = torch.tensor(df_log["user_id"].values, dtype=torch.int64, device=args.device)
    user_feat = dict_embedding["user"].get(index=user_id, if_np=True)

    if args.rm_reward_model_type == 1:
        user_feat = np.tile(A=user_feat[:, None, :], reps=(1, HISTORY_SIZE, 1))

    logging(f"get_data>> Rate of click event: {sum(label) / len(label)}")
    logging(f"get_data>> user_feat: {user_feat.shape} user_history_feat: {history_seq.shape}, "
            f"slate: {slate.shape}, label: {label.shape}")
    return user_feat, history_seq, slate, label


def even_negative_positive_event(df_log):
    return df_log
    # NOTE: we decided to use the original dataset
    # negatives = df_log[df_log["click_in_slate"] == False]
    # # positives = df_log[df_log["click_in_slate"] == True].sample(int(0.7 * negatives.shape[0]))
    # positives = df_log[df_log["click_in_slate"] == True].sample(int(negatives.shape[0]))
    # logging(f"negative event: {negatives.shape[0]}, positive event: {positives.shape[0]}")
    # df_log = pd.concat([negatives, positives], ignore_index=True)
    # df_log = df_log.sample(frac=1).reset_index(drop=True)  # shuffle the rows
    # return df_log


def main(args):
    # Set the random seed
    set_randomSeed(seed=args.random_seed)

    # make the directory if it doesn't exist yet
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    df_log = pd.read_csv(os.path.join(args.data_dir, f"{args.rm_offline_or_online}_log_data.csv"))

    # Even the portion of positive / negative events
    df_log = even_negative_positive_event(df_log=df_log)
    df_train, df_test = train_test_split(df_log, random_state=args.random_seed)

    if args.rm_reward_model_type == 2:
        deep_embedding = BaseEmbedding(num_embeddings=args.num_allItems,
                                       dim_embed=args.dim_deep,
                                       embed_type="pretrained",
                                       embed_path=args.rm_deep_embedding_path,
                                       if_debug=args.if_debug,
                                       device=args.device)
        wide_embedding = BaseEmbedding(num_embeddings=args.num_allItems,
                                       dim_embed=args.dim_wide,
                                       embed_type="pretrained",
                                       embed_path=args.rm_wide_embedding_path,
                                       if_debug=args.if_debug,
                                       device=args.device)
        user_embedding = BaseEmbedding(num_embeddings=args.num_allUsers,
                                       dim_embed=args.dim_user,
                                       embed_type="pretrained",
                                       embed_path=args.user_embedding_path,
                                       if_debug=args.if_debug,
                                       device=args.device)
        dict_embedding = {"deep": deep_embedding, "wide": wide_embedding, "user": user_embedding}
    else:
        dict_embedding = launch_embedding(args=args)

    train_user_feat, train_history_seq, train_slate, train_label = get_data(df_log=df_train,
                                                                            dict_embedding=dict_embedding)
    test_user_feat, test_history_seq, test_slate, test_label = get_data(df_log=df_test,
                                                                        dict_embedding=dict_embedding)
    data = {
        "train_user_feat": train_user_feat, "train_history_seq": train_history_seq, "train_label": train_label,
        "train_slate": train_slate,
        "test_user_feat": test_user_feat, "test_history_seq": test_history_seq, "test_label": test_label,
        "test_slate": test_slate,
    }

    args.rm_weight_path = ""
    if args.rm_reward_model_type == 2:
        model = launch_RewardModel2(args=vars(args))
    else:
        model = launch_RewardModel(args=vars(args))
    args.rm_weight_path = os.path.join(args.save_dir, args.rm_offline_or_online)
    args.if_offline = args.rm_offline_or_online == "offline"
    train(model=model, data=data, dict_embedding=dict_embedding, args=args)


if __name__ == '__main__':
    from value_based.commons.args import get_all_args, add_args

    # Get the hyper-params
    args = get_all_args()

    # =========== DEBUG =======================
    # ========= Reward Model 1: This is the main one!!
    # args.rm_reward_model_type = 1
    # args.num_epochs = 5
    # args.eval_freq = 25
    # args.rm_dim_out = 1
    # args.if_debug = False
    # args.env_name = "sample"
    # args.item_embedding_type = args.user_embedding_type = "pretrained"
    # args.rm_model_type = "mlp"
    # args.rm_if_train_simulator = True
    # args.data_dir = "./data/sample/"
    # args.item_embedding_path = "./trained_weight/reward_model/sample/vae_item_embedding.npy"
    # args.user_embedding_path = f"{args.data_dir}/user_attr.npy"
    # args.batch_size = 32
    # args.save_dir = "./trained_weight/reward_model/sample/"
    # # args.rm_offline_or_online = "offline"
    # args.rm_offline_or_online = "online"
    # args = add_args(args=args)

    # ========= Reward Model 2: Deprecated!!
    # args.rm_reward_model_type = True
    # args.device = "cpu"
    # args.num_epochs = 30
    # args.eval_freq = 25
    # args.rm_dim_out = 1
    # args.if_debug = False
    # args.env_name = "sample"
    # args.item_embedding_type = args.user_embedding_type = "pretrained"
    # args.rm_model_type = "mlp"
    # args.data_dir = "./data/sample/"
    # args.rm_deep_embedding_path = f"{args.data_dir}/deep_attr.npy"
    # args.rm_wide_embedding_path = f"{args.data_dir}/wide_attr.npy"
    # args.user_embedding_path = f"{args.data_dir}/user_attr.npy"
    # args.batch_size = 64
    # args.save_dir = "./trained_weight/reward_model/sample/"
    # args = add_args(args=args)

    # ========= Reward Model 3: Deprecated!!
    # Very simple logistic regression so that don't use it for the experiment!!
    # args.rm_reward_model_type = 3
    # args.rm_data_slate_size = 15
    # args.num_epochs = 101
    # args.eval_freq = 25
    # args.rm_dim_out = 1
    # args.if_debug = False
    # args.env_name = "sample"
    # args.item_embedding_type = args.user_embedding_type = "pretrained"
    # args.rm_model_type = "mlp"
    # args.rm_if_train_simulator = True
    # args.data_dir = "./data/sample/"
    # args.item_embedding_path = "./trained_weight/reward_model/sample/vae_item_embedding.npy"
    # args.user_embedding_path = f"{args.data_dir}/user_attr.npy"
    # args.batch_size = 32
    # args.save_dir = "./trained_weight/reward_model/sample/"
    # args = add_args(args=args)
    # =========== DEBUG =======================

    args.item_embedding_type = args.user_embedding_type = "pretrained"
    logging(args)

    main(args=args)
