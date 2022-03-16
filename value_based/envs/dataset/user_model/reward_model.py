import torch
import numpy as np
import torch.optim as optim
import torch.nn.functional as F
from copy import deepcopy

from value_based.commons.args import HISTORY_SIZE
from value_based.commons.pt_check_grad import check_grad
from value_based.encoder.obs_encoder import SequentialObsEncoder
from value_based.encoder.sequential_models import RNNFamily
from value_based.policy.architecture.networks import NeuralNet
from value_based.envs.dataset.user_model.dl_models.deepfm import DeepFM
from value_based.commons.test import TestCase as test
from gnn.Norm.norm import Norm


class BaseRewardModel(object):
    def __init__(self, args: dict):
        self._args = args
        self._device = self._args.get("device", "cpu")
        self._slate_size = self._args.get("slate_size", 10)
        self._batch_step_size = self._args.get("batch_step_size", 10)

        # Manually set the hyper-params since we didn't have time to retrain the reward models
        _args = deepcopy(args)
        _args["obs_encoder_dim_hidden"] = _args["obs_encoder_dim_out"] = _args["slate_encoder_dim_hidden"] = _args[
            "slate_encoder_dim_out"] = 32
        _args["obs_encoder_mlp_dim_hidden"] = _args["slate_encoder_mlp_dim_hidden"] = "256_32"
        self._args = _args

        self.obs_encoder = SequentialObsEncoder(args=_args)
        self.model = self.rnn = None
        self._check_grad = False
        self.training = True
        self._if_use_Rmodel = True if args.get("rm_model_type", "mlp") != "None" else False
        self._if_debug = self._args.get("if_debug", False)

    def compute_score(self, obs: np.ndarray, item_embedding: np.ndarray):
        """ Estimate rewards

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs
            item_embedding(np.ndarray): batch_size x slate_size x dim_item

        Returns:
            score(np.ndarray or None): batch_size x 1 or None
        """

        if self._if_use_Rmodel:
            # get the base score from the reward model
            if not torch.is_tensor(item_embedding):
                item_embedding = torch.tensor(item_embedding, dtype=torch.float32, device=self._device)
            score = self._compute_score(obs=obs, item_embedding=item_embedding)  # batch_size x 1
            score = score.cpu().detach().numpy()
        else:
            score = np.zeros((self._batch_step_size, self._slate_size))

        return score

    def _compute_score(self, obs: np.ndarray, item_embedding: torch.tensor):
        """ Estimate rewards

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs
            item_embedding(torch.tensor): batch_size x slate_size x dim_item

        Returns:
            if_click(np.ndarray): batch_size x 1
        """
        raise NotImplementedError

    def update(self, obs: np.ndarray, item_embedding: torch.tensor, y: np.ndarray):
        """ Inner method of update

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs
            item_embedding(torch.tensor): batch_size x slate_size x dim_item
            y(np.ndarray): (batch_size)-size array

        Returns:
            loss(float): loss value
        """
        return self._update(obs=obs, item_embedding=item_embedding, y=y)

    def _update(self, obs: np.ndarray, item_embedding: torch.tensor, y: np.ndarray):
        """ Inner method of update

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs
            item_embedding(torch.tensor): batch_size x slate_size x dim_item
            y(np.ndarray): (batch_size)-size array

        Returns:
            loss(float): loss value
        """
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def eval(self):
        raise NotImplementedError

    def set_if_check_grad(self, flg: bool):
        self._check_grad = flg

    def state_dict(self):
        raise NotImplementedError


def launch_core_model(args: dict):
    if args.get("rm_model_type", "mlp") == "mlp":
        # Input: [state_embed, slate_embed]
        # Shape: [obs_encoder_dim_out, (dim_item * 2)]
        # Output: logit of each item in a slate or logit of click on the entire slate
        return NeuralNet(dim_in=args["obs_encoder_dim_out"] + args["slate_encoder_dim_out"],
                         dim_hiddens=args["rm_dim_hidden"],  # by default; 256_128_64_32
                         dim_out=args["rm_dim_out"]).to(device=args["device"])
    elif args.get("rm_model_type", "mlp") == "deepfm":
        return DeepFM(num_feat=args["obs_encoder_dim_out"] + args["dim_item"],
                      num_field=args["obs_encoder_dim_out"] + args["dim_item"]).to(device=args["device"])
    else:
        return None


class RewardModel(BaseRewardModel):
    def __init__(self, args: dict):
        super(RewardModel, self).__init__(args=args)
        if self._if_use_Rmodel:
            self._dropout = 0.0
            self.model = launch_core_model(args=args)
            self.slate_encoder = RNNFamily(dim_in=self._args["slate_encoder_dim_in"],
                                           dim_hidden=self._args["slate_encoder_dim_hidden"],
                                           mlp_dim_hidden=self._args["slate_encoder_mlp_dim_hidden"],
                                           dim_out=self._args["slate_encoder_dim_out"],
                                           batch_first=True,
                                           dropout_rate=0.0,
                                           device=self._device).to(self._device)

            self.norm_state = Norm(norm_type=self._args["rm_norm_type"],
                                   hidden_dim=args["obs_encoder_dim_out"],
                                   if_gnn=False).to(self._device)
            self.norm_slate = Norm(norm_type=self._args["rm_norm_type"],
                                   hidden_dim=args["slate_encoder_dim_out"],
                                   if_gnn=False).to(self._device)

            # Instantiate the optim and set the models to it
            params = [
                *self.model.parameters(),
                *self.obs_encoder.encoder.parameters(),
                *self.slate_encoder.parameters(),
                *self.norm_state.parameters(),
                *self.norm_slate.parameters(),
            ]
            self.opt = optim.Adam(params, lr=self._args["rm_lr"], weight_decay=0.0)
            # self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=0.9)
            self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt,
                                                                     T_max=args.get("num_epochs", 0))

            # Load the pretrained state
            if self._args.get("rm_weight_path", None):
                print("RewardModel>> Loading from {}".format(self._args["rm_weight_path"]))
                _state = torch.load(self._args["rm_weight_path"], map_location=self._device)
                self.model.load_state_dict(_state["model"])
                self.obs_encoder.encoder.load_state_dict(_state["obs_encoder"])
                self.slate_encoder.load_state_dict(_state["slate_encoder"])
                self.norm_slate.load_state_dict(_state["norm_slate"])
                self.norm_state.load_state_dict(_state["norm_state"])
                # self._opt.load_state_dict(_state["opt"])

            # self._loss_fn = torch.nn.CrossEntropyLoss()
            # self._loss_fn = torch.nn.BCEWithLogitsLoss()
            self._loss_fn = torch.nn.BCELoss()

    def _compute_score(self, obs: np.ndarray, item_embedding: torch.tensor) -> torch.tensor:
        """ Estimate rewards

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs
            item_embedding(torch.tensor): batch_size x slate_size x dim_item

        Returns:
            scores(np.ndarray): batch_size x 1
        """
        # Get the slate embedding
        slate_embed = self.slate_encoder(item_embedding)  # batch_size x dim_slate
        slate_embed = F.dropout(slate_embed, self._dropout, training=self.training)

        # Get the state embedding
        state_embed = self.obs_encoder.encode(obs=obs)  # batch_size x dim_obs
        state_embed = F.dropout(state_embed, self._dropout, training=self.training)

        slate_embed = self.norm_slate(slate_embed)
        state_embed = self.norm_state(state_embed)

        # Compute the scores for items in a slate
        _input = torch.cat([state_embed, slate_embed], dim=-1)
        scores = self.model(_input)  # batch_size x 1
        scores = torch.sigmoid(scores)
        return scores

    def _update(self, obs: np.ndarray, item_embedding: torch.tensor, y: np.ndarray):
        """ Inner method of update

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs
            item_embedding(torch.tensor): batch_size x slate_size x dim_item
            y(np.ndarray): (batch_size)-size array

        Returns:
            loss(float): loss value
        """
        if self._if_debug:
            print(f"RewardModel>> obs: {obs.shape}, item_embedding: {item_embedding.shape}, y: {y.shape}")

        # Get the prediction
        pred = self._compute_score(obs=obs, item_embedding=item_embedding)  # batch_size x 1
        y = torch.tensor(y, device=self._device).to(pred.dtype)
        loss = self._loss_fn(pred.view(-1), y)
        self.opt.zero_grad()
        loss.backward()

        # check the gradients
        if self._check_grad:
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.model.named_parameters(), n_decimal=5)
            print("Reward model grad\n\tAve grad: {}\tMax grad: {}".format(ave_grad_dict, max_grad_dict))
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.rnn.named_parameters(), n_decimal=5)
            print("Slate RNN grad\n\tAve grad: {}\tMax grad: {}".format(ave_grad_dict, max_grad_dict))
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.obs_encoder.encoder.named_parameters(),
                                                      n_decimal=5)
            print("State RNN grad\n\tAve grad: {}\tMax grad: {}".format(ave_grad_dict, max_grad_dict))

        self.opt.step()
        self.lr_scheduler.step()
        return loss.item()

    def train(self):
        self.training = True
        if self._if_use_Rmodel:
            self.slate_encoder.train()
            self.obs_encoder.encoder.train()
            self.model.train()
            self.norm_state.train()
            self.norm_slate.train()

    def eval(self):
        self.training = False
        if self._if_use_Rmodel:
            self.slate_encoder.eval()
            self.obs_encoder.encoder.eval()
            self.model.eval()
            self.norm_state.eval()
            self.norm_slate.eval()

    def set_if_check_grad(self, flg: bool):
        self._check_grad = flg

    def state_dict(self):
        if self._if_use_Rmodel:
            return {
                "opt": self.opt.state_dict(),
                "model": self.model.state_dict(),
                "obs_encoder": self.obs_encoder.encoder.state_dict(),
                "slate_encoder": self.slate_encoder.state_dict(),
                "norm_slate": self.norm_slate.state_dict(),
                "norm_state": self.norm_state.state_dict(),
            }
        else:
            return {}


class RewardModelSimple(BaseRewardModel):
    def __init__(self, args: dict):
        super(RewardModelSimple, self).__init__(args=args)
        if self._if_use_Rmodel:
            self._dropout = 0.0
            self.slate_encoder = NeuralNet(dim_in=args["dim_item"] * 16,
                                           dim_hiddens=args["rm_dim_hidden"],
                                           dim_out=args["slate_encoder_dim_out"]).to(device=args["device"])
            self.obs_encoder = NeuralNet(dim_in=args["dim_user"] + (args["dim_item"] * HISTORY_SIZE),
                                         dim_hiddens=args["rm_dim_hidden"],
                                         dim_out=args["obs_encoder_dim_out"]).to(device=args["device"])
            self.model = NeuralNet(dim_in=args["obs_encoder_dim_out"] + args["slate_encoder_dim_out"],
                                   dim_hiddens=args["rm_dim_hidden"],
                                   dim_out=args["rm_dim_out"]).to(device=args["device"])

            # Instantiate the optim and set the models to it
            params = [
                *self.model.parameters(),
                *self.obs_encoder.parameters(),
                *self.slate_encoder.parameters(),
            ]
            self.opt = optim.Adam(params, lr=self._args["rm_lr"], weight_decay=0.0)
            # self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer=self.opt, gamma=0.9)
            # self.lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.opt,
            #                                                          T_max=args.get("num_epochs", 0))

            # Load the pretrained state
            if self._args.get("rm_weight_path", None):
                print("RewardModel>> Loading from {}".format(self._args["rm_weight_path"]))
                _state = torch.load(self._args["rm_weight_path"], map_location=self._device)
                self.model.load_state_dict(_state["model"])
                self.obs_encoder.encoder.load_state_dict(_state["obs_encoder"])
                self.slate_encoder.load_state_dict(_state["slate_encoder"])
                # self._opt.load_state_dict(_state["opt"])

            # self._loss_fn = torch.nn.CrossEntropyLoss()
            # self._loss_fn = torch.nn.BCEWithLogitsLoss()
            self._loss_fn = torch.nn.BCELoss()

    def _compute_score(self, obs: np.ndarray, item_embedding: torch.tensor) -> torch.tensor:
        """ Estimate rewards

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs
            item_embedding(torch.tensor): batch_size x slate_size x dim_item

        Returns:
            scores(np.ndarray): batch_size x 1
        """
        # Get the slate embedding
        bs, slate_size, dim = item_embedding.shape
        item_embedding = item_embedding.reshape(bs, slate_size * dim)
        slate_embed = self.slate_encoder(item_embedding)  # batch_size x dim_slate
        slate_embed = F.dropout(slate_embed, self._dropout, training=self.training)

        # Get the state embedding
        user_feat, user_history_feat = obs[0], obs[1]
        user_feat, user_history_feat = torch.tensor(user_feat, device=self._device), \
                                       torch.tensor(user_history_feat, device=self._device)
        bs, history_size, dim = user_history_feat.shape
        user_history_feat = user_history_feat.reshape(bs, history_size * dim)
        state_embed = self.obs_encoder(torch.cat([user_feat, user_history_feat], dim=-1))  # batch_size x dim_obs
        state_embed = F.dropout(state_embed, self._dropout, training=self.training)

        # Compute the scores for items in a slate
        _input = torch.cat([state_embed, slate_embed], dim=-1)
        scores = self.model(_input)  # batch_size x 1
        scores = torch.sigmoid(scores)
        return scores

    def _update(self, obs: np.ndarray, item_embedding: torch.tensor, y: np.ndarray):
        """ Inner method of update

        Args:
            obs(np.ndarray): batch_size x history_size x dim_obs
            item_embedding(torch.tensor): batch_size x slate_size x dim_item
            y(np.ndarray): (batch_size)-size array

        Returns:
            loss(float): loss value
        """
        if self._if_debug:
            print(f"RewardModel>> obs: {obs.shape}, item_embedding: {item_embedding.shape}, y: {y.shape}")

        # Get the prediction
        pred = self._compute_score(obs=obs, item_embedding=item_embedding)  # batch_size x 1
        y = torch.tensor(y, device=self._device).to(pred.dtype)
        loss = self._loss_fn(pred.view(-1), y)
        self.opt.zero_grad()
        loss.backward()

        # check the gradients
        if self._check_grad:
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.model.named_parameters(), n_decimal=5)
            print("Reward model grad\n\tAve grad: {}\tMax grad: {}".format(ave_grad_dict, max_grad_dict))
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.rnn.named_parameters(), n_decimal=5)
            print("Slate RNN grad\n\tAve grad: {}\tMax grad: {}".format(ave_grad_dict, max_grad_dict))
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.obs_encoder.encoder.named_parameters(),
                                                      n_decimal=5)
            print("State RNN grad\n\tAve grad: {}\tMax grad: {}".format(ave_grad_dict, max_grad_dict))

        self.opt.step()
        return loss.item()

    def train(self):
        self.training = True
        if self._if_use_Rmodel:
            self.slate_encoder.train()
            self.obs_encoder.train()
            self.model.train()

    def eval(self):
        self.training = False
        if self._if_use_Rmodel:
            self.slate_encoder.eval()
            self.obs_encoder.eval()
            self.model.eval()

    def set_if_check_grad(self, flg: bool):
        self._check_grad = flg

    def state_dict(self):
        if self._if_use_Rmodel:
            return {
                "opt": self.opt.state_dict(),
                "model": self.model.state_dict(),
                "obs_encoder": self.obs_encoder.state_dict(),
                "slate_encoder": self.slate_encoder.state_dict(),
            }
        else:
            return {}


def launch_RewardModel(args: dict):
    if args["rm_reward_model_type"] == 1:
        return RewardModel(args=args)
    elif args["rm_reward_model_type"] == 3:
        return RewardModelSimple(args=args)


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        self.args.env_name = "sample"
        self._prep()

    def test(self):
        for model_name in [
            "None",
            # "sample/offline.pkl",
            # "sample/online.pkl",
        ]:
            if model_name != "None":
                self.args.rm_weight_path = "../../../../trained_weight/reward_model/" + model_name

            self.reward_model = launch_RewardModel(args=vars(self.args))

            print("=== Model: {} ===".format(model_name))
            print("=== Test: compute_score ===")
            self._compute_score()

            print("=== Test: update ===")
            self._update()

    def _compute_score(self):
        item_embedding = self.item_embedding.get(index=self.actions)
        obses = self.obses.make_obs(dict_embedding=self.dict_embedding)
        base_scores = self.reward_model.compute_score(obs=obses, item_embedding=item_embedding)
        if base_scores is not None:
            print(self.reward_model.__class__.__name__, base_scores.shape)
            print(self.reward_model.__class__.__name__, base_scores.min(), base_scores.max(), base_scores.sum(),
                  base_scores.mean())
        else:
            print("No Reward Model was used")

    def _update(self):
        item_embedding = self.item_embedding.get(index=self.actions)
        obses = self.obses.make_obs(dict_embedding=self.dict_embedding)
        clicked_items = np.argmax(np.random.randn(self.actions.shape[0], self.args.slate_size), axis=-1)
        loss = self.reward_model.update(obs=obses, item_embedding=item_embedding, y=clicked_items)
        print("[Update] Loss: {}".format(loss))


if __name__ == '__main__':
    Test().test()
