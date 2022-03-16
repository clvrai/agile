import torch
import torch.optim as optim
import numpy as np

from value_based.commons.init_layer import init_layer
from value_based.commons.args import DIM_ITEM_EMBED
from value_based.embedding.wide_and_deep import WideAndDeep
from value_based.envs.dataset.user_model.reward_model import BaseRewardModel
from value_based.commons.pt_check_grad import check_grad
from value_based.encoder.sequential_models import RNNFamily
from value_based.commons.test import TestCase as test


class RewardModel(BaseRewardModel):
    def __init__(self, args: dict):
        super(RewardModel, self).__init__(args=args)
        if self._if_use_Rmodel:
            self.WandD = WideAndDeep(in_wide_dim=self._args["dim_wide"],
                                     in_deep_dim=self._args["dim_deep"],
                                     dim_out=DIM_ITEM_EMBED).to(self._device)
            self.user_rnn = RNNFamily(dim_in=DIM_ITEM_EMBED,
                                      dim_hidden=self._args["slate_encoder_dim_hidden"],
                                      mlp_dim_hidden=self._args["slate_encoder_mlp_dim_hidden"],
                                      dim_out=DIM_ITEM_EMBED // 2,
                                      batch_first=True,
                                      dropout_rate=0.0,
                                      device=self._device).to(self._device)
            self.slate_rnn = RNNFamily(dim_in=DIM_ITEM_EMBED,
                                       dim_hidden=self._args["slate_encoder_dim_hidden"],
                                       mlp_dim_hidden=self._args["slate_encoder_mlp_dim_hidden"],
                                       dim_out=DIM_ITEM_EMBED // 2,
                                       batch_first=True,
                                       dropout_rate=0.0,
                                       device=self._device).to(self._device)
            self.user_feat_mlp = torch.nn.Sequential(
                torch.nn.Linear(self._args["dim_user"], DIM_ITEM_EMBED // 2).apply(init_layer)
            ).to(self._device)
            self.out_mlp = torch.nn.Sequential(
                torch.nn.Linear((DIM_ITEM_EMBED // 2) * 3, 16).apply(init_layer),
                torch.nn.Linear(16, 1).apply(init_layer)
            ).to(self._device)

            # self.norm_history_feat = Norm(norm_type="ln", hidden_dim=DIM_ITEM_EMBED // 2, if_gnn=False).to(self._device)
            # self.norm_user = Norm(norm_type="ln", hidden_dim=DIM_ITEM_EMBED // 2, if_gnn=False).to(self._device)
            # self.norm_slate = Norm(norm_type="ln", hidden_dim=DIM_ITEM_EMBED // 2, if_gnn=False).to(self._device)

            # Instantiate the optimiser and set the models to it
            params = [
                *self.WandD.parameters(),
                *self.user_rnn.parameters(),
                *self.slate_rnn.parameters(),
                *self.user_feat_mlp.parameters(),
                *self.out_mlp.parameters(),
                # *self.norm_history_feat.parameters(),
                # *self.norm_user.parameters(),
                # *self.norm_slate.parameters(),
            ]
            self.opt = optim.Adam(params, lr=1e-4)

            # Load the pretrained state
            if self._args.get("rm_weight_path", None):
                print("RewardModel>> Loading from {}".format(self._args["rm_weight_path"]))
                _state = torch.load(self._args["rm_weight_path"], map_location=self._device)
                self.WandD.load_state_dict(_state["WandD"])
                self.user_rnn.load_state_dict(_state["user_rnn"])
                self.slate_rnn.load_state_dict(_state["slate_rnn"])
                self.user_feat_mlp.load_state_dict(_state["user_feat_mlp"])
                self.out_mlp.load_state_dict(_state["out_mlp"])
                # self._opt.load_state_dict(_state["opt"])

            # self._loss_fn = torch.nn.CrossEntropyLoss()
            # self._loss_fn = torch.nn.BCEWithLogitsLoss()
            self._loss_fn = torch.nn.BCELoss()

    def _compute_score(self,
                       user_feat: torch.tensor,
                       user_history_feat: torch.tensor or dict,
                       item_embedding: torch.tensor or dict):
        # When we pre-train the item-embedding
        if type(user_history_feat) == dict:
            assert type(item_embedding) == dict

            # Get the compact item representation
            # batch_size x history_size x dim_item -> batch_size x history_size x dim_item_embed
            user_history_feat = self.WandD(user_history_feat)

            # Get the user_embed
            user_embed = self.user_feat_mlp(user_feat)

            # Get the compact item representation
            # batch_size x slate_size x dim_item -> batch_size x slate_size x dim_item_embed
            item_embedding = self.WandD(item_embedding)
        else:
            user_embed = user_feat

        # Get the slate embedding: batch_size x slate_size x dim_item_embed -> batch_size x dim_slate
        slate_embed = self.slate_rnn(item_embedding)  # batch_size x dim_slate

        # Compress the user history sequence: batch_size x history_size x dim_item_embed -> batch_size x dim_slate
        user_history_feat = self.user_rnn(user_history_feat)  # batch_size x dim_slate

        # Apply the normalisation;
        # user_history_feat, user_embed, slate_embed =\
        #     self.norm_history_feat(user_history_feat), self.norm_user(user_embed), self.norm_slate(slate_embed)

        # Concat both features to make an input
        _input = torch.cat([user_history_feat, user_embed, slate_embed], dim=-1)

        # Get the final score of a slate given the user
        scores = self.out_mlp(_input)  # batch_size x 1
        scores = torch.sigmoid(scores)
        return scores

    def _update(self,
                user_feat: torch.tensor,
                user_history_feat: dict,
                item_embedding: torch.tensor,
                y: torch.tensor or dict):
        # Get the prediction
        pred = self._compute_score(user_feat=user_feat,
                                   user_history_feat=user_history_feat,
                                   item_embedding=item_embedding)  # batch_size x 1
        y = torch.tensor(y, device=self._device).to(pred.dtype)
        loss = self._loss_fn(pred.view(-1), y)
        self.opt.zero_grad()
        loss.backward()

        # check the gradients
        if self._check_grad:
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.model.named_parameters(), n_decimal=5)
            print("Reward model grad\n\tAve grad: {}\tMax grad: {}".format(ave_grad_dict, max_grad_dict))
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.user_rnn.named_parameters(), n_decimal=5)
            print("Slate RNN grad\n\tAve grad: {}\tMax grad: {}".format(ave_grad_dict, max_grad_dict))
            ave_grad_dict, max_grad_dict = check_grad(named_parameters=self.obs_encoder.encoder.named_parameters(),
                                                      n_decimal=5)
            print("State RNN grad\n\tAve grad: {}\tMax grad: {}".format(ave_grad_dict, max_grad_dict))

        self.opt.step()
        return loss.item()

    def get_item_embedding(self, item_embedding: dict):
        """ get the pretrained item embedding

        Args:
            item_embedding (dict): dict of wide and deep features of all items

        Returns:
            embed (np.ndarray): np array of pretrained embedding
        """
        _, embed = self.WandD(item_embedding, return_embedding=True)
        if torch.is_tensor(embed):
            return embed.detach().cpu().numpy()
        else:
            return embed

    def train(self):
        if self._if_use_Rmodel:
            self.WandD.train()
            self.slate_rnn.train()
            self.user_rnn.train()
            self.user_feat_mlp.train()
            self.out_mlp.train()

    def eval(self):
        if self._if_use_Rmodel:
            self.WandD.eval()
            self.slate_rnn.eval()
            self.user_rnn.eval()
            self.user_feat_mlp.eval()
            self.out_mlp.eval()

    def set_if_check_grad(self, flg: bool):
        self._check_grad = flg

    def state_dict(self):
        if self._if_use_Rmodel:
            return {
                "opt": self.opt.state_dict(),
                "WandD": self.WandD.state_dict(),
                "user_rnn": self.user_rnn.state_dict(),
                "slate_rnn": self.slate_rnn.state_dict(),
                "user_feat_mlp": self.user_feat_mlp.state_dict(),
                "out_mlp": self.out_mlp.state_dict(),
            }
        else:
            return {}


def launch_RewardModel(args: dict):
    return RewardModel(args=args)


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        self.args.env_name = "sample"
        self.args.rm_model_type = "something"
        self._prep()

    def test(self):
        for model_name in [
            "None",
            # "sample/mlp.pkl",
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
        from value_based.commons.args import HISTORY_SIZE
        # === Pretraining
        user_feat = torch.randn(self.args.batch_size, self.args.dim_user)
        user_history_feat = {
            "in_deep": torch.randn(self.args.batch_size, HISTORY_SIZE, self.args.dim_deep),
            "in_wide": torch.randn(self.args.batch_size, HISTORY_SIZE, self.args.dim_wide)
        }
        item_embedding = {
            "in_deep": torch.randn(self.args.batch_size, self.args.slate_size, self.args.dim_deep),
            "in_wide": torch.randn(self.args.batch_size, self.args.slate_size, self.args.dim_wide)
        }
        base_scores = self.reward_model.compute_score(user_feat=user_feat,
                                                      user_history_feat=user_history_feat,
                                                      item_embedding=item_embedding)
        print(base_scores.shape)

        # === RL reward model
        item_embedding = torch.randn(self.args.batch_size, self.args.slate_size, DIM_ITEM_EMBED)
        user_feat = torch.randn(self.args.batch_size, DIM_ITEM_EMBED // 2)
        history_seq = torch.randn(self.args.batch_size, HISTORY_SIZE, DIM_ITEM_EMBED)
        base_scores = self.reward_model.compute_score(user_feat=user_feat,
                                                      user_history_feat=history_seq,
                                                      item_embedding=item_embedding)
        print(base_scores.shape)

    def _update(self):
        from value_based.commons.args import HISTORY_SIZE

        # === Pretraining
        user_feat = torch.randn(self.args.batch_size, self.args.dim_user)
        user_history_feat = {
            "in_deep": torch.randn(self.args.batch_size, HISTORY_SIZE, self.args.dim_deep),
            "in_wide": torch.randn(self.args.batch_size, HISTORY_SIZE, self.args.dim_wide)
        }
        clicked_items = np.argmax(np.random.randn(self.actions.shape[0], self.args.slate_size), axis=-1)
        item_embedding = {
            "in_deep": torch.randn(self.args.batch_size, self.args.slate_size, self.args.dim_deep),
            "in_wide": torch.randn(self.args.batch_size, self.args.slate_size, self.args.dim_wide)
        }
        loss = self.reward_model.update(user_feat=user_feat,
                                        user_history_feat=user_history_feat,
                                        item_embedding=item_embedding,
                                        y=clicked_items)
        print("[Update] Loss: {:.4f}".format(loss))


if __name__ == '__main__':
    Test().test()
