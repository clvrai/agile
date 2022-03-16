import numpy as np
import torch
from value_based.policy.architecture.networks import NeuralNet
from value_based.commons.test import TestCase as test


class ActEncoder(object):
    def __init__(self, args: dict):
        self._args = args
        self._device = self._args.get("device", "cpu")
        self._dim_in = self._args.get("act_encoder_dim_in", 32)
        self._dim_out = self._args.get("act_encoder_dim_out", 32)
        self._num_candidates = self._args.get("num_candidates", 5000)

    def encode(self, node_feat: torch.tensor, batch_size: int):
        """ Encode the node features.

        Args:
            node_feat (torch.tensor): num_candidates x dim_node_feat
            batch_size (int): since batch_size is dependent on the epsilon decay policy,
                              we need to manually receive it from the policy

        Returns:
            gnn_embed (torch.tensor): batch_size x dim_gnn_embed
        """
        if not torch.is_tensor(node_feat):
            node_feat = torch.tensor(node_feat, dtype=torch.float32, device=self._device)

        gnn_embed = self._encode(node_feat=node_feat)  # batch_size x dim_gnn_embed
        if not torch.is_tensor(gnn_embed):
            gnn_embed = torch.tensor(gnn_embed.astype(np.float32), device=self._device)
        return gnn_embed  # batch_size x dim_gnn_embed

    def _encode(self, node_feat: torch.tensor):
        """ Inner method of encoder

        Args:
            node_feat: 1 x dim_node_feat

        Returns:
            gnn_embed: batch_size x (num_candidates * dim_gnn_embed)
        """
        raise NotImplementedError

    @property
    def encoder(self):
        return None


class BasicActEncoder(ActEncoder):
    def _encode(self, node_feat: torch.tensor):
        """ Inner method of encode

        Args:
            node_feat (): batch_size x dim_node

        Returns:
            batch_size x dim_gnn_embed
        """
        return node_feat.mean(dim=-1)[:, None].repeat((1, self._dim_out))  # batch_size x dim_gnn_embed


class DenseActEncoder(ActEncoder):
    def __init__(self, args: dict):
        super(DenseActEncoder, self).__init__(args=args)
        self._encoder = NeuralNet(dim_in=self._dim_in, dim_out=self._dim_out).to(self._device)
        if self._args["if_debug"]: print(">> {}".format(self._encoder))

    def _encode(self, node_feat: torch.tensor):
        """ Inner method of encode

        Args:
            node_feat (): batch_size x dim_node

        Returns:
            batch_size x dim_gnn_embed
        """
        return self._encoder(node_feat)  # batch_size x dim_gnn_embed

    @property
    def encoder(self):
        return self._encoder


class CNNActEncoder(ActEncoder):
    def __init__(self, dim_in, dim_gnn_embed, args):
        super(CNNActEncoder, self).__init__(args=args)
        self._encoder = NeuralNet(dim_in=dim_in, dim_out=dim_gnn_embed).to(self._device)
        if self._args["if_debug"]: print(">> {}".format(self._encoder))

    def _encode(self, node_feat: torch.tensor):
        return self._encoder(node_feat)  # 1 x dim_gnn_embed


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        # self.args.env_name = "ml-100k"
        self.args.env_name = "recsim"
        self.args.graph_type = "gcn"
        self.args.graph_aggregation_type = "sum"
        from value_based.commons.args import add_args
        self.args = add_args(args=self.args)
        self._prep()

    def test(self):
        print("=== _test ===")
        for act_encoder in [
            BasicActEncoder(args=vars(self.args)),
            DenseActEncoder(args=vars(self.args))
        ]:
            print(act_encoder)

            # Sample the node features of available items
            dim_node = self.args.graph_dim_hidden if self.args.graph_type != "gap" else self.args.graph_dim_in
            node_feat = torch.randn(self.args.batch_size, dim_node)

            # Feed in the action encoder to get the global features
            gnn_embed = act_encoder.encode(node_feat=node_feat, batch_size=self.args.batch_size)
            assert gnn_embed.shape == (self.args.batch_size, self.args.act_encoder_dim_out)
            assert torch.all(~torch.isnan(gnn_embed))


if __name__ == '__main__':
    Test().test()
