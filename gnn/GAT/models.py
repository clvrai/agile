# Ref: https://github.com/Diego999/pyGAT/blob/master/models.py
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

from gnn.GAT.layers import GraphAttentionLayer
from gnn.Norm.norm import Norm


class Base(object):
    """ Base Class for returning the attention vector """
    a = None  # (np.ndarray): batch_size x intra_slate_step x num_nodes x num_nodes

    def get_attention(self, stack=False, first=False):
        """ Given the input, this returns the attention vector
        Returns:
             attention(torch.tensor): batch_size x num_nodes x num_nodes
        """
        # if self.out_att is not None:
        #     return self.out_att.get_attention()
        # else:
        #     return None
        if stack:
            return self.a
        elif first:
            return self.attention_heads[0].get_attention()
        elif self.out_att is not None:
            return self.out_att.get_attention()
        else:
            return None

    def stack_attention(self):
        """ Stack the attention across intra-slate time-steps """
        if self.out_att is not None:
            a = self.out_att.get_attention()  # batch_size x num_nodes x num_nodes
            if self.a is None:
                self.a = a[:, None, ...]
            else:
                self.a = np.concatenate([self.a, a[:, None, ...]], axis=1)

    def reset_attention(self):
        self.a = None

    def get_attention_stats(self, args=None, first=False, mean=True):
        attention = self.get_attention(first=first)
        _std = attention.std(-1)
        _max = attention.max(-1)[0]
        _min = attention.min(-1)[0]
        if mean:
            _std = _std.mean()
            _max = _max.mean()
            _min = _min.mean()
        if args is not None:
            args.gat_attention_std = _std
            args.gat_attention_max = _max
            args.gat_attention_min = _min
        else:
            return {"att_std": _std, "att_max": _max, "att_min": _min}


class Base_without_out(Base):
    """ Base class for returning attention vector of first GAT layer """

    def get_attention(self, head_id=0, stack=False, first=True):
        return self.attention_heads[head_id].get_attention()

    def stack_attention(self):
        # compatibility purpose
        pass


class GAT_Final(Base, nn.Module):
    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 dropout: float = 0.6,
                 alpha: float = 0.2,
                 args=None):
        super(GAT_Final, self).__init__()
        self.dropout = dropout
        self.node_feat = None
        args = Namespace(**args) if type(args) == dict else args
        self.args = args

        assert self.args.graph_gat_arch, "graph_gat_arch represents the GAT architecture!"
        self._layer_list = list()
        for i, _layer_name in enumerate(self.args.graph_gat_arch.split("_")):
            if _layer_name == "res":
                self._layer_list.append((_layer_name, None))
            else:
                if_out = i == (len(self.args.graph_gat_arch.split("_")) - 1)
                if _layer_name.endswith("VIZ"):
                    if_vis_attention = True
                    _layer_name = _layer_name.replace("VIZ", "")
                else:
                    if_vis_attention = False
                _layer, dim_in = eval(f"self._add_{_layer_name}")(dim_in=dim_in,
                                                                  dim_hidden=dim_hidden,
                                                                  dim_out=dim_hidden if not if_out else dim_out,
                                                                  dropout=dropout,
                                                                  alpha=alpha,
                                                                  num_heads=num_heads,
                                                                  args=vars(args))
                # self.add_module(f"layer{i}", _layer)
                self._layer_list.append((_layer_name, _layer))
                if if_vis_attention:
                    if type(_layer) == list:
                        _head_id = np.random.randint(low=0, high=len(_layer), size=1)[0]
                        _layer = _layer[_head_id]
                    self.out_att = _layer

    def _add_mha(self, **kwargs):  # mha: multi_head_attention
        _fn = [
            GraphAttentionLayer(dim_in=kwargs["dim_in"],
                                dim_out=kwargs["dim_out"],
                                dropout=kwargs["dropout"],
                                alpha=kwargs["alpha"],
                                concat=True).to(kwargs["args"]["device"])
            for _ in range(kwargs["num_heads"])
        ]
        next_dim_in = kwargs["dim_out"] * kwargs["num_heads"]
        return _fn, next_dim_in

    def _add_norm(self, **kwargs):
        _fn = Norm(norm_type=self.args.graph_norm_type, hidden_dim=kwargs["dim_hidden"]).to(kwargs["args"]["device"])
        next_dim_in = kwargs["dim_in"]
        return _fn, next_dim_in

    def _add_gat(self, **kwargs):
        _fn = GraphAttentionLayer(dim_in=kwargs["dim_in"],
                                  dim_out=kwargs["dim_out"],
                                  dropout=kwargs["dropout"],
                                  alpha=kwargs["alpha"],
                                  concat=False).to(kwargs["args"]["device"])
        next_dim_in = kwargs["dim_out"]
        return _fn, next_dim_in

    def _add_mlp(self, **kwargs):
        _fn = nn.Sequential(nn.Linear(kwargs["dim_in"], kwargs["dim_out"])).to(kwargs["args"]["device"])
        next_dim_in = kwargs["dim_out"]
        return _fn, next_dim_in

    def forward(self, x, adj):
        """
        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        _input = x.clone()
        for _content in self._layer_list:
            _layer_name, _layer = _content
            if _layer_name == "mha":
                x = torch.cat([h(x, adj) for h in _layer], dim=-1)  # batch_size x num_nodes x (dim_hidden * num_heads)
            elif _layer_name in ["mlp", "norm"]:
                x = _layer(x)  # batch_size x num_nodes x dim_hidden
            elif _layer_name == "gat":
                x = _layer(x, adj)  # batch_size x num_nodes x dim_hidden
            elif _layer_name == "res":
                assert x.shape == _input.shape, f"{x.shape}, {_input.shape}"
                # Teleport Term in GNN
                # Ref: https://arxiv.org/pdf/1810.05997.pdf or Eq(3) in https://arxiv.org/pdf/2006.14897.pdf
                x += (_input * self.args.gnn_alpha_teleport)
            else:
                raise ValueError
        return x


class GAT2(Base, nn.Module):
    """ Almost same as GATSimple but one more GAT instead of MLP """

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 alpha: float = 0.2,  # alpha is for LeakyReLU
                 args=None):
        super(GAT2, self).__init__()
        self.node_feat = None

        args = Namespace(**args) if type(args) == dict else args
        self.args = args

        self.attention_heads = [
            GraphAttentionLayer(dim_in=dim_in,
                                dim_out=dim_hidden,
                                dropout=getattr(self.args, "gat_dropout", 0),
                                alpha=getattr(self.args, "gat_leakerelu_alpha", 0.2),
                                concat=True)
            for _ in range(num_heads)
        ]

        for i, attention in enumerate(self.attention_heads):
            self.add_module('attention_{}'.format(i), attention)

        self.out_att = GraphAttentionLayer(dim_in=dim_hidden * num_heads,
                                           dim_out=dim_out,
                                           dropout=getattr(self.args, "gat_dropout", 0),
                                           alpha=getattr(self.args, "gat_leakerelu_alpha", 0.2),
                                           concat=False)

        self.norm = Norm(norm_type=self.args.graph_norm_type, hidden_dim=dim_out)

    def forward(self, input, adj):
        """
        Args:
            input (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        # batch_size x num_nodes x (dim_hidden * num_heads)
        x = torch.cat([att_head(input, adj) for att_head in self.attention_heads], dim=-1)
        x = F.elu(x)
        x = self.out_att(x, adj)  # batch_size x num_nodes x dim_out
        self.node_feat = x  # store the intermediate node-features
        if self.args.gnn_residual_connection:
            assert x.shape == input.shape, f"{x.shape}, {input.shape}"
            # Ref: https://arxiv.org/pdf/1810.05997.pdf or Eq(3) in https://arxiv.org/pdf/2006.14897.pdf
            # Teleport Term in GNN
            x += (input * self.args.gnn_alpha_teleport)
        if self.norm is not None:
            # reshape for batch_size x num_nodes x dim_node
            x = self.norm(x)
        return x


class GATSimple(Base_without_out, nn.Module):
    """Dense version of GAT, made specifically for GCDQN - with no dropout and mlp changes."""

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 alpha: float = 0.2,
                 args=None):
        super(GATSimple, self).__init__()
        self.node_feat = None

        args = Namespace(**args) if type(args) == dict else args
        self.args = args

        self.attention_heads = [
            GraphAttentionLayer(dim_in=dim_in, dim_out=dim_hidden, dropout=0, alpha=alpha, concat=True)
            for _ in range(num_heads)
        ]

        for i, attention in enumerate(self.attention_heads):
            self.add_module('attention_{}'.format(i), attention)

        self.head_mlp = nn.Sequential(
            nn.Linear(dim_hidden * num_heads, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_out))

        if self.args.gnn_layer_norm:
            self.layer_norm = nn.LayerNorm(dim_out)

    def forward(self, input, adj):
        """
        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        # === Intermediate Message Passing; batch_size x num_nodes x (dim_hidden * num_heads)
        x = torch.cat([att_head(input, adj) for att_head in self.attention_heads], dim=-1)

        x = self.head_mlp(x)
        self.node_feat = x  # store the intermediate node-features
        if self.args.gnn_residual_connection:
            assert x.shape == input.shape, f"{x.shape}, {input.shape}"
            x += input
        if self.args.gnn_layer_norm:
            x = self.layer_norm(x)

        return x


class GATSimple_RecSim(Base_without_out, nn.Module):
    """Dense version of GAT, made specifically for GCDQN - with no dropout and mlp changes."""

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 alpha: float = 0.2,
                 args=None):
        super(GATSimple_RecSim, self).__init__()
        self.node_feat = None

        args = Namespace(**args) if type(args) == dict else args
        self.args = args

        self.attention_heads = [
            GraphAttentionLayer(dim_in=dim_in, dim_out=dim_hidden, dropout=0, alpha=alpha, concat=True)
            for _ in range(num_heads)
        ]

        for i, attention in enumerate(self.attention_heads):
            self.add_module('attention_{}'.format(i), attention)

        self.head_mlp = nn.Sequential(
            nn.Linear(dim_hidden * num_heads, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_out))

    def forward(self, input, adj):
        """
        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        # === Intermediate Message Passing; batch_size x num_nodes x (dim_hidden * num_heads)
        x = torch.cat([att_head(input, adj) for att_head in self.attention_heads], dim=-1)

        if self.args.gnn_residual_connection:
            assert x.shape == input.shape, f"{x.shape}, {input.shape}"
            # Teleport Term in GNN
            # Ref: https://arxiv.org/pdf/1810.05997.pdf or Eq(3) in https://arxiv.org/pdf/2006.14897.pdf
            x += (input * self.args.gnn_alpha_teleport)

        x = self.head_mlp(x)
        self.node_feat = x  # store the intermediate node-features
        return x


def test():
    print("=== test ===")
    batch_size = 9
    dim_in, dim_hidden = 32, 16
    dim_out = dim_in
    args = {
        "gcdqn_gat_two_hops": False,
        "gnn_ppo": False,
        "graph_norm_type": "bn",
        "num_candidates": 20,
        "device": "cuda",
        "gat_scale_attention": 1.0,
        "gnn_residual_connection": True,
        "num_heads": 1,
        "gnn_alpha_teleport": 0.9,
        "graph_gat_arch": "mha_res_gatVIZ_res_norm_mha_mlp",
        "gnn_layer_norm": True,
    }

    _input = torch.randn(batch_size, args["num_candidates"], dim_in, device=args["device"])
    Adj = torch.ones(args["num_candidates"], args["num_candidates"], device=args["device"]) - \
          torch.eye(args["num_candidates"], device=args["device"])
    Adj = Adj[None, ...].repeat(batch_size, 1, 1)

    # for graph_gat_arch in [
    #     "gatVIZ",
    #     "mhaVIZ_mlp",
    #     "gat_gatVIZ",
    #     "gat_res_gatVIZ",
    #     "gat_norm_gatVIZ",
    #     "mha_gatVIZ",
    #     "mha_gat_gatVIZ",
    #     "mha_gat_res_gatVIZ",
    #     "mha_gat_norm_gatVIZ_mlp",
    # ]:
    #     args["graph_gat_arch"] = graph_gat_arch
    #     gat = GAT_Final(dim_in=dim_in,
    #                     dim_hidden=dim_in,
    #                     dim_out=dim_out,
    #                     num_heads=args["num_heads"],
    #                     args=args).to(args["device"])
    #     print(f"=== {graph_gat_arch} ===")
    #     out = gat(_input, Adj)
    #     assert out.shape == (batch_size, args["num_candidates"], dim_out)

    for model in [GAT2, GATSimple, GAT_Final, GATSimple_RecSim]:
        gat = model(dim_in=dim_in, dim_hidden=dim_in, dim_out=dim_out, num_heads=1, args=args).to(args["device"])
        for _ in range(5):
            out = gat(_input, Adj)
            gat.stack_attention()
        a = gat.get_attention(first=True)
        print(gat.__class__.__name__, out.shape, a.shape, a.max().item(), a.min().item())
        a = gat.get_attention(first=False)
        print(gat.__class__.__name__, out.shape, a.shape, a.max().item(), a.min().item())
        asdf
        if a is not None:
            print(gat.__class__.__name__, out.shape, a.shape, a.max().item(), a.min().item())
        else:
            print(gat.__class__.__name__, out.shape)


if __name__ == '__main__':
    test()
