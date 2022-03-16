import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

# from value_based.gnn.GAT.layers_geometric import GATConv, create_edge_index_for_fully_connected_graph
from gnn.GAT.layers_geometric2 import GATConv, create_edge_index_for_fully_connected_graph
from gnn.Norm.norm import Norm


class Base(object):
    """ Base Class for returning the attention vector """
    _a = _batch_size = _num_nodes = _args = None

    def get_attention(self, stack=False, first=False):
        """ Given the input, this returns the attention vector
        Returns:
             attention(torch.tensor): batch_size x num_nodes x num_nodes or
                                      batch_size x intra_slate_step x num_nodes x num_nodes
        """
        if self._a is not None:
            # Reshape the attention vector; batch_size x num_senders x num_receivers
            if len(self._a.shape) == 2:
                intra_slate_step, chunk = self._a.shape
                attention = self._a.reshape(intra_slate_step, self._batch_size, self._num_nodes, self._num_nodes)
                attention = attention.permute(1, 0, 2, 3)
            else:
                attention = self._a.reshape(self._batch_size, self._num_nodes, self._num_nodes)
            return attention
        else:
            return None

    def reset_attention(self):
        self._a = None

    def stack_attention(self):
        pass  # compatibility purpose

    def get_attention_stats(self, args):
        attention = self.get_attention()
        _std = attention.std(-1).mean()
        _max = attention.max(-1)[0].mean()
        _min = attention.min(-1)[0].mean()
        if args is not None:
            args.gat_attention_std = _std
            args.gat_attention_max = _max
            args.gat_attention_min = _min
        else:
            return {"att_std": _std, "att_max": _max, "att_min": _min}

    def _stack_attention(self, a):
        if self._a is None:
            self._a = a.squeeze(-1)[None, :]
        else:
            self._a = torch.cat([self._a, a.squeeze(-1)[None, :]], dim=0)


class GAT(Base, nn.Module):
    """Dense version of GAT."""

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 dropout: float = 0.6,
                 args=None):
        super(GAT, self).__init__()
        args = Namespace(**args) if type(args) == dict else args
        self._args = args

        # For compatibility b/w RecSim and CREATE
        self._args.if_create = self._args.env_name.startswith("Create")
        if self._args.if_create:
            self._args.num_candidates = self._args.action_set_size

        self.dropout = dropout
        self.node_feat, self._batch_size, self._num_nodes, self._a = None, None, None, None
        self._dim_out = dim_out

        # 2 x num_edges(num_items**2)
        self.base_edge_index = np.asarray(
            [[i, j] for i in range(args.num_candidates) for j in range(args.num_candidates)]
        ).T

        self.conv = GATConv(dim_in, dim_hidden, heads=num_heads, flow="target_to_source")
        self.norm = Norm(norm_type=self._args.graph_norm_type, hidden_dim=dim_hidden * num_heads)
        self.out_att = GATConv(dim_hidden * num_heads, self._dim_out, concat=False, heads=1, flow="target_to_source")

        self.edge_index_dict = {}

    def forward(self, x, adj):
        """
        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        self._batch_size, self._num_nodes, dim_node = x.shape
        x = x.reshape(self._batch_size * self._num_nodes, dim_node)  # (batch_size * num_nodes) x dim_node
        if (self._batch_size, self._num_nodes) in self.edge_index_dict:
            edge_index = self.edge_index_dict[(self._batch_size, self._num_nodes)]
        else:
            edge_index = create_edge_index_for_fully_connected_graph(batch_size=self._batch_size,
                                                                     num_nodes=self._num_nodes,
                                                                     base_edge_index=self.base_edge_index)
            self.edge_index_dict[(self._batch_size, self._num_nodes)] = edge_index
        edge_index = torch.tensor(edge_index, device=self._args.device)

        # === Intermediate Message Passing
        x = F.dropout(x, self.dropout, training=self.training)  # (batch_size * num_nodes) x dim_node
        x = self.conv(x, edge_index, batch_size=self._batch_size)  # (batch_size * num_nodes) x (dim_hidden * num_heads)
        x = F.dropout(x, self.dropout, training=self.training)  # (batch_size * num_nodes) x (dim_hidden * num_heads)
        self.node_feat = x  # store the intermediate node-features

        if self.norm is not None:
            # reshape for batch_size x num_nodes x dim_node
            x = x.reshape(self._batch_size, self._num_nodes, x.shape[-1])
            x = self.norm(x)
            x = x.reshape(self._batch_size * self._num_nodes, x.shape[-1])

        # === Output layer
        x, a = self.out_att(x, edge_index, return_attention_weights=True,
                            batch_size=self._batch_size)  # (batch_size * num_nodes) x dim_out
        self._stack_attention(a=a)
        out = F.elu(x)  # (batch_size * num_nodes) x dim_out

        out = out.reshape(self._batch_size, self._num_nodes,
                          self._dim_out)  # reshape the output according to batch_size
        return out


class GAT2(Base, nn.Module):
    """Dense version of GAT, made specifically for GCDQN - with no dropout and mlp changes."""

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 args=None):
        super(GAT2, self).__init__()
        args = Namespace(**args) if type(args) == dict else args
        self._args = args

        # For compatibility b/w RecSim and CREATE
        self._args.if_create = self._args.env_name.startswith("Create")
        if self._args.if_create:
            self._args.num_candidates = self._args.action_set_size

        self._dim_out = dim_out
        self.node_feat, self._batch_size, self._num_nodes, self._a = None, None, None, None

        # 2 x num_edges(num_items**2)
        self.base_edge_index = np.asarray(
            [[i, j] for i in range(self._args.num_candidates) for j in range(self._args.num_candidates)]
        ).T

        self.conv = GATConv(dim_in, dim_hidden, heads=num_heads, flow="target_to_source")
        self.norm = Norm(norm_type=self._args.graph_norm_type, hidden_dim=dim_hidden * num_heads)
        self.out_att = GATConv(dim_hidden * num_heads, self._dim_out, concat=False, heads=1, flow="target_to_source")

        self.edge_index_dict = {}

    def forward(self, x, adj):
        """
        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        self._batch_size, self._num_nodes, dim_node = x.shape
        x = x.reshape(self._batch_size * self._num_nodes, dim_node)  # (batch_size * num_nodes) x dim_node
        if (self._batch_size, self._num_nodes) in self.edge_index_dict:
            edge_index = self.edge_index_dict[(self._batch_size, self._num_nodes)]
        else:
            edge_index = create_edge_index_for_fully_connected_graph(batch_size=self._batch_size,
                                                                     num_nodes=self._num_nodes,
                                                                     base_edge_index=self.base_edge_index)
            self.edge_index_dict[(self._batch_size, self._num_nodes)] = edge_index
        edge_index = torch.tensor(edge_index, device=self._args.device)

        # === Intermediate Message Passing
        x, a = self.conv(x, edge_index, return_attention_weights=True,
                         batch_size=self._batch_size)  # (batch_size * num_nodes) x (dim_hidden * num_heads)
        self._stack_attention(a=a)
        self.node_feat = x  # store the intermediate node-features

        if self.norm is not None:
            # reshape for batch_size x num_nodes x dim_node
            x = x.reshape(self._batch_size, self._num_nodes, x.shape[-1])
            x = self.norm(x)
            x = x.reshape(self._batch_size * self._num_nodes, x.shape[-1])

        x = F.elu(x)

        # === Output layer
        out = self.out_att(x, edge_index, batch_size=self._batch_size)  # (batch_size * num_nodes) x dim_out

        # TODO: legacy code so remove!
        # out, a = self.out_att(x, edge_index, return_attention_weights=True,
        #                       batch_size=self._batch_size)  # (batch_size * num_nodes) x dim_out
        # self._a = a.squeeze(-1)
        # if self._a is None:
        #     self._a = a.squeeze(-1)
        # else:
        #     self._a = torch.vstack([self._a, a.squeeze(-1)])

        out = out.reshape(self._batch_size, self._num_nodes,
                          self._dim_out)  # reshape the output according to batch_size
        return out


class GAT3(Base, nn.Module):
    """Dense version of GAT, made specifically for GCDQN - with no dropout and mlp changes."""

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 alpha: float = 0.2,
                 args=None):
        super(GAT3, self).__init__()
        args = Namespace(**args) if type(args) == dict else args
        self._args = args

        # For compatibility b/w RecSim and CREATE
        self._args.if_create = self._args.env_name.startswith("Create")
        if self._args.if_create:
            self._args.num_candidates = self._args.action_set_size

        self.node_feat, self._batch_size, self._num_nodes, self._a = None, None, None, None
        self._dim_out = dim_out

        # 2 x num_edges(num_items**2)
        self.base_edge_index = np.asarray(
            [[i, j] for i in range(args.num_candidates) for j in range(args.num_candidates)]
        ).T

        self.conv = GATConv(dim_in, dim_hidden, heads=num_heads, flow="target_to_source")
        self.norm = Norm(norm_type=self._args.graph_norm_type, hidden_dim=dim_hidden * num_heads)

        if self._args.gcdqn_gat_two_hops:
            self.conv2 = GATConv(dim_hidden * num_heads, dim_hidden, heads=num_heads, flow="target_to_source")

        self.mlp = nn.Sequential(nn.Linear(dim_hidden * num_heads, self._dim_out))

        self.edge_index_dict = {}

    def forward(self, x, adj):
        """
        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        self._batch_size, self._num_nodes, dim_node = x.shape
        x = x.reshape(self._batch_size * self._num_nodes, dim_node)  # (batch_size * num_nodes) x dim_node
        if (self._batch_size, self._num_nodes) in self.edge_index_dict:
            edge_index = self.edge_index_dict[(self._batch_size, self._num_nodes)]
        else:
            edge_index = create_edge_index_for_fully_connected_graph(batch_size=self._batch_size,
                                                                     num_nodes=self._num_nodes,
                                                                     base_edge_index=self.base_edge_index)
            self.edge_index_dict[(self._batch_size, self._num_nodes)] = edge_index
        edge_index = torch.tensor(edge_index, device=self._args.device)

        # === Intermediate Message Passing
        # (batch_size * num_nodes) x (dim_hidden * num_heads)
        x, _ = self.conv(x, edge_index, return_attention_weights=True, batch_size=self._batch_size)

        if self.norm is not None:
            # reshape for batch_size x num_nodes x dim_node
            x = x.reshape(self._batch_size, self._num_nodes, x.shape[-1])
            x = self.norm(x)
            x = x.reshape(self._batch_size * self._num_nodes, x.shape[-1])

        # TODO: get_attention isn't working for multi-head attention....
        # x, a = self.conv(x, edge_index, return_attention_weights=True)
        # if self._a is None:
        #     self._a = a.squeeze(-1)
        # else:
        #     self._a = torch.vstack([self._a, a.squeeze(-1)])
        if self._args.gcdqn_gat_two_hops:
            x = self.conv2(x, edge_index,
                           batch_size=self._batch_size)  # (batch_size * num_nodes) x (dim_hidden * num_heads)

        out = self.mlp(x)

        out = out.reshape(self._batch_size, self._num_nodes, self._dim_out)  # reshape the output
        return out


class GAT4(Base, nn.Module):
    """ For visualisation of attention """

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 alpha: float = 0.2,
                 args=None):
        super(GAT4, self).__init__()
        self.node_feat, self._batch_size, self._num_nodes, self._a = None, None, None, None
        args = Namespace(**args) if type(args) == dict else args
        self._args = args

        # For compatibility b/w RecSim and CREATE
        self._args.if_create = self._args.env_name.startswith("Create")
        if self._args.if_create:
            self._args.num_candidates = self._args.action_set_size

        self._dim_out = dim_out

        # 2 x num_edges(num_items**2)
        self.base_edge_index = np.asarray(
            [[i, j] for i in range(self._args.num_candidates) for j in range(self._args.num_candidates)]
        ).T

        self.conv = GATConv(dim_in, dim_hidden, heads=num_heads, flow="target_to_source")
        norm_type = self._args.get('graph_norm_type', 'gn') if hasattr(self._args,
                                                                       'get') else self._args.graph_norm_type
        self.norm = Norm(norm_type=norm_type, hidden_dim=dim_hidden * num_heads)

        if self._args.gcdqn_gat_two_hops:
            self.conv2 = GATConv(dim_hidden * num_heads, dim_hidden, heads=num_heads, flow="target_to_source")

        self.out_att = GATConv(dim_hidden * num_heads, dim_hidden, concat=False, heads=1, flow="target_to_source")
        self.mlp = nn.Sequential(nn.Linear(dim_hidden, self._dim_out))

        self.edge_index_dict = {}

    def forward(self, x, adj):
        """
        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        self._batch_size, self._num_nodes, dim_node = x.shape
        x = x.reshape(self._batch_size * self._num_nodes, dim_node)  # (batch_size * num_nodes) x dim_node
        if (self._batch_size, self._num_nodes) in self.edge_index_dict:
            edge_index = self.edge_index_dict[(self._batch_size, self._num_nodes)]
        else:
            edge_index = create_edge_index_for_fully_connected_graph(batch_size=self._batch_size,
                                                                     num_nodes=self._num_nodes,
                                                                     base_edge_index=self.base_edge_index)
            self.edge_index_dict[(self._batch_size, self._num_nodes)] = edge_index
        edge_index = torch.tensor(edge_index, device=self._args.device)

        # === Intermediate Message Passing
        x = self.conv(x, edge_index, batch_size=self._batch_size)  # (batch_size * num_nodes) x (dim_hidden * num_heads)

        if self.norm is not None:
            # reshape for batch_size x num_nodes x dim_node
            x = x.reshape(self._batch_size, self._num_nodes, x.shape[-1])
            x = self.norm(x)
            x = x.reshape(self._batch_size * self._num_nodes, x.shape[-1])

        if self._args.gcdqn_gat_two_hops:
            x = self.conv2(x, edge_index)  # (batch_size * num_nodes) x (dim_hidden * num_heads)
        x, a = self.out_att(x, edge_index, return_attention_weights=True,
                            batch_size=self._batch_size)  # (batch_size * num_nodes) x dim_out
        self._stack_attention(a=a)
        out = self.mlp(x)
        out = out.reshape(self._batch_size, self._num_nodes, self._dim_out)  # reshape the output
        return out


class GAT5(Base, nn.Module):
    """ For visualisation of attention """

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 alpha: float = 0.2,
                 args=None):
        super(GAT5, self).__init__()
        self.node_feat, self._batch_size, self._num_nodes, self._a = None, None, None, None
        args = Namespace(**args) if type(args) == dict else args
        self._args = args

        # For compatibility b/w RecSim and CREATE
        self._args.if_create = self._args.env_name.startswith("Create")
        if self._args.if_create:
            self._args.num_candidates = self._args.action_set_size

        self._dim_out = dim_out

        # 2 x num_edges(num_items**2)
        self.base_edge_index = np.asarray(
            [[i, j] for i in range(self._args.num_candidates) for j in range(self._args.num_candidates)]
        ).T

        self.out_att = GATConv(dim_in, dim_out, concat=False, heads=1, flow="target_to_source")

        self.edge_index_dict = {}

    def forward(self, x, adj):
        """
        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        self._batch_size, self._num_nodes, dim_node = x.shape
        x = x.reshape(self._batch_size * self._num_nodes, dim_node)  # (batch_size * num_nodes) x dim_node
        if (self._batch_size, self._num_nodes) in self.edge_index_dict:
            edge_index = self.edge_index_dict[(self._batch_size, self._num_nodes)]
        else:
            edge_index = create_edge_index_for_fully_connected_graph(batch_size=self._batch_size,
                                                                     num_nodes=self._num_nodes,
                                                                     base_edge_index=self.base_edge_index)
            self.edge_index_dict[(self._batch_size, self._num_nodes)] = edge_index
        edge_index = torch.tensor(edge_index, device=self._args.device)

        out, a = self.out_att(x, edge_index, return_attention_weights=True,
                              batch_size=self._batch_size)  # (batch_size * num_nodes) x dim_out
        self._stack_attention(a=a)
        out = out.reshape(self._batch_size, self._num_nodes, self._dim_out)  # reshape the output
        return out


# todo: Issue in Gradient flow when tested in gcdqn.py
class GATSimple_RecSim(Base, nn.Module):
    """Dense version of GAT, made specifically for GCDQN - with no dropout and mlp changes."""

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_heads: int = 1,
                 alpha: float = 0.2,
                 args=None):
        super(GATSimple_RecSim, self).__init__()

        self.node_feat, self._batch_size, self._num_nodes, self._a = None, None, None, None
        args = Namespace(**args) if type(args) == dict else args
        self._args = args

        # For compatibility b/w RecSim and CREATE
        self._args.if_create = self._args.env_name.startswith("Create")
        if self._args.if_create:
            self._args.num_candidates = self._args.action_set_size

        self._dim_out = dim_out

        # 2 x num_edges(num_items**2)
        self.base_edge_index = np.asarray(
            [[i, j] for i in range(self._args.num_candidates) for j in range(self._args.num_candidates)]
        ).T

        self.attention_heads = GATConv(dim_in, dim_hidden, heads=num_heads, flow="target_to_source")
        norm_type = self._args.get('graph_norm_type', 'gn') if hasattr(self._args,
                                                                       'get') else self._args.graph_norm_type
        self.norm = Norm(norm_type=norm_type, hidden_dim=dim_hidden * num_heads)

        self.head_mlp = nn.Sequential(
            nn.Linear(dim_hidden * num_heads, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_out))

        self.edge_index_dict = {}

    def forward(self, x, adj):
        """
        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_out
        """
        self._batch_size, self._num_nodes, dim_node = x.shape
        x = x.reshape(self._batch_size * self._num_nodes, dim_node)  # (batch_size * num_nodes) x dim_node
        if (self._batch_size, self._num_nodes) in self.edge_index_dict:
            edge_index = self.edge_index_dict[(self._batch_size, self._num_nodes)]
        else:
            edge_index = create_edge_index_for_fully_connected_graph(batch_size=self._batch_size,
                                                                     num_nodes=self._num_nodes,
                                                                     base_edge_index=self.base_edge_index)
            self.edge_index_dict[(self._batch_size, self._num_nodes)] = edge_index
        edge_index = torch.tensor(edge_index, device=self._args.device)

        # === Intermediate Message Passing; (batch_size * num_nodes) x (dim_hidden * num_heads)
        _x, a = self.attention_heads(x, edge_index, batch_size=self._batch_size, return_attention_weights=True)
        self._stack_attention(a=a)

        if self._args.gnn_residual_connection:
            assert _x.shape == x.shape, f"{x.shape}, {_x.shape}"
            # Teleport Term in GNN
            # Ref: https://arxiv.org/pdf/1810.05997.pdf or Eq(3) in https://arxiv.org/pdf/2006.14897.pdf
            _x += (x * self._args.gnn_alpha_teleport)

        out = self.head_mlp(_x)
        # TODO: Try LayerNorm
        out = out.reshape(self._batch_size, self._num_nodes, self._dim_out)  # reshape the output
        return out


def test():
    from value_based.commons.seeds import set_randomSeed
    set_randomSeed(seed=1)
    print("=== test ===")
    batch_size = 9
    dim_in, dim_hidden = 32, 16
    dim_out, num_heads = 1, 1
    args = {
        "env_name": "recsim",
        "gcdqn_gat_two_hops": False,
        "gnn_ppo": False,
        "graph_norm_type": "bn",
        "num_candidates": 10,
        "device": "cpu",
        "gat_scale_attention": 1.0,
        "gnn_residual_connection": True,
        "gnn_alpha_teleport": 0.9,
    }

    _input = torch.randn(batch_size, args["num_candidates"], dim_in)
    Adj = torch.ones(args["num_candidates"], args["num_candidates"]) - torch.eye(args["num_candidates"])
    Adj = Adj[None, ...].repeat(batch_size, 1, 1)

    for model in [GAT, GAT2, GAT3, GAT4, GAT5, GATSimple_RecSim]:
        gat = model(dim_in=dim_in, dim_hidden=dim_in, dim_out=dim_out, num_heads=num_heads, args=args)
        for _ in range(5):
            out = gat(_input, Adj)
            gat.stack_attention()
        a = gat.get_attention()
        if a is not None:
            print(gat.__class__.__name__, out.shape, a.shape)
        else:
            print(gat.__class__.__name__, out.shape)


if __name__ == '__main__':
    test()
