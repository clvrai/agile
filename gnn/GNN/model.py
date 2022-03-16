import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace

from gnn.Norm.norm import Norm

try:
    from torch_geometric.nn import GCNConv, DenseGCNConv
except:
    from gnn.GCN.model_geometric import DenseGCNConv

    # TODO: do something...
    GCNConv = None


class GNN_Q(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out=1, **kwargs):
        super().__init__()
        self.conv1 = GCNConv(dim_in, dim_hidden, cached=True, normalize=True)
        self.conv2 = GCNConv(dim_hidden, dim_out, cached=True, normalize=True)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)

        return x


class Dense_GNN(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out=1, args=None, **kwargs):
        super().__init__()
        self.conv1 = DenseGCNConv(dim_in, dim_hidden)
        self.mlp1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(dim_hidden + dim_in, dim_hidden),
            nn.ReLU()
        )
        self.conv2 = DenseGCNConv(dim_hidden, dim_hidden)
        self.mlp2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * dim_hidden, dim_out),
        )

    def forward(self, x, adj):
        x1 = self.conv1(x, adj, add_loop=False)  # Aggregates all edge messages
        x2 = self.mlp1(torch.cat([x1, x], dim=-1))
        x3 = self.conv2(x2, adj, add_loop=False)
        x4 = self.mlp2(torch.cat([x3, x2], dim=-1))

        return x4


class Dense_GNN_MLP(torch.nn.Module):
    def __init__(self, dim_in, dim_hidden, dim_out=1, args=None, **kwargs):
        super().__init__()
        self.mlp0 = nn.Sequential(
            nn.Linear(dim_in, dim_hidden),
            nn.ReLU()
        )
        self.conv1 = DenseGCNConv(dim_hidden, dim_hidden)
        self.mlp1 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * dim_hidden, dim_hidden),
            nn.ReLU()
        )
        self.conv2 = DenseGCNConv(dim_hidden, dim_hidden)
        self.mlp2 = nn.Sequential(
            nn.ReLU(),
            nn.Linear(2 * dim_hidden, dim_out),
        )

    def forward(self, x, adj):
        x0 = self.mlp0(x)
        x1 = self.conv1(x0, adj, add_loop=False)  # Aggregates all edge messages
        x2 = self.mlp1(torch.cat([x0, x1], dim=-1))
        x3 = self.conv2(x2, adj, add_loop=False)
        x4 = self.mlp2(torch.cat([x2, x3], dim=-1))

        return x4


class GCN2(torch.nn.Module):
    """ Same architecture as GAT2 """
    def __init__(self, dim_in, dim_hidden, dim_out=1, args=None, **kwargs):
        super().__init__()
        self.node_feat = None

        args = Namespace(**args) if type(args) == dict else args
        self.args = args

        self.conv = DenseGCNConv(dim_in, dim_hidden)
        self.out_conv = DenseGCNConv(dim_hidden, dim_out)
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
        x = self.conv(input, adj, add_loop=False)
        x = F.elu(x)
        x = self.out_conv(x, adj, add_loop=False)
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


def test():
    print("=== test ===")
    batch_size = 9
    dim_in, dim_hidden = 32, 16
    dim_out = dim_in
    args = {
        "gcdqn_gat_two_hops": False,
        "gnn_ppo": False,
        "graph_norm_type": "bn",
        "num_candidates": 10,
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

    for model in [GCN2]:
        gat = model(dim_in=dim_in, dim_hidden=dim_in, dim_out=dim_out, num_heads=1, args=args).to(args["device"])
        for _ in range(5):
            out = gat(_input, Adj)
        print(gat.__class__.__name__, out.shape)


if __name__ == '__main__':
    test()
