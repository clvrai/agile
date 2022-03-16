"""ref: https://github.com/tkipf/pygcn/blob/master/pygcn/layers.py"""
import torch
import torch.nn as nn

from value_based.commons.init_layer import init_layer
from value_based.commons.pt_activation_fns import ACTIVATION_FN
from value_based.commons.test import TestCase as test


class GraphConvolution(nn.Module):
    """ GCN with Dropout for Homogeneous Graph

        References
        - https://arxiv.org/abs/1609.02907
    """

    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.0, type_hidden_act_fn=21):
        super(GraphConvolution, self).__init__()

        if dropout != 0.0:
            self.dropout = nn.Dropout(dropout)
        else:
            self.dropout = None
        self.linear = nn.Linear(in_features=dim_in, out_features=dim_out).apply(init_layer)
        self.act = ACTIVATION_FN[type_hidden_act_fn]()

    def forward(self, node_feat, adj_mat):
        """
        Args:
            node_feat: batch_size x num_candidates x (dim_item + dim_state)
            adj_mat: batch_size x num_candidates x num_candidates

        Returns:
            output: batch_size x num_candidates x dim_hidden
        """
        if self.dropout is not None:
            # Apply Dropout
            node_feat = self.dropout(node_feat)  # batch_size x num_candidates x (dim_item + dim_state)

        # Formula: z_tp1 = sigma(AWZ_t)
        Z = self.act(self.linear(node_feat))  # batch_size x num_candidates x dim_hidden
        Z = torch.matmul(adj_mat, Z)  # batch_size x num_candidates x dim_hidden
        # Z = torch.bmm(adj_mat, Z)  # this does the same thing!
        return Z


class Test(test):
    def __init__(self):
        self._get_args()
        # self.args.if_debug = True
        self.args.if_debug = False
        # self.args.env_name = "ml-100k"
        self.args.env_name = "recsim"
        self._prep()

    def test(self):
        print("=== test ===")
        node_feat = torch.randn(self.args.batch_size,
                                self.args.num_candidates,
                                self.args.graph_dim_in).to(device=self.args.device)
        adj_mat = torch.ones(size=(self.args.batch_size, self.args.num_candidates, self.args.num_candidates),
                             dtype=torch.float32,
                             device=self.args.device)
        adj_mat -= torch.eye(self.args.num_candidates)

        layer = GraphConvolution(dim_in=self.args.graph_dim_in,
                                 dim_out=self.args.graph_dim_out,
                                 type_hidden_act_fn=21,
                                 dropout=0.0).to(device=self.args.device)
        out = layer(node_feat, adj_mat)
        print(out.shape)
        print(out)


if __name__ == '__main__':
    Test().test()
