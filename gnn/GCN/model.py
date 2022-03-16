import torch
import torch.nn as nn

from gnn.GCN.layer import GraphConvolution
from value_based.commons.pt_activation_fns import ACTIVATION_FN
from value_based.commons.test import TestCase as test


class GCN(nn.Module):
    """ref: https://github.com/tkipf/pygcn/blob/master/pygcn/models.py"""

    def __init__(self,
                 dim_in: int,
                 dim_hidden: int = 32,
                 dim_out: int = 1,
                 num_hops: int = 2,
                 dropout: float = 0.6,
                 type_hidden_act_fn=10):
        """

        Args:
            dim_in ():
            dim_hidden ():
            dim_out ():
            num_hops ():
            dropout ():
            type_hidden_act_fn ():
        """
        super(GCN, self).__init__()

        self._num_hops = num_hops
        self.node_feat = None

        _in, _out = dim_in, dim_hidden
        for i in range(self._num_hops):
            setattr(self, "gcn{}".format(i), GraphConvolution(dim_in=_in,
                                                              dim_out=_out,
                                                              dropout=dropout,
                                                              type_hidden_act_fn=type_hidden_act_fn))
            setattr(self, "act{}".format(i), ACTIVATION_FN[type_hidden_act_fn]())
            _in, _out = _out, _out

        self.out_gcn = GraphConvolution(dim_in=_in,
                                        dim_out=dim_out,
                                        dropout=dropout,
                                        type_hidden_act_fn=type_hidden_act_fn)
        self.out_act = ACTIVATION_FN[type_hidden_act_fn]()

    def forward(self, x, adj):
        # === Intermediate Message Passing
        for i in range(self._num_hops):
            x = getattr(self, "gcn{}".format(i))(x, adj)
            x = getattr(self, "act{}".format(i))(x)
        self.node_feat = x  # store the intermediate node-features

        # === Output layer
        x = self.out_gcn(x, adj)
        # when dim_out is 1
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        x = self.out_act(x)
        return x


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
        gcn = GCN(num_hops=self.args.graph_num_hops,
                  dim_in=self.args.graph_dim_in,
                  dim_out=self.args.graph_dim_out).to(device=self.args.device)
        out = gcn(node_feat, adj_mat)
        print(out.shape)
        print(out)


if __name__ == '__main__':
    Test().test()
