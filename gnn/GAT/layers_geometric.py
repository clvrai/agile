import torch
import numpy as np
import torch.nn.functional as F

from torch import Tensor
from copy import deepcopy
from torch.nn import Parameter, Linear
from typing import Union, Tuple, Optional

from gnn.torch_geometric_utils.utils import softmax
from gnn.torch_geometric_utils.torch_geometric_nn_inits import zeros, glorot
from gnn.torch_geometric_utils.torch_geometric_typing import OptPairTensor, Adj, Size, OptTensor
from gnn.torch_geometric_utils.message_passing import MessagePassing


def create_edge_index_for_fully_connected_graph(batch_size: int = 5, num_nodes: int = 10, base_edge_index=None):
    """ create_edge_index_for_fully_connected_graph

    Args:
        batch_size (int):
        num_nodes (int):
        base_edge_index (np.ndarray or None): Can be None but if it exists, then it's going to be used to populate the
                                              edges of sub-graphs

    Returns:
        edge_index (np.ndarray): 2 x (batch_size * num_edges)
    """
    if base_edge_index is None:
        # 2 x num_edges(num_items**2)
        edge_index = np.asarray([[i, j] for i in range(num_nodes) for j in range(num_nodes)]).T
    else:
        assert base_edge_index.shape[0] == 2, "Shape of edge_index is 2 x (batch_size * num_edges)"
        edge_index = base_edge_index

    """ This edge index contains the edges of multiple sub-graphs
        ** Create edge index for each sub-graph and concatenate all the edges in those sub-graphs into edge_index
        as one gigantic graph to support the batch input
    """
    _edge_index = deepcopy(edge_index)
    for batch_index in range(1, batch_size):  # We concatenate a sub-graph to the base edge_index!
        edge_index = np.hstack([edge_index, _edge_index + (num_nodes * batch_index)])
    return edge_index  # 2 x (batch_size * num_edges)


class GATConv(MessagePassing):
    _alpha: OptTensor

    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int, heads: int = 1, concat: bool = True,
                 negative_slope: float = 0.2, dropout: float = 0.0, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin = Linear(in_channels, heads * out_channels, False)
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False)
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False)

        # The learnable parameters to compute attention coefficients:
        self.att_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_dst = Parameter(torch.Tensor(1, heads, out_channels))

        if bias and concat:
            self.bias = Parameter(torch.Tensor(heads * out_channels))
        elif bias and not concat:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        if hasattr(self, 'lin'):
            glorot(self.lin.weight)
        else:
            glorot(self.lin_src.weight)
            glorot(self.lin_dst.weight)
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None,
                return_attention_weights=None):
        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        assert x.dim() == 2, "Static graphs not supported in 'GATConv'"

        # (batch_size * num_nodes) x num_heads x dim_hidden
        x_src = x_dst = self.lin(x).view(-1, self.heads, self.out_channels)
        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)  # (batch_size * num_nodes) x num_heads
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)
        alpha = (alpha_src, alpha_dst)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, x=x, alpha=alpha, size=size)

        alpha = self._alpha
        assert alpha is not None
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias
            # out += 1e-6 * torch.rand_like(out)  # Ref: https://discuss.pytorch.org/t/how-to-fix-this-nan-bug/90291/6

        if isinstance(return_attention_weights, bool):
            return out, alpha
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], size_info: dict = None) -> Tensor:
        # Given edge-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i  # (batch_size * (num_nodes**2)) x num_heads

        alpha = F.leaky_relu(alpha, self.negative_slope)  # (batch_size * (num_nodes**2)) x num_heads
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        out = x_j * alpha.unsqueeze(-1)
        return out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


def test_create_edge_index_for_fully_connected_graph():
    batch_size = 5
    num_nodes = 10
    base_edge_index = np.asarray([[i, j] for i in range(num_nodes) for j in range(num_nodes)]).T
    res = create_edge_index_for_fully_connected_graph(batch_size=batch_size,
                                                      num_nodes=num_nodes,
                                                      base_edge_index=base_edge_index)
    print(res)


if __name__ == '__main__':
    test_create_edge_index_for_fully_connected_graph()
