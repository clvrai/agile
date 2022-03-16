import torch
import numpy as np
import torch.nn.functional as F

from copy import deepcopy
from typing import Union, Tuple, Optional
from torch import Tensor
from torch.nn import Parameter

# from torch_geometric.utils import softmax
# from value_based.gnn.torch_geometric_utils.original_softmax import softmax
# from value_based.gnn.torch_geometric_utils.utils import softmax
from gnn.torch_geometric_utils.original_softmax import softmax2 as softmax

from gnn.torch_geometric_utils.utils import remove_self_loops, add_self_loops
from gnn.torch_geometric_utils.message_passing import MessagePassing
from gnn.torch_geometric_utils.torch_geometric_typing import OptPairTensor, Adj, Size, OptTensor
from gnn.torch_geometric_utils.torch_geometric_nn_inits import zeros, glorot
from gnn.torch_geometric_utils.torch_geometric_nn_dense import Linear


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
    # 2 x num_edges(num_items**2)
    edge_index = np.asarray([[i, j] for i in range(num_nodes) for j in range(num_nodes)]).T

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
                 negative_slope: float = 0.2, dropout: float = 0.0,
                 add_self_loops: bool = True, bias: bool = True, **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        # In case we are operating in bipartite graphs, we apply separate
        # transformations 'lin_src' and 'lin_dst' to source and target nodes:
        if isinstance(in_channels, int):
            self.lin_src = Linear(in_channels, heads * out_channels, bias=False, weight_initializer='glorot')
            self.lin_dst = self.lin_src
        else:
            self.lin_src = Linear(in_channels[0], heads * out_channels, False, weight_initializer='glorot')
            self.lin_dst = Linear(in_channels[1], heads * out_channels, False, weight_initializer='glorot')

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
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()
        glorot(self.att_src)
        glorot(self.att_dst)
        zeros(self.bias)

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj,
                size: Size = None, return_attention_weights=None, batch_size: int = None):
        r""" [IMPORTANT] In this method, num_nodes is assumed to be (batch_size x num_nodes) .
                         So, num_nodes represents the num of uniques nodes in the batch of input.
        Args:
            x: num_nodes x dim_node
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C = self.heads, self.out_channels
        num_nodes = int(x.shape[0] // batch_size) if batch_size is not None else 0
        size_info = {"num_heads": H, "dim_out": self.out_channels, "batch_size": batch_size, "num_nodes": num_nodes}

        # We first transform the input node features. If a tuple is passed, we
        # transform source and target node features via separate weights:
        if isinstance(x, Tensor):
            assert x.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = x_dst = self.lin_src(x).view(-1, H, C)  # num_nodes x num_heads x dim_out
        else:  # Tuple of source and target node features:
            x_src, x_dst = x
            assert x_src.dim() == 2, "Static graphs not supported in 'GATConv'"
            x_src = self.lin_src(x_src).view(-1, H, C)
            if x_dst is not None:
                x_dst = self.lin_dst(x_dst).view(-1, H, C)

        x = (x_src, x_dst)

        # Next, we compute node-level attention coefficients, both for source
        # and target nodes (if present):
        alpha_src = (x_src * self.att_src).sum(dim=-1)  # num_nodes x num_heads
        alpha_dst = None if x_dst is None else (x_dst * self.att_dst).sum(-1)  # num_nodes x num_heads
        alpha = (alpha_src, alpha_dst)

        if self.add_self_loops:
            if isinstance(edge_index, Tensor):
                # We only want to add self-loops for nodes that appear both as
                # source and target nodes:
                num_nodes = x_src.size(0)
                if x_dst is not None:
                    num_nodes = min(num_nodes, x_dst.size(0))
                num_nodes = min(size) if size is not None else num_nodes
                edge_index, _ = remove_self_loops(edge_index)
                edge_index, _ = add_self_loops(edge_index, num_nodes=num_nodes)

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        # x: (num_nodes x num_heads x dim_out, num_nodes x num_heads x dim_out)
        # alpha: (num_nodes x num_heads, num_nodes x num_heads)
        out = self.propagate(edge_index=edge_index,
                             x=x,
                             alpha=alpha,
                             size=size,
                             size_info=size_info)  # num_nodes x num_heads x dim_out

        alpha = self._alpha  # num_edges x num_heads
        assert alpha is not None
        self._alpha = None

        if self.concat:
            # num_nodes x num_heads x dim_out -> num_nodes x (num_heads * dim_out)
            out = out.view(-1, self.heads * self.out_channels)
        else:
            out = out.mean(dim=1)

        if self.bias is not None:
            out += self.bias  # num_nodes x (num_heads * dim_out)

        if isinstance(return_attention_weights, bool):
            if isinstance(edge_index, Tensor):
                # return out, (edge_index, alpha)
                return out, alpha
        else:
            return out

    def message(self, x_j: Tensor, alpha_j: Tensor, alpha_i: OptTensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int], size_info: dict) -> Tensor:
        # Given egel-level attention coefficients for source and target nodes,
        # we simply need to sum them up to "emulate" concatenation:
        alpha = alpha_j if alpha_i is None else alpha_j + alpha_i  # num_edges x num_heads

        alpha = F.leaky_relu(alpha, self.negative_slope)  # num_edges x num_heads
        alpha = softmax(alpha, index, ptr, size_i, size_info=size_info)  # num_edges x num_heads
        self._alpha = alpha  # Save for later use.
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)  # num_edges x num_heads
        return x_j * alpha.unsqueeze(-1)  # num_edges x num_heads x dim_out

    def __repr__(self):
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)


def test_create_edge_index_for_fully_connected_graph():
    batch_size = 2
    num_nodes = 3
    base_edge_index = np.asarray([[i, j] for i in range(num_nodes) for j in range(num_nodes)]).T
    res = create_edge_index_for_fully_connected_graph(batch_size=batch_size,
                                                      num_nodes=num_nodes,
                                                      base_edge_index=base_edge_index)
    print(res)


if __name__ == '__main__':
    test_create_edge_index_for_fully_connected_graph()
