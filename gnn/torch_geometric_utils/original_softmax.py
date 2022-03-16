import torch
from typing import Optional
from torch import Tensor

from gnn.torch_geometric_utils.torch_sparse_tensor import SparseTensor
from gnn.torch_geometric_utils.torch_scatter_scatter import scatter
# from torch_scatter.scatter import scatter


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (Tensor, Optional[int]) -> int
    pass


@torch.jit._overload
def maybe_num_nodes(edge_index, num_nodes=None):
    # type: (SparseTensor, Optional[int]) -> int
    pass


def maybe_num_nodes(edge_index, num_nodes=None):
    if num_nodes is not None:
        return num_nodes
    elif isinstance(edge_index, Tensor):
        return int(edge_index.max()) + 1
    else:
        return max(edge_index.size(0), edge_index.size(1))


# @torch.jit.script
# def softmax(src: Tensor, index: Optional[Tensor] = None,
#             ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
#             dim: int = 0) -> Tensor:
#     if ptr is not None:
#         dim = dim + src.dim() if dim < 0 else dim
#         size = ([1] * dim) + [-1]
#         ptr = ptr.view(size)
#         src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
#         out = (src - src_max).exp()
#         out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
#     elif index is not None:
#         N = maybe_num_nodes(index, num_nodes)
#         src_max = scatter(src, index, dim, dim_size=N, reduce='max')  # num_nodes x num_heads
#         src_max = src_max.index_select(dim, index)  # num_edges x num_heads
#         out = (src - src_max).exp()  # num_edges x num_heads
#         out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')  # num_nodes x num_heads
#         out_sum = out_sum.index_select(dim, index)  # num_edges x num_heads
#     else:
#         raise NotImplementedError
#
#     return out / (out_sum + 1e-16)  # num_edges x num_heads


def softmax2(src: Tensor, index: Optional[Tensor] = None,
             ptr: Optional[Tensor] = None,
             num_nodes: Optional[int] = None,
             dim: int = 0,
             size_info: dict = None) -> Tensor:
    """ We assume the fully connected graph so that num_edges = (batch_size * (num_nodes ** 2))

    Args:
        src (torch.tensor): num_edges x num_heads  ** src is just `alpha` from message function
        index (torch.tensor): 2 x num_edges
        num_nodes (int): num of unique nodes in a batch: (batch_size * num_nodes)

    Returns:

    """
    N = maybe_num_nodes(index, num_nodes)

    # Original lines
    # src_max = scatter(src, index, dim, dim_size=N, reduce='max')  # num_nodes x num_heads
    # src_max = src_max.index_select(dim, index)  # num_edges x num_heads

    # Attempt 1: Subtract the mean instead of max
    # src_max = scatter(src, index, dim, dim_size=N, reduce='mean')  # num_nodes x num_heads
    # print(src_max[:3])
    # src_max = src_max.index_select(dim, index)  # num_edges x num_heads

    _src = src.reshape(size_info["num_nodes"], size_info["num_nodes"], size_info["batch_size"], size_info["num_heads"])
    _src_max = torch.max(_src, dim=2)[0]  # num_nodes x batch_size x num_heads
    _src_max = _src_max.unsqueeze(2).repeat(1, 1, size_info["batch_size"], 1).reshape(
        size_info["batch_size"] * size_info["num_nodes"] ** 2, size_info["num_heads"]
    )
    # print(src_max.shape, _src_max.shape)
    # print(src_max[:10])
    # print(_src_max[:10])
    # assert src_max.sum().item() == _src_max.sum().item(), f"{src_max.sum().item()}, {_src_max.sum().item()}"
    # asdf
    # out = (src - src_max).exp()  # num_edges x num_heads

    out = (src - _src_max).exp()  # num_edges x num_heads
    out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')  # num_nodes x num_heads
    out_sum = out_sum.index_select(dim, index)  # num_edges x num_heads
    return out / (out_sum + 1e-16)  # num_edges x num_heads
