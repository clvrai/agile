from typing import Optional, Tuple

import torch


# import value_based.gnn.torch_geometric_utils.torch_scatter_init  # don't delete!


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int):
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand_as(other)
    return src


def scatter_sum(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    raise ValueError
    # return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def _get_count(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
               out: Optional[torch.Tensor] = None,
               dim_size: Optional[int] = None) -> torch.Tensor:
    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count.clamp_(1)
    count = broadcast(count, out, dim)
    return count


def scatter_mean(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                 out: Optional[torch.Tensor] = None,
                 dim_size: Optional[int] = None) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)
    count = _get_count(src, index, dim, out, dim_size)
    if torch.is_floating_point(out):
        # out.true_divide_(count)
        out.div(count)
    else:
        # out.floor_divide_(count)
        out.div(count, rounding_mode='trunc')
    return out


def scatter_std(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
                out: Optional[torch.Tensor] = None,
                dim_size: Optional[int] = None) -> torch.Tensor:
    # Get Mean
    src_mean = scatter_mean(src=src, index=index, out=out, dim=dim, dim_size=dim_size)
    src_mean = src_mean.index_select(dim, index)

    # sum(x - mean)**2
    out = torch.square(src - src_mean)
    out = scatter_sum(src=out, index=index, dim=dim, dim_size=dim_size)
    out = out.index_select(dim, index)

    # (sum(x - mean)**2) / n
    dim_size = out.size(dim)
    count = _get_count(src, index, dim, out, dim_size)
    if torch.is_floating_point(out):
        # out.true_divide_(count)
        out.div(count)
    else:
        # out.floor_divide_(count)
        out.div(count, rounding_mode='trunc')

    out = torch.sqrt(out)
    return out


def scatter_min(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    raise ValueError
    # return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_max(
        src: torch.Tensor, index: torch.Tensor, dim: int = -1,
        out: Optional[torch.Tensor] = None,
        dim_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    raise ValueError
    # return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def scatter(src: torch.Tensor, index: torch.Tensor, dim: int = -1,
            out: Optional[torch.Tensor] = None, dim_size: Optional[int] = None,
            reduce: str = "sum") -> torch.Tensor:
    if reduce == 'sum' or reduce == 'add':
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == 'mul':
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == 'mean':
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == 'std':
        return scatter_std(src, index, dim, out, dim_size)
    elif reduce == 'min':
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == 'max':
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError
