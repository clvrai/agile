""" Collection of utils copied from torch geometric repo """
import torch


def expand_left(src: torch.Tensor, dim: int, dims: int) -> torch.Tensor:
    for _ in range(dims + dim if dim < 0 else dim):
        src = src.unsqueeze(0)
    return src


import sys
from getpass import getuser
from tempfile import NamedTemporaryFile as TempFile, gettempdir
from importlib.util import module_from_spec, spec_from_file_location

import os
import os.path as osp
import errno


def makedirs(path):
    try:
        os.makedirs(osp.expanduser(osp.normpath(path)))
    except OSError as e:
        if e.errno != errno.EEXIST and osp.isdir(path):
            raise e


def class_from_module_repr(cls_name, module_repr):
    path = osp.join(gettempdir(), f'{getuser()}_pyg_jit')
    makedirs(path)
    with TempFile(mode='w+', suffix='.py', delete=False, dir=path) as f:
        f.write(module_repr)
    spec = spec_from_file_location(cls_name, f.name)
    mod = module_from_spec(spec)
    sys.modules[cls_name] = mod
    spec.loader.exec_module(mod)
    return getattr(mod, cls_name)


import pyparsing as pp
from itertools import product
from typing import Callable, Tuple, Dict, List


def split_types_repr(types_repr: str) -> List[str]:
    out = []
    i = depth = 0
    for j, char in enumerate(types_repr):
        if char == '[':
            depth += 1
        elif char == ']':
            depth -= 1
        elif char == ',' and depth == 0:
            out.append(types_repr[i:j].strip())
            i = j + 1
    out.append(types_repr[i:].strip())
    return out


def sanitize(type_repr: str):
    type_repr = re.sub(r'<class \'(.*)\'>', r'\1', type_repr)
    type_repr = type_repr.replace('typing.', '')
    type_repr = type_repr.replace('torch_sparse.tensor.', '')
    type_repr = type_repr.replace('Adj', 'Union[Tensor, SparseTensor]')

    # Replace `Union[..., NoneType]` by `Optional[...]`.
    sexp = pp.nestedExpr(opener='[', closer=']')
    tree = sexp.parseString(f'[{type_repr.replace(",", " ")}]').asList()[0]

    def union_to_optional_(tree):
        for i in range(len(tree)):
            e, n = tree[i], tree[i + 1] if i + 1 < len(tree) else []
            if e == 'Union' and n[-1] == 'NoneType':
                tree[i] = 'Optional'
                tree[i + 1] = tree[i + 1][:-1]
            elif e == 'Union' and 'NoneType' in n:
                idx = n.index('NoneType')
                n[idx] = [n[idx - 1]]
                n[idx - 1] = 'Optional'
            elif isinstance(e, list):
                tree[i] = union_to_optional_(e)
        return tree

    tree = union_to_optional_(tree)
    type_repr = re.sub(r'\'|\"', '', str(tree)[1:-1]).replace(', [', '[')

    return type_repr


def param_type_repr(param) -> str:
    if param.annotation is inspect.Parameter.empty:
        return 'torch.Tensor'
    return sanitize(re.split(r':|='.strip(), str(param))[1])


def return_type_repr(signature) -> str:
    return_type = signature.return_annotation
    if return_type is inspect.Parameter.empty:
        return 'torch.Tensor'
    elif str(return_type)[:6] != '<class':
        return sanitize(str(return_type))
    elif return_type.__module__ == 'builtins':
        return return_type.__name__
    else:
        return f'{return_type.__module__}.{return_type.__name__}'


def parse_types(func: Callable) -> List[Tuple[Dict[str, str], str]]:
    source = inspect.getsource(func)
    signature = inspect.signature(func)

    # Parse `# type: (...) -> ...` annotation. Note that it is allowed to pass
    # multiple `# type:` annotations in `forward()`.
    iterator = re.finditer(r'#\s*type:\s*\((.*)\)\s*->\s*(.*)\s*\n', source)
    matches = list(iterator)

    if len(matches) > 0:
        out = []
        args = list(signature.parameters.keys())
        for match in matches:
            arg_types_repr, return_type = match.groups()
            arg_types = split_types_repr(arg_types_repr)
            arg_types = OrderedDict((k, v) for k, v in zip(args, arg_types))
            return_type = return_type.split('#')[0].strip()
            out.append((arg_types, return_type))
        return out

    # Alternatively, parse annotations using the inspected signature.
    else:
        ps = signature.parameters
        arg_types = OrderedDict((k, param_type_repr(v)) for k, v in ps.items())
        return [(arg_types, return_type_repr(signature))]


def resolve_types(arg_types: Dict[str, str],
                  return_type_repr: str) -> List[Tuple[List[str], str]]:
    out = []
    for type_repr in arg_types.values():
        if type_repr[:5] == 'Union':
            out.append(split_types_repr(type_repr[6:-1]))
        else:
            out.append([type_repr])
    return [(x, return_type_repr) for x in product(*out)]


import re
import inspect
from collections import OrderedDict
from typing import Dict, List, Any, Optional, Callable, Set


class Inspector(object):
    def __init__(self, base_class: Any):
        self.base_class: Any = base_class
        self.params: Dict[str, Dict[str, Any]] = {}

    def inspect(self, func: Callable,
                pop_first: bool = False) -> Dict[str, Any]:
        params = inspect.signature(func).parameters
        params = OrderedDict(params)
        if pop_first:
            params.popitem(last=False)
        self.params[func.__name__] = params

    def keys(self, func_names: Optional[List[str]] = None) -> Set[str]:
        keys = []
        for func in func_names or list(self.params.keys()):
            keys += self.params[func].keys()
        return set(keys)

    def __implements__(self, cls, func_name: str) -> bool:
        if cls.__name__ == 'MessagePassing':
            return False
        if func_name in cls.__dict__.keys():
            return True
        return any(self.__implements__(c, func_name) for c in cls.__bases__)

    def implements(self, func_name: str) -> bool:
        return self.__implements__(self.base_class.__class__, func_name)

    def types(self, func_names: Optional[List[str]] = None) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for func_name in func_names or list(self.params.keys()):
            func = getattr(self.base_class, func_name)
            arg_types = parse_types(func)[0][0]
            for key in self.params[func_name].keys():
                if key in out and out[key] != arg_types[key]:
                    raise ValueError(
                        (f'Found inconsistent types for argument {key}. '
                         f'Expected type {out[key]} but found type '
                         f'{arg_types[key]}.'))
                out[key] = arg_types[key]
        return out

    def distribute(self, func_name, kwargs: Dict[str, Any]):
        out = {}
        for key, param in self.params[func_name].items():
            if key == "size_info": continue
            data = kwargs.get(key, inspect.Parameter.empty)
            if data is inspect.Parameter.empty:
                if param.default is inspect.Parameter.empty:
                    raise TypeError(f'Required parameter {key} is empty.')
                data = param.default
            out[key] = data
        return out


def func_header_repr(func: Callable, keep_annotation: bool = True) -> str:
    source = inspect.getsource(func)
    signature = inspect.signature(func)

    if keep_annotation:
        return ''.join(re.split(r'(\).*?:.*?\n)', source,
                                maxsplit=1)[:2]).strip()

    params_repr = ['self']
    for param in signature.parameters.values():
        params_repr.append(param.name)
        if param.default is not inspect.Parameter.empty:
            params_repr[-1] += f'={param.default}'

    return f'def {func.__name__}({", ".join(params_repr)}):'


def func_body_repr(func: Callable, keep_annotation: bool = True) -> str:
    source = inspect.getsource(func)
    body_repr = re.split(r'\).*?:.*?\n', source, maxsplit=1)[1]
    if not keep_annotation:
        body_repr = re.sub(r'\s*# type:.*\n', '', body_repr)
    return body_repr


def remove_self_loops(edge_index, edge_attr: Optional[torch.Tensor] = None):
    r"""Removes every self-loop in the graph given by :attr:`edge_index`, so
    that :math:`(i,i) \not\in \mathcal{E}` for every :math:`i \in \mathcal{V}`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_attr (Tensor, optional): Edge weights or multi-dimensional
            edge features. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    mask = edge_index[0] != edge_index[1]
    edge_index = edge_index[:, mask]
    if edge_attr is None:
        return edge_index, None
    else:
        return edge_index, edge_attr[mask]


def add_self_loops(edge_index, edge_weight: Optional[torch.Tensor] = None,
                   fill_value: float = 1., num_nodes: Optional[int] = None):
    r"""Adds a self-loop :math:`(i,i) \in \mathcal{E}` to every node
    :math:`i \in \mathcal{V}` in the graph given by :attr:`edge_index`.
    In case the graph is weighted, self-loops will be added with edge weights
    denoted by :obj:`fill_value`.

    Args:
        edge_index (LongTensor): The edge indices.
        edge_weight (Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): If :obj:`edge_weight` is not :obj:`None`,
            will add self-loops with edge weights of :obj:`fill_value` to the
            graph. (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)

    :rtype: (:class:`LongTensor`, :class:`Tensor`)
    """
    N = maybe_num_nodes(edge_index, num_nodes)

    loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        loop_weight = edge_weight.new_full((N,), fill_value)
        edge_weight = torch.cat([edge_weight, loop_weight], dim=0)

    edge_index = torch.cat([edge_index, loop_index], dim=1)

    return edge_index, edge_weight


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


from typing import Optional

import torch
from torch import Tensor

from gnn.torch_geometric_utils.torch_scatter_scatter import scatter
from gnn.torch_geometric_utils.torch_scatter_segment_csr import segment_csr


def gather_csr(src: torch.Tensor, indptr: torch.Tensor,
               out: Optional[torch.Tensor] = None) -> torch.Tensor:
    return torch.ops.torch_scatter.gather_csr(src, indptr, out)


# @torch.jit.script
def softmax(src: Tensor, index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
            dim: int = 0) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.

    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)

    :rtype: :class:`Tensor`
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)  # (batch_size * (num_nodes**2))

        """ Here I am leaving some attempts to make this softmax work!! """

        # Attempt 1
        # Original one but not working unless you need to pip install torch_scatter library..
        # src_max = scatter(src, index, dim, dim_size=N, reduce='max')

        # This has an error in shape...
        # side1 = N
        # side2 = src.shape[0] // N
        # src2 = src.clone()  # (batch_size * num_nodes) x num_heads
        # src2 = src2.reshape(side1 // side2, side2, side2)  # (batch_size * num_nodes) x num_heads
        # src2_max = src2.max(dim=1)[0]
        # src_max = src2_max.unsqueeze(1).repeat(1, side2, 1)
        # src_max = src_max.reshape(*src.shape)

        # this is at least working w/o any error but not good
        src_max = torch.max(src)

        out = (src - src_max).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce="sum")
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)
