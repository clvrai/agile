""" === torch_geometric.typing === """
from typing import Tuple, Optional, Union
from torch import Tensor
from gnn.torch_geometric_utils.torch_sparse_tensor import SparseTensor

Adj = Union[Tensor, SparseTensor]
OptTensor = Optional[Tensor]
PairTensor = Tuple[Tensor, Tensor]
OptPairTensor = Tuple[Tensor, Optional[Tensor]]
PairOptTensor = Tuple[Optional[Tensor], Optional[Tensor]]
Size = Optional[Tuple[int, int]]
NoneType = Optional[Tensor]
