""" GraphNorm
- Repo: https://github.com/lsj2408/GraphNorm/blob/master/GraphNorm_ws/gnn_ws/gnn_example/model/Norm/norm.py
- Paper: https://arxiv.org/pdf/2009.03294.pdf
"""

import torch
import torch.nn as nn


class Norm(nn.Module):

    def __init__(self, norm_type: str = "gn", hidden_dim: int = 32, if_gnn: bool = True):
        super(Norm, self).__init__()
        assert norm_type in ["bn", "gn", "ln", "None"]
        self._norm_type = norm_type
        self._if_gnn = if_gnn
        if self._norm_type == "bn":
            self.norm = nn.BatchNorm1d(num_features=hidden_dim)
        elif self._norm_type == "gn":
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))
        elif self._norm_type == "ln":
            self.norm = nn.LayerNorm(normalized_shape=hidden_dim)
        else:
            self.norm = None

    def forward(self, x: torch.tensor):
        """ GraphNorm for Fully connected Homogeneous Graph;

        References:
            See Eq(8) of the paper!

        Args:
            x (torch.tensor): batch_size x num_nodes x dim_node

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_node
        """
        if self._norm_type == "bn":
            if self._if_gnn:
                # ref: See the shape of input; https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html
                x = x.permute(0, 2, 1)  # batch_size x num_nodes x dim_node -> batch_size x dim_node x num_nodes
                out = self.norm(x)
                out = out.permute(0, 2, 1)  # batch_size x dim_node x num_nodes -> batch_size x num_nodes x dim_node
            else:
                out = self.norm(x)
            return out
        elif self._norm_type == "ln":
            out = self.norm(x)
            return out
        elif self.norm is None:
            return x

        batch_size, num_nodes, dim_node = x.shape

        mean = torch.mean(x, dim=1)  # mean over nodes in each graph; batch_size x dim_node
        mean = mean.unsqueeze(1).repeat(1, num_nodes, 1)  # batch_size x num_nodes x dim_node
        sub = x - (mean * self.mean_scale)  # batch_size x num_nodes x dim_node
        std = torch.sqrt(torch.sum(sub.pow(2), dim=1))  # std over nodes in each graph; batch_size x dim_node
        std = std.unsqueeze(1).repeat(1, num_nodes, 1)  # batch_size x num_nodes x dim_node
        # std += 1e-6 * torch.rand_like(std)  # Ref: https://discuss.pytorch.org/t/how-to-fix-this-nan-bug/90291/6
        out = (self.weight * (sub / std)) + self.bias  # batch_size x num_nodes x dim_node
        return out


def _test():
    print("=== test ===")
    batch_size, num_nodes, dim_node = 16, 20, 32
    x = torch.randn(batch_size, num_nodes, dim_node)
    norm = Norm(norm_type="gn", hidden_dim=dim_node)
    x = norm(x)
    print(x.shape)
    print(x.max(), x.min())
    norm = Norm(norm_type="bn", hidden_dim=dim_node)
    x = norm(x)
    print(x.shape)
    print(x.max(), x.min())
    norm = Norm(norm_type="ln", hidden_dim=dim_node)
    x = norm(x)
    print(x.shape)
    print(x.max(), x.min())


if __name__ == '__main__':
    _test()
