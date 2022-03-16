# Ref: https://github.com/Diego999/pyGAT/blob/master/layers.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class GraphAttentionLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, dim_in: int, dim_out: int, dropout: float = 0.6, alpha: float = 0.2, concat: bool = True,
                 scale_attention: float = 1.):  # alpha is for LeakyReLU
        super(GraphAttentionLayer, self).__init__()
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._dropout = dropout
        self._alpha = alpha
        self._concat = concat

        self._layer = nn.Linear(in_features=dim_in, out_features=dim_out)
        self.a = nn.Parameter(torch.empty(size=(2 * dim_out, 1)), requires_grad=True)
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self._alpha)
        self._attention = None
        self._scale_attention = scale_attention

    def forward(self, h, adj):
        """
        Args:
            h (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x num_nodes x dim_hidden
        """
        # Get the hidden node repr.
        z = self._layer(h)  # batch_size x num_nodes x dim_hidden

        # === Attention Mechanism
        a_input = self._prepare_attentional_mechanism_input(z)  # batch_size x num_nodes x dim_hidden
        weighted_a_input = torch.matmul(a_input, self.a).squeeze(dim=-1)  # batch_size x num_nodes x num_nodes
        e = self.leakyrelu(weighted_a_input)  # batch_size x num_nodes x num_nodes
        zeros_mat = -9e15 * torch.ones_like(e)  # batch_size x num_nodes x num_nodes
        attention_unnorm = torch.where(adj > 0, e, zeros_mat)  # batch_size x num_nodes x num_nodes
        attention_unnorm *= self._scale_attention
        attention = F.softmax(attention_unnorm, dim=-1)  # batch_size x num_nodes x num_nodes
        self._attention = attention
        attention = F.dropout(attention, self._dropout, training=self.training)  # batch_size x num_nodes x num_nodes

        # Apply the attention to the exchanged node features
        h_prime = torch.matmul(attention, z)  # batch_size x num_nodes x dim_hidden

        # if apply the non-linear activation to the last layer?
        if self._concat:
            out = F.elu(h_prime)  # batch_size x num_nodes x dim_hidden
        else:
            out = h_prime  # batch_size x num_nodes x dim_hidden
        return out

    def _prepare_attentional_mechanism_input(self, z: torch.tensor):
        """ Transform the latent node vectors for the input of Attention Mechanism as follows
            - a_input = concat([z[source], z[target]])

        Args:
            z (torch.tensor): batch_size x num_nodes x dim_hidden

        Returns:
            send_recv_mat (torch.tensor): batch_size x num_nodes x num_nodes x (dim_hidden * 2)
        """
        batch_size, num_nodes, dim_hidden = z.size()
        z_rep = z.repeat_interleave(num_nodes, dim=1)  # batch_size x (num_nodes * num_nodes) x dim_hidden
        z_rep_alternating = z.repeat(1, num_nodes, 1)  # batch_size x (num_nodes * num_nodes) x dim_hidden
        # batch_size x (num_nodes * num_nodes) x (dim_hidden * 2)
        send_recv_mat = torch.cat([z_rep, z_rep_alternating], dim=-1)
        # batch_size x num_nodes x num_nodes x (dim_hidden * 2)
        send_recv_mat = send_recv_mat.view(batch_size, num_nodes, num_nodes, self._dim_out * 2)
        return send_recv_mat

    def get_attention(self):
        """ Given the input, this returns the attention vector
        Returns:
             _weighted_a_input(np.ndarray): batch_size x num_nodes x num_nodes
        """
        return self._attention.detach().cpu().numpy()

    def get_attention_stats(self, args):
        args.gat_attention_std = self._attention.std(-1).mean().detach().cpu().numpy()
        args.gat_attention_max = self._attention.max(-1)[0].mean().detach().cpu().numpy()
        args.gat_attention_min = self._attention.min(-1)[0].mean().detach().cpu().numpy()


class SpecialSpmmFunction(torch.autograd.Function):
    """Special function for only sparse region backpropataion layer."""

    @staticmethod
    def forward(ctx, indices, values, shape, b):
        assert indices.requires_grad == False
        a = torch.sparse_coo_tensor(indices, values, shape)
        ctx.save_for_backward(a, b)
        ctx.N = shape[0]
        return torch.matmul(a, b)

    @staticmethod
    def backward(ctx, grad_output):
        a, b = ctx.saved_tensors
        grad_values = grad_b = None
        if ctx.needs_input_grad[1]:
            grad_a_dense = grad_output.matmul(b.t())
            edge_idx = a._indices()[0, :] * ctx.N + a._indices()[1, :]
            grad_values = grad_a_dense.view(-1)[edge_idx]
        if ctx.needs_input_grad[3]:
            grad_b = a.t().matmul(grad_output)
        return None, grad_values, None, grad_b


class SpecialSpmm(nn.Module):
    def forward(self, indices, values, shape, b):
        return SpecialSpmmFunction.apply(indices, values, shape, b)


class SpGraphAttentionLayer(nn.Module):
    """
    Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, dim_in, dim_out, dropout, alpha, concat=True):
        super(SpGraphAttentionLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.alpha = alpha
        self.concat = concat

        self.W = nn.Parameter(torch.zeros(size=(dim_in, dim_out)))
        nn.init.xavier_normal_(self.W.data, gain=1.414)

        self.a = nn.Parameter(torch.zeros(size=(1, 2 * dim_out)))
        nn.init.xavier_normal_(self.a.data, gain=1.414)

        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.special_spmm = SpecialSpmm()

    def forward(self, input, adj):
        dv = 'cuda' if input.is_cuda else 'cpu'

        N = input.size()[0]
        edge = adj.nonzero().t()

        h = torch.mm(input, self.W)
        # h: N x out
        assert not torch.isnan(h).any()

        # Self-attention on the nodes - Shared attention mechanism
        edge_h = torch.cat((h[edge[0, :], :], h[edge[1, :], :]), dim=1).t()
        # edge: 2*D x E

        edge_e = torch.exp(-self.leakyrelu(self.a.mm(edge_h).squeeze()))
        assert not torch.isnan(edge_e).any()
        # edge_e: E

        e_rowsum = self.special_spmm(edge, edge_e, torch.Size([N, N]), torch.ones(size=(N, 1), device=dv))
        # e_rowsum: N x 1

        edge_e = self.dropout(edge_e)
        # edge_e: E

        h_prime = self.special_spmm(edge, edge_e, torch.Size([N, N]), h)
        assert not torch.isnan(h_prime).any()
        # h_prime: N x out

        h_prime = h_prime.div(e_rowsum)
        # h_prime: N x out
        assert not torch.isnan(h_prime).any()

        if self.concat:
            # if this layer is not last layer,
            return F.elu(h_prime)
        else:
            # if this layer is last layer,
            return h_prime


def test():
    print("=== test ===")
    batch_size = 10
    num_nodes = 7
    dim_in, dim_out = 32, 16

    _input = torch.randn(batch_size, num_nodes, dim_in)
    Adj = torch.ones(num_nodes, num_nodes) - torch.eye(num_nodes)
    Adj = Adj[None, ...].repeat(batch_size, 1, 1)

    layer = GraphAttentionLayer(dim_in=dim_in, dim_out=dim_out)
    out = layer(_input, Adj)
    print(out)


if __name__ == '__main__':
    test()
