""" Test the GATConv with our synthetic data """

import numpy as np
import torch
import torch.nn.functional as F

from gnn.GAT.layers_geometric import GATConv, create_edge_index_for_fully_connected_graph


class GAT(torch.nn.Module):
    def __init__(self, dim_in, dim_out):
        super(GAT, self).__init__()
        self.hid = 8
        self.in_head = 8
        self.out_head = 1

        self.conv1 = GATConv(dim_in, self.hid, heads=self.in_head, dropout=0.6)
        self.conv2 = GATConv(self.hid * self.in_head, dim_out, concat=False, heads=self.out_head, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        out = F.log_softmax(x, dim=1)
        out = out.reshape(batch_size, num_nodes)
        return out


batch_size = 4
num_nodes = 5
dim_in = 10
dim_out = 1
x = np.random.randn(batch_size, num_nodes, dim_in)
x = x.reshape(batch_size * num_nodes, dim_in)  # (batch_size * num_nodes) x dim_in
edge_index = np.asarray([[i, j] for i in range(num_nodes) for j in range(num_nodes)]).T  # 2 x num_edges(num_nodes**2)
edge_index = create_edge_index_for_fully_connected_graph(batch_size=batch_size,
                                                         num_nodes=num_nodes,
                                                         base_edge_index=edge_index)
print(edge_index)
print(edge_index.shape)
device = "cpu"

model = GAT(dim_in=dim_in, dim_out=dim_out).to(device)

# Into torch tensor
x = torch.tensor(x, device=device).type(torch.float32)
edge_index = torch.tensor(edge_index, device=device)
print(x.dtype, x.shape)

out = model(x, edge_index)  # batch_size x num_items
print(out.shape)
