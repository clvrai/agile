import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import Namespace


class DeepSet(torch.nn.Module):
    """ Same architecture as GAT2 """

    def __init__(self, dim_in, dim_hidden, dim_out, args=None, **kwargs):
        super().__init__()
        self.node_feat = None

        args = Namespace(**args) if type(args) == dict else args
        self.args = args

        self.pre_mean = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden))

        self.post_mean = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_out))

    def forward(self, input, adj=None):
        """
        Args:
            input (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x dim_out
        """
        x = self.pre_mean(input)
        x = torch.mean(x, dim=1)
        x = self.post_mean(x)
        return x


class BiLSTM(torch.nn.Module):
    """ Same architecture as GAT2 """

    def __init__(self, dim_in, dim_hidden, dim_out, args=None, **kwargs):
        super().__init__()
        self.node_feat = None
        self.dim_hidden = dim_hidden

        args = Namespace(**args) if type(args) == dict else args
        self.args = args

        self.pre_lstm = nn.Sequential(
            nn.Linear(dim_in, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_hidden))

        self.lstm = nn.LSTM(dim_hidden, dim_hidden, num_layers=args.lstm_summarizer_num_layers, batch_first=True,
                            bidirectional=True)

        self.post_lstm = nn.Sequential(
            nn.Linear(dim_hidden, dim_hidden), nn.ReLU(),
            nn.Linear(dim_hidden, dim_out))

    def forward(self, input, adj=None):
        """
        Args:
            input (torch.tensor): batch_size x num_nodes x dim_node
            adj (torch.tensor): batch_size x num_nodes x num_nodes

        Returns:
            out (torch.tensor): batch_size x dim_out
        """
        x = self.pre_lstm(input)
        x, _ = self.lstm(x)
        x = x.view(x.shape[0], -1, 2, self.dim_hidden)
        x = torch.mean(x, dim=[1, 2])
        x = self.post_lstm(x)
        return x
