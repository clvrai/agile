import torch
from torch import nn
from value_based.policy.architecture.networks import NeuralNet
from value_based.commons.init_layer import init_layer


class RNNFamily(nn.Module):
    def __init__(self,
                 dim_in=28,
                 dim_hidden=128,
                 mlp_dim_hidden="256_32",
                 dim_out=1,
                 batch_first=False,
                 num_layers=1,
                 dropout_rate=0.0,  # 0.25,
                 model_type="gru",
                 device="cpu",
                 **kwargs):
        super(RNNFamily, self).__init__()
        self._dim_in = dim_in
        self._dim_out = dim_out
        self._dim_hidden = dim_hidden
        self._num_layers = num_layers
        self._device = device
        self._model_type = model_type
        self._batch_first = batch_first

        if self._model_type == "lstm":
            self.seq_model = nn.LSTM(dim_in, dim_hidden, num_layers, dropout=dropout_rate, batch_first=batch_first)
        elif self._model_type == "gru":
            self.seq_model = nn.GRU(dim_in, dim_hidden, num_layers, dropout=dropout_rate, batch_first=batch_first)
        else:
            raise ValueError

        # self.seq_model.apply(init_layer)
        self.mlp = NeuralNet(dim_in=dim_hidden, dim_hiddens=mlp_dim_hidden, dim_out=dim_out)

    def forward(self, inputs):
        if self._batch_first:
            batch_size, seq_len, dim_item = inputs.shape
        else:
            seq_len, batch_size, dim_item = inputs.shape

        # Set initial hidden and cell states
        h0 = torch.zeros(self._num_layers, batch_size, self._dim_hidden).to(self._device)
        c0 = torch.zeros(self._num_layers, batch_size, self._dim_hidden).to(self._device)

        # Forward propagate
        # if batch_first: seq_length x batch_size x hidden_size
        # else: batch_size x seq_length x hidden_size
        if self._model_type == "lstm":
            hidden, _ = self.seq_model(inputs, (h0, c0))
        elif self._model_type == "gru":
            hidden, _ = self.seq_model(inputs, h0)
        else:
            raise ValueError

        # Decode the hidden state of the last time step
        if self._batch_first:
            _input = hidden[:, -1, :]
        else:
            _input = hidden[-1, :, :]

        # MLP to produce the score for items in each slate
        out = self.mlp(_input)
        return out


def _test_obs():
    """ test method """
    print("=== _test ===")
    batch_size, dim_in, dim_out = 5, 10, 20
    history_size = 3
    device = "cpu"

    model = RNNFamily(dim_in=dim_in, dim_out=dim_out, batch_first=True, device=device)
    print(model)
    history_seq = torch.randn(batch_size, history_size, dim_in)
    output = model(history_seq)
    print(output.shape)


if __name__ == '__main__':
    _test_obs()
