import torch
from torch import nn
from value_based.commons.pt_activation_fns import ACTIVATION_FN
from value_based.commons.init_layer import init_layer
from value_based.commons.test import TestCase as test


class NeuralNet(nn.Module):
    def __init__(self, dim_in=28, dim_hiddens="256_32", dim_out=1, type_hidden_act_fn=1):
        super(NeuralNet, self).__init__()
        dim_hiddens = "{}_".format(dim_in) + dim_hiddens
        self._list = dim_hiddens.split("_")
        for i in range(len(self._list) - 1):
            _in, _out = self._list[i], self._list[i + 1]
            setattr(self, "dense{}".format(i), nn.Linear(int(_in), int(_out)).apply(init_layer))
            setattr(self, "act{}".format(i), ACTIVATION_FN[type_hidden_act_fn]())
        self.out = nn.Linear(int(_out), dim_out)

    def forward(self, x):
        for i in range(len(self._list) - 1):
            x = getattr(self, "dense{}".format(i))(x)
            x = getattr(self, "act{}".format(i))(x)
        x = self.out(x)
        return x


def _test_NeuralNet():
    """ test method """
    print("=== _test_NeuralNet ===")
    model = NeuralNet()
    batch_size, num_items = 28, 28
    x = torch.randn(batch_size, num_items)
    output = model(x)
    print(output.shape)


if __name__ == '__main__':
    _test_NeuralNet()
