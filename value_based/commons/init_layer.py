import torch.nn as nn


def init_layer(m):
    """ Initialise the layer's weight and bias """

    if type(m) in [nn.GRU, nn.LSTM, nn.RNN]:
        for name, param in m.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
    else:
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('relu'))
        nn.init.constant_(m.bias.data, 0)
    return m


def _test():
    print("=== _test ===")
    import torch
    import numpy as np
    from copy import deepcopy
    torch.manual_seed(2021)

    for layer in [nn.LSTM, nn.Linear]:
        layer1 = layer(3, 3)
        layer2 = deepcopy(layer1)
        assert np.all(layer1.weight.data.detach().numpy() == layer2.weight.data.detach().numpy())
        layer1.apply(init_layer)
        assert not np.all(layer1.weight.data.detach().numpy() == layer2.weight.data.detach().numpy())


if __name__ == '__main__':
    _test()
