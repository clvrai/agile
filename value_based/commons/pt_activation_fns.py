""" Used to define a pytorch based neural network """
from torch import nn

ACTIVATION_FN = {
    1: nn.ELU,
    2: nn.Hardshrink,
    4: nn.Hardtanh,
    6: nn.LeakyReLU,
    7: nn.LogSigmoid,
    8: nn.MultiheadAttention,
    9: nn.PReLU,
    10: nn.ReLU,
    11: nn.ReLU6,
    12: nn.RReLU,
    13: nn.SELU,
    14: nn.CELU,
    16: nn.Sigmoid,
    18: nn.Softplus,
    19: nn.Softshrink,
    20: nn.Softsign,
    21: nn.Tanh,
    22: nn.Tanhshrink,
    23: nn.Threshold,
    24: nn.Softmin,
    25: nn.Softmax,
    26: nn.Softmax2d,
    27: nn.LogSoftmax,
    28: nn.AdaptiveLogSoftmaxWithLoss,
    29: nn.Identity
}
