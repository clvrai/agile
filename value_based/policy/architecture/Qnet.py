import torch
from torch import nn
from value_based.commons.pt_activation_fns import ACTIVATION_FN
from value_based.commons.init_layer import init_layer
from value_based.commons.test import TestCase as test


class QNetwork(nn.Module):
    """ Q-network module: Eq(9) in Cascading DQN """

    def __init__(self, dim_in=28, dim_hiddens="256_32", dim_out=1, type_act_fn=1):
        super(QNetwork, self).__init__()
        # Same architecture as Cascading DQN paper
        dim_hiddens = "{}_".format(dim_in) + dim_hiddens
        self._list = dim_hiddens.split("_")
        for i in range(len(self._list) - 1):
            _in, _out = self._list[i], self._list[i + 1]
            setattr(self, "dense{}".format(i), nn.Linear(int(_in), int(_out)).apply(init_layer))
            setattr(self, "act{}".format(i), ACTIVATION_FN[type_act_fn]())
        self.out = nn.Linear(int(_out), dim_out)

    def forward(self, inputs):
        if type(inputs) == list:
            # When CDQN
            for i, _x in enumerate(inputs):
                if i == 0:
                    x = _x
                else:
                    x = torch.cat([x, _x], dim=-1)
            del _x
        else:
            # When DQN
            x = inputs
            del inputs

        for i in range(len(self._list) - 1):
            x = getattr(self, "dense{}".format(i))(x)
            x = getattr(self, "act{}".format(i))(x)
        x = self.out(x)
        return x.view(x.shape[0], x.shape[1])  # batch_size x num_candidates


class Test(test):
    def __init__(self):
        self._get_args()
        self.args.device = "cuda"
        # self.args.if_debug = True
        self.args.if_debug = False
        self.args.if_visualise_debug = True
        self.args.env_name = "recsim"
        self.args.batch_size = 3
        self._prep()

    def test(self):
        print("=== _test ===")
        dim_in = 10
        x = torch.randn(self.args.batch_size, dim_in, device=self.args.device)
        model = QNetwork(dim_in=dim_in,
                         dim_hiddens=self.args.q_net_dim_hidden).to(self.args.device)
        print(next(model.parameters()).is_cuda)
        out = model(x)
        print(out)


if __name__ == '__main__':
    Test().test()
