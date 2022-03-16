import torch
import itertools


class OptimiserFactory(object):
    def __init__(self, if_single_optimiser=True, if_one_shot_instantiation=False, **optim_args):
        self._if_single_optimiser = if_single_optimiser
        self._if_one_shot_instantiation = if_one_shot_instantiation
        self._optim_args = optim_args

        if self._if_single_optimiser:
            if self._if_one_shot_instantiation:
                self._params = list()
            else:
                self._optimiser = torch.optim.Adam(params=[torch.tensor(0)], **optim_args)  # temp instantiation
        else:
            self._optimisers = list()

    def add_params(self, params_dict: dict):
        """ https://pytorch.org/docs/stable/optim.html#how-to-use-an-optimizer """
        assert "params" in params_dict
        if self._if_single_optimiser:
            if self._if_one_shot_instantiation:
                self._params.append(params_dict["params"])
            else:
                self._optimiser.add_param_group(param_group=params_dict)
        else:
            self._optimisers.append(torch.optim.Adam(**params_dict))

    def get_optimiser(self):
        """ Returns the list of instantiated optimiser(s) """
        if self._if_single_optimiser:
            if self._if_one_shot_instantiation:
                return [torch.optim.Adam(params=itertools.chain(*self._params), **self._optim_args)]
            else:
                # https://discuss.pytorch.org/t/delete-parameter-group-from-optimizer/46814
                del self._optimiser.param_groups[0]  # optim.param_groups = []
                return [self._optimiser]
        else:
            return self._optimisers


def _test():
    from value_based.policy.architecture.network import NeuralNet
    print("=== test ===")
    for if_single_optimiser in [True, False]:
        for if_one_shot_instantiation in [True, False]:
            print("if_single_optimiser: {} if_one_shot_instantiation: {}".format(str(if_single_optimiser),
                                                                                 str(if_one_shot_instantiation)))
            optim_factory = OptimiserFactory(if_single_optimiser=if_single_optimiser,
                                             if_one_shot_instantiation=if_one_shot_instantiation)
            optim_factory.add_params(params_dict={"params": NeuralNet().parameters()})
            optim_factory.add_params(params_dict={"params": NeuralNet().parameters()})
            optim_factory.add_params(params_dict={"params": NeuralNet().parameters()})
            optim_list = optim_factory.get_optimiser()
            assert type(optim_list) == list
            print(optim_list)


if __name__ == '__main__':
    _test()
