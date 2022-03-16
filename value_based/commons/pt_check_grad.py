""" Ref:
    - https://pytorch.org/tutorials/beginner/examples_nn/two_layer_net_nn.html
    - https://pytorch.org/docs/master/notes/extending.html#extending-torch-autograd
"""
import torch

from value_based.commons.utils import float_round


def check_grad(named_parameters, n_decimal=5):
    """ ref: https://discuss.pytorch.org/t/check-gradient-flow-in-network/15063/10 """
    ave_grad_dict, max_grad_dict = dict(), dict()
    for name, params in named_parameters:
        if params.requires_grad and ("bias" not in name):
            ave_grad_dict[name] = float_round(params.grad.abs().mean().cpu().detach().numpy(), n_decimal)
            max_grad_dict[name] = float_round(params.grad.abs().max().cpu().detach().numpy(), n_decimal)
            # ave_grad_dict[name] = float_round(params.grad.mean().cpu().detach().numpy(), n_decimal)
            # max_grad_dict[name] = float_round(params.grad.max().cpu().detach().numpy(), n_decimal)
    return ave_grad_dict, max_grad_dict


def test_float_round():
    print("=== Test: float_round ===")

    print(float_round(0.12341234, 2))
    print(float_round(0.12341234, 3))
    print(float_round(0.12341234, 4))


def test_check_grad():
    print("=== Test: check_grad ===")
    torch.manual_seed(2021)

    # N is batch size; D_in is input dimension;
    # H is hidden dimension; D_out is output dimension.
    N, D_in, H, D_out = 64, 1000, 100, 10

    # Create random Tensors to hold inputs and outputs
    x = torch.randn(N, D_in)
    y = torch.randn(N, D_out)

    model = torch.nn.Sequential(
        torch.nn.Linear(D_in, H),
        torch.nn.ReLU(),
        torch.nn.Linear(H, D_out),
    )

    opt = torch.optim.Adam(model.parameters())
    loss_fn = torch.nn.MSELoss(reduction='sum')

    for t in range(500):
        # Forward pass
        y_pred = model(x)

        # Compute loss
        loss = loss_fn(y_pred, y)

        # Zero the gradients before running the backward pass.
        opt.zero_grad()

        # Backward pass
        loss.backward()

        # check the gradient: NOTE; don't use this,,, it's useless piece of shit...
        # test = torch.autograd.gradcheck(func=model, inputs=x, eps=1e-6, atol=1e-4)
        # print(test)

        if t % 100 == 99:
            print(t, loss.item())
            check_grad(named_parameters=model.named_parameters())

        # Update the weights using gradient descent
        opt.step()


if __name__ == '__main__':
    test_float_round()
    test_check_grad()
