import torch
import numpy as np
from torch import nn
from value_based.commons.init_layer import init_layer
from value_based.embedding.VAE.vanilla_vae import VanillaVAE as VAE


class WideAndDeep(nn.Module):
    """ Wide and Deep; Architecture follows the original implementation of CDQN paper's one
        `./value_based/shared/LR_embedding.py`
    """

    def __init__(self, in_wide_dim: int = 28, in_deep_dim: int = 28, dim_out: int = 1, if_softmax: bool = False):
        super(WideAndDeep, self).__init__()
        self.deep = nn.Sequential(
            nn.Linear(in_deep_dim, 50).apply(init_layer),
            nn.Linear(50, 16).apply(init_layer)
        )
        self.wide = nn.Sequential(
            nn.Linear(in_wide_dim, 500).apply(init_layer),
            nn.Linear(500, 100).apply(init_layer),
            nn.Linear(100, 16).apply(init_layer)
        )
        self.act = nn.ELU()
        self.out = nn.Linear(32, dim_out)
        self._if_softmax = if_softmax

        if self._if_softmax:
            self.act_out = nn.Softmax(dim=-1)

    def forward(self, x, return_embedding: bool = False):
        _in_wide, _in_deep = x["in_wide"], x["in_deep"]
        _deep = self.act(self.deep(_in_deep))
        _wide = self.act(self.wide(_in_wide))
        _item_embedding = torch.cat([_deep, _wide], dim=-1)  # Concatenate the outputs of wide&deep nets
        x = self.out(_item_embedding)
        if self._if_softmax:
            x = self.act_out(x)
        if return_embedding:
            return x, _item_embedding
        else:
            return x


class WideAndDeepVAE(nn.Module):
    """ VAE based Wide and Deep """

    def __init__(self, in_wide_dim: int = 28, in_deep_dim: int = 28):
        super(WideAndDeepVAE, self).__init__()
        self.deep = VAE(in_channels=in_deep_dim, latent_dim=16)
        self.wide = VAE(in_channels=in_wide_dim, latent_dim=16)

    def forward(self, x, return_embedding: bool = False):
        _in_wide, _in_deep = x["in_wide"], x["in_deep"]
        [deep_out, deep_input, deep_mu, deep_log_var] = self.deep(_in_deep)
        [wide_out, wide_input, wide_mu, wide_log_var] = self.wide(_in_wide)
        _item_embedding = torch.cat([deep_mu, wide_mu], dim=-1)  # Concatenate the outputs of wide&deep nets
        out = {"d_out": deep_out, "d_input": deep_input, "d_mu": deep_mu, "d_log_var": deep_log_var,
               "w_out": wide_out, "w_input": wide_input, "w_mu": wide_mu, "w_log_var": wide_log_var}
        if return_embedding:
            return out, _item_embedding.detach().cpu().numpy()
        else:
            return out

    def loss_function(self, x):
        deep_res = self.deep.loss_function(x["d_out"], x["d_input"], x["d_mu"], x["d_log_var"], M_N=0.005)
        wide_res = self.wide.loss_function(x["w_out"], x["w_input"], x["w_mu"], x["w_log_var"], M_N=0.005)
        return deep_res, wide_res


def _test_WideAndDeep():
    DEVICE = "cpu"
    num_category = 3

    sparse_feat = np.random.randn(100, 709)
    dense_feat = np.random.randn(100, 14)
    labels = torch.randint(low=0, high=num_category - 1, size=[100], device=DEVICE)
    print(sparse_feat.shape, dense_feat.shape, labels.shape)

    model = WideAndDeep(in_wide_dim=sparse_feat.shape[1],
                        in_deep_dim=dense_feat.shape[1],
                        dim_out=num_category).to(device=DEVICE)
    opt = torch.optim.Adam(params=model.parameters())
    inputs = {"in_deep": torch.tensor(dense_feat.astype(np.float32), device=DEVICE),
              "in_wide": torch.tensor(sparse_feat.astype(np.float32), device=DEVICE)}
    y_hat, _item_embedding = model(inputs, return_embedding=True)

    opt.zero_grad()
    loss = torch.nn.CrossEntropyLoss()(y_hat, labels)
    loss.backward()
    opt.step()
    print(loss.item(), _item_embedding.shape)


def _test_WideAndDeepVAE():
    DEVICE = "cpu"
    num_category = 3

    sparse_feat = np.random.randn(100, 709)
    dense_feat = np.random.randn(100, 14)
    labels = torch.randint(low=0, high=num_category - 1, size=[100], device=DEVICE)
    print(sparse_feat.shape, dense_feat.shape, labels.shape)

    model = WideAndDeepVAE(in_wide_dim=sparse_feat.shape[1], in_deep_dim=dense_feat.shape[1]).to(device=DEVICE)
    opt = torch.optim.Adam(params=model.parameters())
    inputs = {"in_deep": torch.tensor(dense_feat.astype(np.float32), device=DEVICE),
              "in_wide": torch.tensor(sparse_feat.astype(np.float32), device=DEVICE)}
    out = model(inputs)
    loss = model.loss_function(x=out)
    print(loss)


if __name__ == '__main__':
    # _test_WideAndDeep()
    _test_WideAndDeepVAE()