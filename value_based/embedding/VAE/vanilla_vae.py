""" https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py """

import torch
from torch import nn
from torch.nn import functional as F

from value_based.commons.init_layer import init_layer
from value_based.embedding.VAE.base import BaseVAE
from value_based.embedding.VAE.types_ import *


class VanillaVAE(BaseVAE):

    def __init__(self, in_channels: int, latent_dim: int, hidden_dims: List = None, **kwargs) -> None:
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        _in_channels = in_channels
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Linear(_in_channels, h_dim).apply(init_layer),
                    nn.BatchNorm1d(h_dim),
                    nn.LeakyReLU()
                )
            )
            _in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim).apply(init_layer)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim).apply(init_layer)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]).apply(init_layer)

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.Linear(hidden_dims[i], hidden_dims[i + 1]).apply(init_layer),
                    nn.BatchNorm1d(hidden_dims[i + 1]),
                    nn.LeakyReLU()
                )
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.Linear(hidden_dims[-1], hidden_dims[-1]).apply(init_layer),
            nn.BatchNorm1d(hidden_dims[-1]),
            nn.LeakyReLU(),
            nn.Linear(hidden_dims[-1], in_channels).apply(init_layer),  # Reconstruct the input data
            nn.Tanh()
        )

    def encode(self, input: Tensor) -> List[Tensor]:
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        result = self.decoder_input(z)
        result = result.view(-1, 512)
        result = self.decoder(result)
        result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, input: Tensor, **kwargs) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        out = self.decode(z)
        return [out, input, mu, log_var]

    def loss_function(self, *args, **kwargs) -> dict:
        recons, input, mu, log_var = args[0], args[1], args[2], args[3]

        kld_weight = kwargs['M_N']  # Account for the minibatch samples from the dataset
        recons_loss = F.mse_loss(recons, input)

        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss.item(), 'Reconstruction_Loss': recons_loss.item(), 'KLD': -kld_loss.item()}

    def sample(self, num_samples: int, current_device: int, **kwargs) -> Tensor:
        z = torch.randn(num_samples, self.latent_dim)
        z = z.to(current_device)
        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor, **kwargs) -> Tensor:
        return self.forward(x)[0]


def test():
    print("=== test ===")
    num_samples, dim_in = 16, 10
    sample = torch.randn(num_samples, dim_in)
    model = VanillaVAE(in_channels=dim_in, latent_dim=32)
    [out, input, mu, log_var] = model(sample)
    print(out.shape, input.shape, mu.shape, log_var.shape)

    loss = model.loss_function(out, input, mu, log_var, M_N=0.005)
    print(loss)


if __name__ == '__main__':
    test()
