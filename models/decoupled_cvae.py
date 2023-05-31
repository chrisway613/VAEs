import torch
import torch.nn as nn
import torch.nn.functional as F

from types_ import *
from decoupled_vae import Encoder, Decoder


class ConditionalVAE(nn.Module):
    """ The code implementation of Conditional VAE. """
    def __init__(
        self,
        img_size: Union[List[int], Tuple[int]],
        latent_dim: int,
        condition_dim: int,
        hidden_dims: Union[List[int], Tuple[int]] = None,
        in_channels: int = 3,
        decode_activation: nn.Module = nn.Tanh(),
        return_mu_log_var: bool = True
    ) -> None:
        super().__init__()

        if isinstance(img_size, int):
            img_size = [img_size] * 2
        self.img_size = img_size

        self.latent_dim = latent_dim
        self.condition_dim = condition_dim
        self.return_mu_log_var = return_mu_log_var

        self.cond_embed = nn.Linear(condition_dim, img_size[0] * img_size[1])
        self.input_embed = nn.Conv2d(in_channels, in_channels, 1)

        if not hidden_dims:
            hidden_dims = [32, 64, 128, 256, 512]
        hidden_dims = list(hidden_dims)

        # Encoder
        # add one input channel for the extra label channel
        self.encoder = Encoder(in_channels + 1, hidden_dims)

        ''' Map to gaussian distribution '''

        h, w = img_size
        feat_h, feat_w = h // (2 ** len(hidden_dims)), w // (2 ** len(hidden_dims))
        flatten_dim = hidden_dims[-1] * feat_h * feat_w

        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(flatten_dim, latent_dim)

        ''' Decoder '''

        hidden_dims.reverse()
        feat_size = (feat_h, feat_w)
        # Sum latent dim and condition dim as the latent dim for Decoder
        # cuz CVAE concat latent code and condition variables, then feed it to the Decoder.
        self.decoder = Decoder(feat_size, latent_dim + condition_dim, hidden_dims, in_channels, activation=decode_activation)
    
    def _reparam(self, mu: Tensor, log_var: Tensor):
        """Reparameterization trick to sample from N(mu, var) from N(0, 1).

        Args:
            mu (Tensor): Mean value of the latent gaussian distribution.
            log_var (Tensor): Log of variance of the latent gaussian distribution.

        Returns:
            Tensor: tensors sampled from N(mu, var).
        """
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        z = mu + std * eps

        return z

    def _map_to_gaussian(self, encoded: Tensor):
        """Map to latent gaussian distribution.

        Args:
            encoded (Tensor): The encoded tensors, outputs from encoder.
        Returns:
            Tuple[Tensor]: mu(Mean value) & log_var(Log value of variance).
        """
        bs = encoded.size(0)
        flattened = encoded.reshape(bs, -1)

        mu = self.fc_mu(flattened)
        log_var = self.fc_log_var(flattened)

        return mu, log_var

    def _conditioned_input(self, x: Tensor, cond: Tensor):
        embedded_cond = self.cond_embed(cond)
        embedded_cond = embedded_cond.reshape(-1, self.img_size[0], self.img_size[1]).unsqueeze(1)
        embedded_x = self.input_embed(x)
        
        return torch.cat([embedded_x, embedded_cond], dim=1)

    def _conditioned_latent(self, z: Tensor, cond: Tensor):\
        return torch.cat([z, cond], dim=1)
    
    def forward(self, x: Tensor, cond: Tensor):
        cond = cond.float()
        x = self._conditioned_input(x, cond)

        encoded = self.encoder(x)
        mu, log_var = self._map_to_gaussian(encoded)
        z = self._reparam(mu, log_var)
        z = self._conditioned_latent(z, cond)
        decoded = self.decoder(z)

        outputs = (decoded,)
        if self.return_mu_log_var:
            outputs += (mu, log_var)
        return outputs

    def generate(self, cond: Tensor = None, num_samples: int = 1):
        """Samples from the latent space and return the corresponding image space map with specified condition.

        Args:
            cond: (int): Condition variables.
            num_samples (int): Number of required samples.

        Returns:
            Tensor: Decoded tensors in the image space.
        """

        dev = self.decoder.fc.weight.device

        if cond is None:
            cond = torch.zeros(num_samples, dtype=torch.long)
            cond = F.one_hot(cond, num_classes=self.condition_dim)
        cond = cond.to(dev)

        z = torch.randn(num_samples, self.latent_dim, device=dev)
        z = self._conditioned_latent(z)
        
        return self.decoder(z)

    def reconstruct(self, x: Tensor):
        """Given input images(tensor), returns the reconstructed ones.

        Args:
            x (Tensor): Input tensors that come from images.

        Returns:
            Tensor: Reconstructed images.
        """
        return self(x)[0]
