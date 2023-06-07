import gc
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from types_ import *


class VectorQuantizer(nn.Module):
    def __init__(
        self, 
        num_embeddings: int,
        embed_dim: int,
        commitment_loss_weight: float = 0.25,
        return_indices: bool = False
    ) -> None:
        super().__init__()

        self.embed_dim = embed_dim
        self.num_embeddings = num_embeddings
        self.commitment_loss_weight = commitment_loss_weight

        self.embedding = nn.Embedding(num_embeddings, embed_dim)
        # Uniform distributed in positive & negative
        self.embedding.weight.data.uniform_(-1.0 / num_embeddings, 1.0 / num_embeddings)

        self.return_indices = return_indices
    
    def get_embed_indices(self, flat_latents: Tensor):
        # (bs*h*w, num_embeddings)
        dist = (flat_latents ** 2).sum(dim=-1, keepdim=True) \
            - 2 * torch.matmul(flat_latents, self.embedding.weight.T) \
                + (self.embedding.weight ** 2).sum(dim=-1)
        # (bs*h*w,)
        embed_indices = dist.argmin(dim=-1)

        return embed_indices

    def quantize(self, embed_indices: Tensor):
        return self.embedding(embed_indices)
    
    def forward(self, latents: Tensor):
        embed_dim, h, w = latents.shape[1:]
        if embed_dim != self.embed_dim:
            raise RuntimeError(f"Latent dims: {embed_dim} mismatch embedding dim: {self.embed_dim}.")
        
        # (bs, embed_dim, h, w) -> (bs, h, w, embed_dim) -> (bs*h*w, embed_dim)
        latents = latents.permute(0, 2, 3, 1).reshape(-1, embed_dim)
        # (bs*h*w,)
        embed_indices = self.get_embed_indices(latents)
        # (bs*h*w, embed_dim)
        quantized_latents = self.quantize(embed_indices)

        if self.training:
            embed_loss = F.mse_loss(quantized_latents, latents.detach())
            commitment_loss = F.mse_loss(latents, quantized_latents.detach())
            vq_loss = embed_loss + self.commitment_loss_weight * commitment_loss

        # Copy the gradient of quantized latents straight forward to latents
        quantized_latents = (quantized_latents - latents).detach() + latents
        # (bs*h*w, embed_dim) -> (bs, h, w, embed_dim) -> (bs, embed_dim, h, w)
        quantized_latents = quantized_latents.reshape(-1, h, w, embed_dim).permute(0, 3, 1, 2)

        outputs = (quantized_latents,)
        if self.return_indices:
            # (bs*h*w,) -> (bs, h, w)
            embed_indices = embed_indices.reshape(-1, h, w)
            outputs += (embed_indices,)
        if self.training:
            outputs += (vq_loss,)
        
        return outputs


class Residual(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 1, bias=False)
        )
    
    def forward(self, x: Tensor):
        return x + self.block(x)


class Encoder(nn.Module):
    def __init__(
        self,
        in_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        embed_dim: int,
        num_residual_layers: int = 6
    ) -> None:
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [128, 256]
        
        down_sample = []
        for dim in hidden_dims:
            down_sample.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, dim, 4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            in_dim = dim
        self.down_sample = nn.Sequential(*down_sample)

        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, in_dim, 3, padding=1),
            nn.LeakyReLU()
        )

        residual_layers = []
        for _ in range(num_residual_layers):
            residual_layers.append(Residual(in_dim, in_dim))
        residual_layers.append(nn.LeakyReLU())
        self.residual_layers = nn.Sequential(*residual_layers)

        self.conv_embed = nn.Sequential(
            nn.Conv2d(in_dim, embed_dim, 1),
            nn.LeakyReLU()
        )
    
    def forward(self, x: Tensor):
        x = self.down_sample(x)
        x = self.conv(x)
        x = self.residual_layers(x)

        return self.conv_embed(x)


class Decoder(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        out_dim: int,
        num_residual_layers: int = 6,
        out_activation: nn.Module = nn.Tanh()
    ) -> None:
        super().__init__()

        self.embed_to_hidden = nn.Sequential(
            nn.Conv2d(embed_dim, hidden_dims[0], 3, padding=1),
            nn.LeakyReLU()
        )

        residual_layers = []
        for _ in range(num_residual_layers):
            residual_layers.append(Residual(hidden_dims[0], hidden_dims[0]))
        residual_layers.append(nn.LeakyReLU())
        self.residual_layers = nn.Sequential(*residual_layers)

        up_sample = []
        in_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            up_sample.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_dim, dim, 4, stride=2, padding=1),
                    nn.LeakyReLU()
                )
            )
            in_dim = dim
        up_sample.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_dim, out_dim, 4, stride=2, padding=1),
                out_activation
            )
        )
        self.up_sample = nn.Sequential(*up_sample)

    def forward(self, z: Tensor):
        z = self.embed_to_hidden(z)
        z = self.residual_layers(z)

        return self.up_sample(z)


class VQVAE(nn.Module):
    def __init__(
        self,
        img_size: Union[List[int], Tuple[int]],
        embed_dim: int,
        num_embeddings: int,
        hidden_dims: Union[List[int], Tuple[int]] = None,
        in_channels: int = 3,
        num_residual_layers: int = 6,
        decode_activation: nn.Module = nn.Tanh(),
        vq_commitment_loss_weight: float = 0.25,
        return_embed_indices: bool = False
    ) -> None:
        super().__init__()

        if isinstance(img_size, int):
            img_size = [img_size] * 2
        self.img_size = img_size

        if hidden_dims is None:
            hidden_dims = [128, 256]
        hidden_dims = list(hidden_dims)

        # Encoder
        self.encoder = Encoder(
            in_channels,
            hidden_dims, embed_dim,
            num_residual_layers=num_residual_layers
        )

        self.vq_layer = VectorQuantizer(
            num_embeddings, embed_dim,
            commitment_loss_weight=vq_commitment_loss_weight,
            return_indices=return_embed_indices
        )

        hidden_dims.reverse()
        # Decoder
        self.decoder = Decoder(
            embed_dim, hidden_dims, in_channels,
            num_residual_layers=num_residual_layers,
            out_activation=decode_activation
        )
    
    def forward(self, x: Tensor):
        encoded = self.encoder(x)

        vq_outputs = self.vq_layer(encoded)
        if self.vq_layer.training:
            vq_outputs, vq_loss = vq_outputs[:-1], vq_outputs[-1]
        if self.vq_layer.return_indices:
            embed_indices = vq_outputs[1]
        quantized_latents = vq_outputs[0]

        reconstructed = self.decoder(quantized_latents)

        outputs = (reconstructed,)        
        if self.vq_layer.return_indices:
            outputs += (embed_indices,)
        if self.vq_layer.training:
            outputs += (vq_loss,)
        
        return outputs

    def compute_loss(self, outputs: Tensor, original: Tensor, reconstruct_loss_type: str = "mse", return_dict: bool = True):
        reconstructed = outputs[0]
        if reconstruct_loss_type == "mse":
            reconstruct_loss = F.mse_loss(reconstructed, original)
        elif reconstruct_loss_type == "bce":
            reconstruct_loss_type = F.binary_cross_entropy(reconstructed, original, reduction="none").sum((1, 2, 3)).mean()
        else:
            raise NotImplementedError(f"Current only support 'MSE' or 'BCE' for 'reconstruct_loss_type', but got: {reconstruct_loss_type}.")

        vq_loss = torch.tensor(0., device=reconstruct_loss.device)
        if self.vq_layer.training:
            vq_loss = outputs[-1]
        
        loss = reconstruct_loss + vq_loss
        if return_dict:
            return {
                "loss": loss,
                "reconstruct_loss": reconstruct_loss.detach(),
                "vq_loss": vq_loss.detach()
            }
        else:
            return loss, reconstruct_loss.detach(), vq_loss.detach()

    def reconstruct(self, x: Tensor):
        return self(x)[0]

    def generate(self, embed: Tensor):
        # TODO: implement below
        pass


def parse_args():
    parser = argparse.ArgumentParser(description="VQ-VAE training")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only debug model implementation."
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        dev = torch.device(6)

        img_size = 64
        embed_dim = 128
        num_embeddings = 256

        img_ts = torch.randn(2, 3, img_size, img_size, device=dev)
        
        model = VQVAE(img_size, embed_dim, num_embeddings)
        model.train()
        model.to(dev)

        outputs = model(img_ts)
        loss_dict = model.compute_loss(outputs, img_ts)
        print(loss_dict)

        loss = loss_dict["loss"]
        loss.backward()

        del loss, loss_dict, model, img_ts
        gc.collect()
        torch.cuda.empty_cache()
    else:
        pass

    print("System End.")
