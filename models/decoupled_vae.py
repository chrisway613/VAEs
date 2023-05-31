import os
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage

from transformers import get_scheduler

from types_ import *


class Encoder(nn.Module):
    """ Encoder of VAE.

    Args:
        in_dim (int): Input dim, that is usually the number of channels of a image.
        hidden_dims (Union[List[int], Tuple[int]]): Hidden dim of each down-sample stage, each one is an integer.
    """
    def __init__(self, in_dim: int, hidden_dims: Union[List[int], Tuple[int]]) -> None:
        super().__init__()
        
        # Donw-sample stages, each one is Conv->BN->LeakyReLU.
        down_sample = []
        for dim in hidden_dims:
            down_sample.append(
                nn.Sequential(
                    nn.Conv2d(in_dim, dim, 3, stride=2, padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU()
                )
            )
            in_dim = dim
        self.down_sample = nn.Sequential(*down_sample)
    
    def forward(self, x: Tensor):
        """Encoding process.

        Args:
            x (Tensor): Input tensors, usually comes from images.

        Returns:
            Tensor: Encoded tensors(down-sampled).
        """
        return self.down_sample(x)


class Decoder(nn.Module):
    """ Decoder of VAE.

    Args:
        feat_size (Union[List[int], Tuple[int]]): Size(resolution) of feature map.
        latent_dim (int): Dim of latent variables.
        hidden_dims (Union[List[int], Tuple[int]]): Hidden dim of each up-sample stage, each one is an integer.
        out_dim (int): Dim of decoded variables, transformed by decoder.
        activation (nn.Module): Activation layer. Defaults to \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}
    """
    def __init__(
        self,
        feat_size: Union[List[int], Tuple[int]],
        latent_dim: int,
        hidden_dims: Union[List[int], Tuple[int]],
        out_dim: int,
        activation: nn.Module = nn.Tanh()
    ) -> None:
        super().__init__()
    
        self.feat_size = feat_size

        h, w = feat_size
        fc_dim = h * w * hidden_dims[0]
        self.fc = nn.Linear(latent_dim, fc_dim)

        # Up-sample stages
        up_samples = []
        in_dim = hidden_dims[0]
        for dim in hidden_dims[1:]:
            up_samples.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_dim, dim, 3, stride=2, padding=1, output_padding=1),
                    nn.BatchNorm2d(dim),
                    nn.LeakyReLU()
                )
            )
            in_dim = dim
        up_samples.append(
            nn.Sequential(
                nn.ConvTranspose2d(in_dim, in_dim, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(in_dim),
                nn.LeakyReLU()
            )
        )
        self.up_samples = nn.Sequential(*up_samples)

        # Reconstruct(decode) from latent codes. 
        self.reconstruct = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, padding=1),
            activation
        )

    def forward(self, z: Tensor):
        """Decoding process.

        Args:
            z (Tensor): Latent codes.

        Returns:
            Tensor: Reconstructed variables.
        """
        bs = z.size(0)
        h, w = self.feat_size

        z = self.fc(z)
        z = z.reshape(bs, -1, h, w)

        z = self.up_samples(z)
        result = self.reconstruct(z)

        return result


class VAE(nn.Module):
    """The code implementation of Variational Auto-Encoder.

    Args:
        img_size (Union[List[int], Tuple[int]]): Image size(resolution), a list of tuple with form: (height, width).
        latent_dim (int): Dim of latent code.
        hidden_dims (Union[List[int], Tuple[int]]): Hidden dim of each stages.
        in_channels (int): input channels of the input tensors. Defaults to: 3(cuz the inputs are usually RGB images).
        decode_activation (nn.Module): The activation function of decoder. Defaults to: \tanh(x) = \frac{\exp(x) - \exp(-x)} {\exp(x) + \exp(-x)}.
        return_mu_log_var (bool): Whether to return the encoded gaussian distribution parameters. Defaults to: True.
    """
    def __init__(
        self,
        img_size: Union[List[int], Tuple[int]],
        latent_dim: int,
        hidden_dims: Union[List[int], Tuple[int]] = None,
        in_channels = 3,
        decode_activation: nn.Module = nn.Tanh(),
        return_mu_log_var: bool = True
    ) -> None:
        super().__init__()

        if isinstance(img_size, int):
            img_size = [img_size] * 2
        self.img_size = img_size

        if not hidden_dims:
            hidden_dims = [32, 64, 128, 256, 512]
        hidden_dims = list(hidden_dims)

        self.latent_dim = latent_dim
        self.return_mu_log_var = return_mu_log_var

        # Encoder
        self.encoder = Encoder(in_channels, hidden_dims)

        ''' Map to gaussian distribution '''

        h, w = img_size
        feat_h, feat_w = h // (2 ** len(hidden_dims)), w // (2 ** len(hidden_dims))
        flatten_dim = hidden_dims[-1] * feat_h * feat_w

        self.fc_mu = nn.Linear(flatten_dim, latent_dim)
        self.fc_log_var = nn.Linear(flatten_dim, latent_dim)

        ''' Decoder '''

        hidden_dims.reverse()
        feat_size = (feat_h, feat_w)
        self.decoder = Decoder(feat_size, latent_dim, hidden_dims, in_channels, activation=decode_activation)
    
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

    def forward(self, x: Tensor):
        encoded = self.encoder(x)
        mu, log_var = self._map_to_gaussian(encoded)
        z = self._reparam(mu, log_var)
        decoded = self.decoder(z)

        outputs = (decoded,)
        if self.return_mu_log_var:
            outputs += (mu, log_var)
        return outputs

    def generate(self, num_samples: int):
        """Samples from the latent space and return the corresponding image space map.

        Args:
            num_samples (int): Number of required samples.

        Returns:
            Tensor: Decoded tensors in the image space.
        """
        dev = self.decoder.fc.weight.device
        z = torch.randn(num_samples, self.latent_dim, device=dev)
        
        return self.decoder(z)

    def reconstruct(self, x: Tensor):
        """Given input images(tensor), returns the reconstructed ones.

        Args:
            x (Tensor): Input tensors that come from images.

        Returns:
            Tensor: Reconstructed images.
        """
        return self(x)[0]


class MyDataset(Dataset):
    """Custom dataset.

    Args:
        root (str): The root directory of dataset.
        extension (str): The extension name of data file. Defaults to: .jpg.
        transform (torchvision.transforms.Compose): A sequence that defines image transformation.
    """
    def __init__(self, root: str, extension: str = ".jpg", transform: Compose = None) -> None:
        super().__init__()

        self.imgs = [os.path.join(root, img_file) for img_file in os.listdir(root) if img_file.endswith(extension)]
        self.transform = transform

    def __getitem__(self, index) -> Any:
        # Returns a converted copy of the image.
        img = Image.open(self.imgs[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        return img

    def __len__(self) -> int:
        return len(self.imgs)


def vae_loss(
    original: Tensor, reconstructed: Tensor,
    mu: Tensor, log_var: Tensor,
    reconstruct_loss_type: str = "mse",
    kl_weight: float = 1.0,
    return_dict: bool = True
):
    """Loss function for VAE.

    Args:
        original (Tensor): Input tensors for VAE, usually come from images.
        reconstructed (Tensor): Reconstructed tensors, decoded by the decoder of VAE.
        mu (Tensor): Mean value of gaussian distributon.
        log_var (Tensor): Log value of variance of gaussian distribution.
        reconstruct_loss_type (str, optional): Loss function type of reconstruction loss. Defaults to "mse".
        kl_weight (float, optional): Weight of the KL-Divergence loss item. Defaults to 1.0.
        return_dict (bool, optional): _description_. Defaults to True.

    Returns:
        Union[Dict[str, Tensor], Tuple[Tensor]]: Each loss term and their sum.
    """
    if reconstruct_loss_type == "mse":
        reconstruct_loss = F.mse_loss(reconstructed, original)
    elif reconstruct_loss_type == "bce":
        reconstruct_loss_type = F.binary_cross_entropy(reconstructed, original, reduction="none").sum((1, 2, 3)).mean()
    else:
        raise NotImplementedError(f"Current only support 'MSE' or 'BCE' for 'reconstruct_loss_type', but got: {reconstruct_loss_type}.")
    
    # \frac{1}/{2} * (mu^{2} + var - \log var - 1)
    kl_div = (0.5 * (mu ** 2 + log_var.exp() - log_var - 1).sum(1)).mean()
    
    loss = reconstruct_loss + kl_weight * kl_div
    if return_dict:
        return {
            "loss": loss,
            "reconstruct_loss": reconstruct_loss.detach(),
            "kl_div": kl_div.detach()
        }
    else:
        return loss, reconstruct_loss.detach(), kl_div.detach()


def parse_args():
    parser = argparse.ArgumentParser(description="an VAE program.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Whether to debug only."
    )
    parser.add_argument(
        "--train_1pic",
        action="store_true",
        help="Whether to train with only 1 picture."
    )
    args= parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.test:
        dev = torch.device(6)

        img = torch.randn(1, 3, 128, 96, device=dev)
        model = VAE(img.shape[-2:], 32, decode_activation=nn.Sigmoid())
        model.to(dev)

        reconstructed_img, mu, log_var = model(img)
        loss_dict = vae_loss(img, reconstructed_img, mu, log_var)
        print(loss_dict)

        loss = loss_dict['loss']
        loss.backward()

        del img, loss, model
        torch.cuda.empty_cache()
    elif args.train_1pic:
        data_dir = "/home.local/weicai/VAEs/imgs"
        img1 = Image.open(os.path.join(data_dir, "cw4.jpg"))
        print(img1.size, img1.mode, img1.getbands())

        w, h = img1.size
        ratio = w / h
        resized_w, resized_h = int((1024 * ratio) // 32 * 32), 1024
        resized_img1 = img1.resize((resized_w, resized_h))
        print(resized_img1.size, resized_img1.mode)
        resized_img1.save(os.path.join(data_dir, "resized_cw4.jpg"))

        dev = torch.device(6)

        ts = ToTensor()(resized_img1).unsqueeze(0).to(dev)

        model = VAE(ts.shape[-2:], 256)
        model.to(dev)

        total_iters = 500000

        optimizer = AdamW(model.parameters(), lr=2e-3)
        lr_scheduler = LinearLR(optimizer, total_iters=total_iters)

        model.train()
        optimizer.zero_grad(set_to_none=True)

        best_reconstruct = best_kl_div = math.inf

        for i in tqdm(range(total_iters), desc="Training VAE"):
            reconstructed_ts, mu, log_var = model(ts)
            loss_dict = vae_loss(ts, reconstructed_ts, mu, log_var, kl_weight=1.)

            loss = loss_dict['loss']
            loss.backward()

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)

            reconstruct_loss = loss_dict['reconstruct_loss'].item()
            kl_div = loss_dict['kl_div'].item()

            if reconstruct_loss < best_reconstruct:
                best_reconstruct = reconstruct_loss
                ckpt = "best_reconstruct_1pic.pt"
                torch.save(model.state_dict(), ckpt)
            
            if kl_div < best_kl_div:
                best_kl_div = kl_div
                ckpt = "best_kl_div_1pic.pt"
                torch.save(model.state_dict(), ckpt)

            if i % 100 == 0:
                print(
                    f"Iter {i + 1}\t Lr {optimizer.param_groups[0]['lr']}\t"
                    f"Loss {loss.item()}\t"
                    f"Reconstruct Loss {reconstruct_loss}\t"
                    f"KL-Div {kl_div}\n"
                )

            if (i + 1) % 1000 == 0:
                reconstructed_ts = reconstructed_ts.detach().cpu().squeeze(0)
                reconstructed_img = ToPILImage()(reconstructed_ts)

                dst = os.path.join(data_dir, f"reconstruct_cw4_iter{i + 1}.jpg")
                reconstructed_img.save(dst)
                print(f"Reconstruct image has been saved to: {dst}")

                del reconstructed_ts

        torch.cuda.empty_cache()
        print("Training finished.")

        model.eval()

        generate_sd = torch.load("best_kl_div_1pic.pt", map_location=dev)
        model.load_state_dict(generate_sd)

        with torch.no_grad():
            decoded = model.generate(1)
        
        decoded = decoded.cpu().squeeze(0)
        generated_img = ToPILImage()(decoded)

        dst = os.path.join(data_dir, f"generate.jpg")
        generated_img.save(dst)
        print(f"Generated image has been saved to: {dst}")

        torch.cuda.empty_cache()

        reconstruct_sd = torch.load("best_reconstruct1_pic.pt", map_location=dev)
        model.load_state_dict(reconstruct_sd)

        with torch.no_grad():
            decoded = model.reconstruct(ts.detach_())
        
        decoded = decoded.cpu().squeeze(0)
        reconstructed_img = ToPILImage()(decoded)

        dst = os.path.join(data_dir, f"reconstruct.jpg")
        reconstructed_img.save(dst)
        print(f"Reconstructed image has been saved to: {dst}")
    else:
        dataset_dir = "/home.local/weicai/VAEs/dataset"
        dataset = MyDataset(dataset_dir, transform=Compose([ToTensor()]))
        img_size = dataset[0].shape[-2:]
        
        batch_size = 2
        dataloader = DataLoader(dataset, batch_size, pin_memory=True, num_workers=2)

        dev = torch.device(7)

        model = VAE(img_size, 256)
        model.to(dev)

        total_iters = 300000

        optimizer = AdamW(model.parameters(), lr=6e-4)
        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=total_iters
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        ckpt_dir = "checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)

        kl_weight = 1.0
        log_freq, save_freq = 30, 60000
        best_reconstruct = best_kl_div = math.inf

        for i in tqdm(range(total_iters), desc="Training VAE"):
            for batch in dataloader:
                batch = batch.to(dev)
                reconstructed, mu, log_var = model(batch)
                loss_dict = vae_loss(batch, reconstructed, mu, log_var, kl_weight=kl_weight)

                loss = loss_dict['loss']
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)

                reconstruct_loss = loss_dict['reconstruct_loss'].item()
                kl_div = loss_dict['kl_div'].item()

                if reconstruct_loss < best_reconstruct:
                    best_reconstruct = reconstruct_loss
                    ckpt = "best_reconstruct.pt"
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt))
                
                if kl_div < best_kl_div:
                    best_kl_div = kl_div
                    ckpt = "best_kl_div.pt"
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt))

                if i % log_freq == 0:
                    print(
                        f"Iter {i + 1}\t Lr {optimizer.param_groups[0]['lr']}\t"
                        f"Loss {loss.item()}\t"
                        f"Reconstruct Loss {reconstruct_loss}\t"
                        f"KL-Div {kl_div}\n"
                    )

                if (i + 1) % save_freq == 0:
                    reconstructed = reconstructed.detach().cpu()
                    for j, img_ts in enumerate(reconstructed):
                        reconstructed_img = ToPILImage()(img_ts)
                        dst = os.path.join(dataset_dir, f"reconstruct_img_{j + 1}_iter{i + 1}.jpg")
                        reconstructed_img.save(dst)
                        print(f"Reconstructed image{j + 1} has been saved to: {dst}")
                    del reconstructed

        torch.cuda.empty_cache()
        print("Training finished.")

        ckpt = "last.pt"
        torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt))
        print(f"The last checkpoint has been save to: {ckpt}")

        model.eval()

        generate_sd = torch.load(os.path.join(ckpt_dir, "best_kl_div.pt"), map_location=dev)
        model.load_state_dict(generate_sd)

        with torch.no_grad():
            decoded = model.generate(1)
        
        decoded = decoded.cpu().squeeze(0)
        generated_img = ToPILImage()(decoded)

        dst = os.path.join(dataset_dir, f"generate.jpg")
        generated_img.save(dst)
        print(f"Generated image has been saved to: {dst}")

        torch.cuda.empty_cache()

        reconstruct_sd = torch.load(os.path.join(ckpt_dir, "best_reconstruct.pt"), map_location=dev)
        model.load_state_dict(reconstruct_sd)

        for k, ts in enumerate(dataset):
            ts = ts.unsqueeze(0).to(dev)
            with torch.no_grad():
                decoded = model.reconstruct(ts)
        
            decoded = decoded.cpu().squeeze(0)
            reconstructed_img = ToPILImage()(decoded)

            dst = os.path.join(dataset_dir, f"reconstruct_img{k}.jpg")
            reconstructed_img.save(dst)
            print(f"Reconstructed image{k} has been saved to: {dst}")
        
        print("System End.")
