import os
import gc
import math
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from tqdm import tqdm

from torch.optim import AdamW

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, ToPILImage

from transformers import get_scheduler

from types_ import *
from decoupled_vae import Encoder, Decoder, vae_loss


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
            cond: (Tensor): Condition variables.
            num_samples (int): Number of required samples.

        Returns:
            Tensor: Decoded tensors in the image space.
        """

        dev = self.decoder.fc.weight.device

        if cond is None:
            cond = torch.zeros(num_samples, dtype=torch.long)
            cond = F.one_hot(cond, num_classes=self.condition_dim)
        cond = cond.to(dev)

        num_samples = cond.size(0)
        z = torch.randn(num_samples, self.latent_dim, device=dev)
        z = self._conditioned_latent(z, cond)
        
        return self.decoder(z)

    def reconstruct(self, x: Tensor, cond: Tensor):
        """Given input images(tensor), returns the reconstructed ones.

        Args:
            x (Tensor): Input tensors that come from images.
            cond: (Tensor): Condition variables.

        Returns:
            Tensor: Reconstructed images.
        """
        return self(x, cond)[0]


class MyDataset(Dataset):
    """Custom dataset.

    Args:
        root (str): The root directory of dataset.
        num_conds (int): Number of different types of conditions.
        extension (str): The extension name of data file. Defaults to: .jpg.
        transform (torchvision.transforms.Compose): A sequence that defines image transformation.
    """
    def __init__(self, root: str, num_conds, extension: str = ".jpg", transform: Compose = None) -> None:
        super().__init__()

        self.num_conds = num_conds
        self.imgs = [os.path.join(root, img_file) for img_file in os.listdir(root) if img_file.endswith(extension)]
        self.transform = transform

    def __getitem__(self, index) -> Any:
        # Returns a converted copy of the image.
        img = Image.open(self.imgs[index]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        
        cond = F.one_hot(torch.as_tensor(index), num_classes=self.num_conds)

        return img, cond

    def __len__(self) -> int:
        return len(self.imgs)
    

def parse_args():
    parser = argparse.ArgumentParser(description="Everything about CVAE.")

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

        bs = 2
        img_size = 64
        latent_dim = 128
        condition_dim = 2

        imgs = torch.randn(bs, 3, img_size, img_size, device=dev)
        conds = torch.randint(0, condition_dim, (bs,), device=dev)
        conds = F.one_hot(conds, num_classes=condition_dim)

        model = ConditionalVAE(img_size, latent_dim, condition_dim)
        model.to(dev)

        outputs = model(imgs, conds)
        loss_dict = vae_loss(imgs, *outputs)
        print(loss_dict)

        loss = loss_dict["loss"]
        loss.backward()

        del imgs, model, loss_dict, loss
        gc.collect()
        torch.cuda.empty_cache()
    else:
        dataset_dir = "/home.local/weicai/VAEs/cvae_dataset"
        dataset = MyDataset(dataset_dir, 2, transform=Compose([ToTensor()]))
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True, pin_memory=True, num_workers=2)

        dev = torch.device(0)

        latent_dim = 256
        img_size = dataset[0][0].shape[-2:]
        condition_dim = dataset[0][1].size(-1)

        model = ConditionalVAE(img_size, latent_dim, condition_dim)
        model.to(dev)

        total_iters = 500000
        gradient_accumulation_steps = 5

        optimizer = AdamW(model.parameters(), lr=3e-4)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_iters // gradient_accumulation_steps
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        ckpt_dir = "cvae_checkpoints"
        os.makedirs(ckpt_dir)

        output_dir = "cvae_outputs"
        os.makedirs(output_dir)

        kl_weight = 1.0
        log_freq, save_freq = 500, 100000
        best_reconstruct = best_kl_div = math.inf

        for i in tqdm(range(total_iters), desc="Training CVAE"):
            for imgs, conds in dataloader:
                imgs = imgs.to(dev)
                conds = conds.to(dev)
                reconstructed, mu, log_var = model(imgs, conds)
                loss_dict = vae_loss(imgs, reconstructed, mu, log_var, kl_weight=kl_weight)

                loss = loss_dict['loss'] / gradient_accumulation_steps
                loss.backward()

                loss = loss_dict['loss'].item()
                reconstruct_loss = loss_dict['reconstruct_loss'].item()
                kl_div = loss_dict['kl_div'].item()
                
                if i % log_freq == 0:
                    print(
                        f"Iter {i + 1}\t Lr {optimizer.param_groups[0]['lr']}\t"
                        f"Loss {loss}\t"
                        f"Reconstruct Loss {reconstruct_loss}\t"
                        f"KL-Div {kl_div}\n"
                    )

                if reconstruct_loss < best_reconstruct:
                    best_reconstruct = reconstruct_loss
                    ckpt = "cvae_best_reconstruct.pt"
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt))
                
                if kl_div < best_kl_div:
                    best_kl_div = kl_div
                    ckpt = "cvae_best_kl_div.pt"
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt))

                if (i + 1) % save_freq == 0:
                    reconstructed = reconstructed.detach().cpu()
                    for img_ts, one_hot_cond in zip(reconstructed, conds):
                        reconstructed_img = ToPILImage()(img_ts)
                        cond = one_hot_cond.argmax()
                        dst = os.path.join(output_dir, f"cvae_reconstruct_img_cond_{cond + 1}_iter{i + 1}.jpg")
                        reconstructed_img.save(dst)
                        print(f"Reconstructed image with condition {cond + 1} has been saved to: {dst}")
                    del reconstructed
                
                if (i + 1) % gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

        torch.cuda.empty_cache()
        print("Training finished.")

        ckpt = "cvae_last.pt"
        torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt))
        print(f"The last checkpoint has been save to: {ckpt}")

        model.eval()

        num_samples = 2
        # cond = torch.randint(0, condition_dim, (num_samples,))
        cond = torch.tensor([0, 1])
        print(f"Condition for generation: {cond}")
        one_hot_cond = F.one_hot(cond, num_classes=condition_dim)

        with torch.no_grad():
            decoded = model.generate(one_hot_cond, num_samples=num_samples)
        
        for sample_i, sample in enumerate(decoded):
            sample = sample.cpu()
            generated_img = ToPILImage()(sample)

            sample_cond = cond[sample_i].item()
            dst = os.path.join(output_dir, f"final_generated_sample_cond_{sample_cond + 1}.jpg")
            generated_img.save(dst)
            print(f"Generated image with condition {sample_cond + 1} from final checkpoint has been saved to: {dst}")

        torch.cuda.empty_cache()

        generate_sd = torch.load(os.path.join(ckpt_dir, "cvae_best_kl_div.pt"), map_location=dev)
        model.load_state_dict(generate_sd)

        with torch.no_grad():
            decoded = model.generate(one_hot_cond, num_samples=num_samples)
        
        for sample_i, sample in enumerate(decoded):
            sample = sample.cpu()
            generated_img = ToPILImage()(sample)

            sample_cond = cond[sample_i].item()
            dst = os.path.join(output_dir, f"generated_sample_cond_{sample_cond + 1}.jpg")
            generated_img.save(dst)
            print(f"Generated image with condition {sample_cond + 1} from the best generative checkpoint has been saved to: {dst}")

        torch.cuda.empty_cache()

        reconstruct_sd = torch.load(os.path.join(ckpt_dir, "cvae_best_reconstruct.pt"), map_location=dev)
        model.load_state_dict(reconstruct_sd)

        for img, one_hot_cond in dataset:
            img = img.unsqueeze(0).to(dev)
            one_hot_cond = one_hot_cond.unsqueeze(0).to(dev)

            with torch.no_grad():
                decoded = model.reconstruct(img, one_hot_cond)
        
            decoded = decoded.cpu().squeeze(0)
            reconstructed_img = ToPILImage()(decoded)

            reconstruct_cond = one_hot_cond.argmax(dim=-1).item()
            dst = os.path.join(output_dir, f"final_reconstruct_img_cond_{reconstruct_cond + 1}.jpg")
            reconstructed_img.save(dst)
            print(f"Reconstructed image with condition {cond + 1} from the best reconstructed checkpoint has been saved to: {dst}")
        
    print("System End.")
