import gc
import os
import math
import logging
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import AdamW
from torchvision.transforms import Compose, ToTensor

from transformers.optimization import get_scheduler

from types_ import *

from decoupled_vq_vae import VQVAE
from decoupled_vae import MyDataset


logger = logging.getLogger(__name__)
logger.setLevel('INFO')
logger.addHandler(logging.StreamHandler())


class StateStdout:
    def __init__(self, logger: logging.Logger = None, begin: str = "Start", end: str = "Done") -> None:
        self.logger = logger
        self.begin = begin
        self.end = end
    
    def __enter__(self):
        if self.logger is None:
            print(self.begin)
        else:
            self.logger.info(self.begin)

        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.logger is None:
            print(self.end)
        else:
            self.logger.info(self.end)

        return True


class MaskedConv2d(nn.Conv2d):
    def __init__(self, mask: Tensor, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]], **kwargs) -> None:
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        # (h, w) -> (1, 1, h, w)
        self.register_buffer("mask", mask[None, None])
    
    def forward(self, x: Tensor):
        self.weight.data *= self.mask
        return super().forward(x)


class VerticalStackConv(MaskedConv2d):
    def __init__(self, mask_type: str, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]], **kwargs) -> None:
        if mask_type not in ("A", "B"):
            raise ValueError(f"mask_type must be 'A' or 'B', got: {mask_type}")
        self.mask_type = mask_type

        if isinstance(kernel_size, int):
            kernel_size = (kernel_size,) * 2

        mask = torch.zeros(kernel_size)
        mask[:(kernel_size[0] // 2), :] = 1.
        if mask_type == "B":
            mask[kernel_size[0] // 2, :] = 1.

        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)


class HorizontalStackConv(MaskedConv2d):
    def __init__(self, mask_type: str, in_channels: int, out_channels: int, kernel_size: Union[int, Tuple[int]], **kwargs) -> None:
        if mask_type not in ("A", "B"):
            raise ValueError(f"mask_type must be 'A' or 'B', got: {mask_type}")
        self.mask_type = mask_type

        if isinstance(kernel_size, int):
            kernel_size = (1, kernel_size)
        if kernel_size[0] != 1:
            raise ValueError(f"The first value of kernel size must be 1, got: {kernel_size[0]}")

        padding = kwargs.pop("padding", 0)
        if isinstance(padding, int):
            padding = (0, padding)
        kwargs["padding"] = padding
        
        mask = torch.zeros(kernel_size)
        mask[:, :(kernel_size[1] // 2)] = 1.
        if mask_type == "B":
            mask[:, kernel_size[1] // 2] = 1.

        super().__init__(mask, in_channels, out_channels, kernel_size, **kwargs)


class GatedMaskedConv(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int, dilation: int = 1) -> None:
        super().__init__()

        padding = dilation * (kernel_size - 1) // 2

        self.conv_vertical = VerticalStackConv(
            "B", in_channels, in_channels * 2,
            kernel_size, padding=padding, dilation=dilation
        )
        self.conv_vert_to_horizon = nn.Conv2d(in_channels * 2, in_channels * 2, 1)
        
        self.conv_horizontal = HorizontalStackConv(
            "B", in_channels, in_channels * 2,
            kernel_size, padding=padding, dilation=dilation
        )
        self.conv_horizon_out = nn.Conv2d(in_channels, in_channels, 1)
    
    def forward(self, v_stack: Tensor, h_stack: Tensor):
        v_stack_feat = self.conv_vertical(v_stack)
        v_val, v_gate = v_stack_feat.chunk(2, dim=1)
        v_stack_out = torch.tanh(v_val) * torch.sigmoid(v_gate)

        h_stack_feat = self.conv_horizontal(h_stack) + self.conv_vert_to_horizon(v_stack_feat)
        h_val, h_gate = h_stack_feat.chunk(2, dim=1)
        h_stack_feat = torch.tanh(h_val) * torch.sigmoid(h_gate)
        h_stack_out = self.conv_horizon_out(h_stack_feat) + h_stack

        return v_stack_out, h_stack_out


class GatedPixelCNN(nn.Module):
    def __init__(self, in_channels: int, channels: int, out_channels: int) -> None:
        super().__init__()

        self.conv_vstack = VerticalStackConv("A", in_channels, channels, 3, padding=1)
        self.conv_hstack = HorizontalStackConv("A", in_channels, channels, 3, padding=1)

        self.conv_blocks = nn.ModuleList([
            GatedMaskedConv(channels, 3),
            GatedMaskedConv(channels, 3, dilation=2),
            GatedMaskedConv(channels, 3)
        ])

        self.conv_out = nn.Conv2d(channels, out_channels, 1)
    
    def forward(self, x: Tensor):
        v_stack = self.conv_vstack(x)
        h_stack = self.conv_hstack(x)
        for block in self.conv_blocks:
            v_stack, h_stack = block(v_stack, h_stack)
        
        return self.conv_out(F.elu(h_stack))


def parse_args():
    parser = argparse.ArgumentParser(description="GatedPixelCNN training")

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Only for verify model implementation."
    )
    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        dev = torch.device(7)

        bs = 2
        img_size = 16
        num_embeddings = 256

        # (bs, img_size, img_size)
        embed_indices = torch.randint(0, num_embeddings, (bs, img_size, img_size), device=dev)
        # (bs, img_size, img_size, num_embeddings)
        one_hot_indices = F.one_hot(embed_indices, num_classes=num_embeddings)

        # (bs, img_size, img_size, num_embeddings) -> (bs, num_embeddings, img_size, img_size);
        # long -> float32
        img_ts = one_hot_indices.permute(0, 3, 1, 2).contiguous().float()

        model = GatedPixelCNN(num_embeddings, 128, num_embeddings)
        model.train()
        model.to(dev)

        with StateStdout(logger=logger, begin="Tesing forward & backward..", end="Done!"):
            # (bs, num_embeddings, img_size, img_size)
            logits = model(img_ts)

            loss = F.cross_entropy(logits, embed_indices)
            loss.backward()
            print(f"Loss: {loss}")
        
        del loss, logits, img_ts, one_hot_indices, embed_indices
        gc.collect()
        torch.cuda.empty_cache()

        model.eval()
        priors = torch.zeros(bs, img_size, img_size, dtype=torch.long, device=dev)
        with StateStdout(logger=logger, begin="Testing auto-regressive generation..", end="Done!"):
            for row in range(img_size):
                for col in range(img_size):
                    one_hot_priors = F.one_hot(priors, num_classes=num_embeddings)
                    # torch.long -> torch.float32
                    input_ts = one_hot_priors.permute(0, 3, 1, 2).contiguous().float()
                    with torch.inference_mode():
                        pixel_logits = model(input_ts)[:, :, row, col]

                    # (bs, num_embeddings)
                    pixel_probs = F.softmax(pixel_logits, dim=-1)
                    # (bs, num_embeddings) -> (bs, 1) -> (bs,)
                    # dtype: torch.long
                    pixel_values = torch.multinomial(pixel_probs, 1).squeeze(-1)
                    priors[:, row, col] = pixel_values
        
        del priors, model
        gc.collect()
        torch.cuda.empty_cache()
    else:
        dev = torch.device(1)

        dataset_dir = "/home.local/weicai/VAEs/dataset"
        dataset = MyDataset(dataset_dir, transform=Compose([ToTensor()]))
        img_size = dataset[0].shape[-2:]

        embed_dim = 64
        num_embeddings = 512

        with StateStdout(logger=logger, begin="Loading VQ-VAE model..", end="Done!"):
            vq_vae = VQVAE(img_size, embed_dim, num_embeddings)
            vq_vae.eval()
            vq_vae.to(dev)
        
            ckpt = "/home.local/weicai/VAEs/models/vq_vae_ckpt/best.pt"
            sd = torch.load(ckpt, map_location=dev)
            vq_vae.load_state_dict(sd)

        with StateStdout(logger=logger, begin="Collecting embedding indices..", end="Done!"):
            embed_indices = []
            for img_ts in dataset:
                img_ts = img_ts.unsqueeze(0).to(dev)
                with torch.no_grad():
                    latent = vq_vae.encoder(img_ts)

                # (b, d, h, w) -> (b, h, w, d)
                latent = latent.permute(0, 2, 3, 1).contiguous()
                b, h, w, d = latent.shape
                # (b, h, w, d) -> (b * h * w, d)
                latent = latent.reshape(-1, d)
                
                # (b * h * w,)
                embed_ind = vq_vae.vq_layer.get_embed_indices(latent)
                # (b * h * w,) -> (b, h, w)
                embed_ind = embed_ind.reshape(b, h, w).cpu()
                embed_indices.append(embed_ind)
        
        with StateStdout(logger=logger, begin="Loading GatedPixelCNN model..", end="Done!"):
            model = GatedPixelCNN(num_embeddings, embed_dim, num_embeddings)
            model.to(dev)

        total_steps = 1600
        optimizer = AdamW(model.parameters(), lr=1e-3)
        lr_scheduler = get_scheduler(
            "linear",
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )

        model.train()
        optimizer.zero_grad(set_to_none=True)

        ckpt_dir = "gated_pixel_cnn_ckpt"
        os.makedirs(ckpt_dir)

        with StateStdout(logger=logger, begin="Training GatedPixelCNN.."):
            best = math.inf
            for step in range(total_steps):
                mean_loss = 0.
                for ind in embed_indices:
                    ind = ind.to(dev)
                    # (b, h, w) -> (b, h, w, c)
                    one_hot_ind = F.one_hot(ind, num_classes=num_embeddings)
                    # (b, h, w, c) -> (b, c, h, w)
                    # long -> float32
                    one_hot_ind = one_hot_ind.permute(0, 3, 1, 2).contiguous().float()

                    logit = model(one_hot_ind)

                    loss = F.cross_entropy(logit, ind) / len(embed_indices)
                    loss.backward()
                    
                    loss = loss.item()
                    mean_loss += loss

                if mean_loss < best:
                    best = mean_loss
                    torch.save(model.state_dict(), os.path.join(ckpt_dir, "best.pt"))

                if step % 10 == 0:
                    print(f"Step [{step + 1}/{total_steps}]\tLoss {mean_loss}\tLr {optimizer.param_groups[0]['lr']}")

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        
            dst = os.path.join(ckpt_dir, "last.pt")
            torch.save(model.state_dict(), dst)
            print(f"The last checkpoint has been saved to: {dst}")

    print("System End.")
