#!/usr/bin/env python3
"""
Train INL-VAE for image tokenization.

This VAE will encode images to latent space for diffusion training.

Usage:
    python train_vae.py --dataset laion/laion-art --image_size 256 --batch_size 32
    python train_vae.py --dataset imagenet-1k --image_size 256 --epochs 100
"""

import os
import time
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("Warning: datasets not installed. Use --local_data")

from inl_diffusion.vae import INLVAE


# ============================================================================
# CONFIGURATION
# ============================================================================

# Training
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
EPOCHS = int(os.getenv("EPOCHS", "100"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
KL_WEIGHT = float(os.getenv("KL_WEIGHT", "1e-6"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10"))

# Model
BASE_CHANNELS = int(os.getenv("BASE_CHANNELS", "128"))
LATENT_DIM = int(os.getenv("LATENT_DIM", "4"))

# Output
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "vae_checkpoints"))


# ============================================================================
# DATASET
# ============================================================================

def get_transform(image_size: int):
    """Get image transforms."""
    return transforms.Compose([
        transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])


class ImageDataset(torch.utils.data.IterableDataset):
    """Streaming image dataset from HuggingFace."""

    def __init__(self, dataset_name: str, image_size: int = 256, split: str = "train"):
        self.transform = get_transform(image_size)

        print(f"Loading dataset: {dataset_name}")

        # Try different dataset configs
        try:
            if dataset_name == "imagenet-1k":
                self.dataset = load_dataset("imagenet-1k", split=split, streaming=True)
                self.image_key = "image"
            elif "laion" in dataset_name.lower():
                self.dataset = load_dataset(dataset_name, split=split, streaming=True)
                self.image_key = "image"
            else:
                self.dataset = load_dataset(dataset_name, split=split, streaming=True)
                # Try to find image key
                self.image_key = "image"

            print(f"Loaded {dataset_name}")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            raise

    def __iter__(self):
        for item in self.dataset:
            try:
                image = item.get(self.image_key) or item.get("img") or item.get("jpg")
                if image is None:
                    continue

                # Convert to RGB
                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                # Apply transforms
                image = self.transform(image)
                yield image
            except Exception:
                continue


def get_local_dataloader(data_dir: str, image_size: int, batch_size: int, num_workers: int = 4):
    """Get dataloader for local image folder."""
    from torchvision.datasets import ImageFolder

    transform = get_transform(image_size)
    dataset = ImageFolder(data_dir, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


# ============================================================================
# TRAINING
# ============================================================================

class PerceptualLoss(nn.Module):
    """Perceptual loss using VGG features."""

    def __init__(self):
        super().__init__()
        try:
            from torchvision.models import vgg16, VGG16_Weights
            vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        except:
            from torchvision.models import vgg16
            vgg = vgg16(pretrained=True)

        self.features = nn.Sequential(*list(vgg.features)[:23]).eval()

        for param in self.features.parameters():
            param.requires_grad = False

        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Normalize from [-1, 1] to ImageNet normalization
        x = (x + 1) / 2
        y = (y + 1) / 2
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std

        x_features = self.features(x)
        y_features = self.features(y)

        return F.mse_loss(x_features, y_features)


def train_vae(args):
    """Train VAE."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_dir = OUTPUT_DIR / "samples"
    samples_dir.mkdir(exist_ok=True)

    # TensorBoard
    run_name = f"vae_{IMAGE_SIZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(OUTPUT_DIR / "runs" / run_name)

    # Model
    vae = INLVAE(
        image_size=IMAGE_SIZE,
        base_channels=BASE_CHANNELS,
        latent_dim=LATENT_DIM,
    ).to(device)

    num_params = vae.get_num_params()
    print(f"\nINL-VAE Parameters: {num_params / 1e6:.2f}M")
    print(f"  Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"  Latent size: {IMAGE_SIZE // 8}x{IMAGE_SIZE // 8}x{LATENT_DIM}")
    print(f"  Base channels: {BASE_CHANNELS}")

    # Dataset
    if args.local_data:
        dataloader = get_local_dataloader(args.local_data, IMAGE_SIZE, BATCH_SIZE)
        print(f"\nLoaded local data from {args.local_data}")
    else:
        if not HAS_DATASETS:
            raise RuntimeError("datasets not installed. Use --local_data or pip install datasets")
        dataset = ImageDataset(args.dataset, IMAGE_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        print(f"\nLoaded streaming dataset: {args.dataset}")

    # Losses
    perceptual_loss = PerceptualLoss().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(vae.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS * 10000)

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING START")
    print(f"{'='*60}")

    global_step = 0
    start_time = time.time()

    for epoch in range(EPOCHS):
        vae.train()
        epoch_losses = {"total": 0, "recon": 0, "perceptual": 0, "kl": 0}
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{EPOCHS}")

        for batch in pbar:
            if isinstance(batch, (tuple, list)):
                images = batch[0]
            else:
                images = batch

            images = images.to(device)

            # Forward
            recon, mean, logvar = vae(images)

            # Losses
            recon_loss = F.mse_loss(recon, images)
            perc_loss = perceptual_loss(recon, images)
            kl_loss = -0.5 * torch.mean(1 + logvar - mean.pow(2) - logvar.exp())

            total_loss = recon_loss + 0.1 * perc_loss + KL_WEIGHT * kl_loss

            # Backward
            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(vae.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Logging
            epoch_losses["total"] += total_loss.item()
            epoch_losses["recon"] += recon_loss.item()
            epoch_losses["perceptual"] += perc_loss.item()
            epoch_losses["kl"] += kl_loss.item()
            num_batches += 1

            pbar.set_postfix({
                "loss": f"{total_loss.item():.4f}",
                "recon": f"{recon_loss.item():.4f}",
                "kl": f"{kl_loss.item():.4f}",
            })

            # TensorBoard
            writer.add_scalar("train/total_loss", total_loss.item(), global_step)
            writer.add_scalar("train/recon_loss", recon_loss.item(), global_step)
            writer.add_scalar("train/perceptual_loss", perc_loss.item(), global_step)
            writer.add_scalar("train/kl_loss", kl_loss.item(), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)

            global_step += 1

            # Save samples periodically
            if global_step % 1000 == 0:
                vae.eval()
                with torch.no_grad():
                    # Original vs reconstruction
                    comparison = torch.cat([images[:8], recon[:8]])
                    comparison = (comparison + 1) / 2  # [-1,1] -> [0,1]
                    grid = make_grid(comparison, nrow=8)
                    save_image(grid, samples_dir / f"step_{global_step}.png")
                    writer.add_image("samples/reconstruction", grid, global_step)
                vae.train()

        # Epoch summary
        avg_losses = {k: v / num_batches for k, v in epoch_losses.items()}
        elapsed = time.time() - start_time

        print(f"\nEpoch {epoch+1}/{EPOCHS} | "
              f"Loss: {avg_losses['total']:.4f} | "
              f"Recon: {avg_losses['recon']:.4f} | "
              f"KL: {avg_losses['kl']:.4f} | "
              f"Time: {elapsed/60:.1f}min")

        # Save checkpoint
        if (epoch + 1) % SAVE_EVERY == 0:
            checkpoint_path = OUTPUT_DIR / f"vae_epoch_{epoch+1}.pt"
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": vae.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": {
                    "image_size": IMAGE_SIZE,
                    "base_channels": BASE_CHANNELS,
                    "latent_dim": LATENT_DIM,
                },
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_path = OUTPUT_DIR / "vae_final.pt"
    torch.save({
        "model_state_dict": vae.state_dict(),
        "config": {
            "image_size": IMAGE_SIZE,
            "base_channels": BASE_CHANNELS,
            "latent_dim": LATENT_DIM,
        },
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train INL-VAE")
    parser.add_argument("--dataset", type=str, default="laion/laion-art",
                        help="HuggingFace dataset name")
    parser.add_argument("--local_data", type=str, default=None,
                        help="Path to local image folder")
    parser.add_argument("--image_size", type=int, default=256,
                        help="Image size")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=1e-6,
                        help="KL divergence weight")

    args = parser.parse_args()

    # Override globals with args
    global BATCH_SIZE, IMAGE_SIZE, EPOCHS, LEARNING_RATE, KL_WEIGHT
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size
    EPOCHS = args.epochs
    LEARNING_RATE = args.lr
    KL_WEIGHT = args.kl_weight

    train_vae(args)


if __name__ == "__main__":
    main()
