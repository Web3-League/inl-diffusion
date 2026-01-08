#!/usr/bin/env python3
"""
Train INL-VAE for image tokenization.

This VAE will encode images to latent space for diffusion training.

Usage:
    python train_vae.py --dataset laion/laion-art --max_steps 100000
    python train_vae.py --dataset huggan/wikiart --max_steps 50000 --batch_size 32
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
MAX_STEPS = int(os.getenv("MAX_STEPS", "100000"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "1000"))
KL_WEIGHT = float(os.getenv("KL_WEIGHT", "1e-6"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10000"))
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", "100"))

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
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # [-1, 1]
    ])


class ImageDataset(torch.utils.data.IterableDataset):
    """Streaming image dataset from HuggingFace."""

    def __init__(self, dataset_name: str, image_size: int = 256, split: str = "train"):
        self.transform = get_transform(image_size)
        self.dataset_name = dataset_name

        print(f"Loading dataset: {dataset_name}")

        try:
            self.dataset = load_dataset(dataset_name, split=split, streaming=True)
            print(f"Loaded {dataset_name} (streaming)")
        except Exception as e:
            print(f"Error loading {dataset_name}: {e}")
            raise

    def __iter__(self):
        for item in self.dataset:
            try:
                # Try different image keys
                image = (item.get("image") or item.get("img") or
                        item.get("jpg") or item.get("png"))
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
    """Train VAE with step-based training."""
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

    print(f"\n{'='*60}")
    print("INL-VAE TRAINING")
    print(f"{'='*60}")
    print(f"Parameters: {num_params / 1e6:.2f}M")
    print(f"Image size: {IMAGE_SIZE}x{IMAGE_SIZE}")
    print(f"Latent size: {IMAGE_SIZE // 8}x{IMAGE_SIZE // 8}x{LATENT_DIM}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"Save every: {SAVE_EVERY} steps")
    print(f"{'='*60}\n")

    # Dataset
    if args.local_data:
        dataloader = get_local_dataloader(args.local_data, IMAGE_SIZE, BATCH_SIZE)
        print(f"Loaded local data from {args.local_data}")
    else:
        if not HAS_DATASETS:
            raise RuntimeError("datasets not installed. Use --local_data or pip install datasets")
        dataset = ImageDataset(args.dataset, IMAGE_SIZE)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)
        print(f"Loaded streaming dataset: {args.dataset}")

    # Losses
    perceptual_loss = PerceptualLoss().to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(vae.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Training loop
    print(f"\nTensorBoard: {OUTPUT_DIR / 'runs' / run_name}")
    print("Starting training...\n")

    step = 0
    start_time = time.time()
    running_loss = 0
    running_recon = 0
    running_kl = 0

    vae.train()
    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while step < MAX_STEPS:
        for batch in dataloader:
            if step >= MAX_STEPS:
                break

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

            # Learning rate warmup + cosine decay
            if step < WARMUP_STEPS:
                lr = LEARNING_RATE * step / WARMUP_STEPS
            else:
                progress = (step - WARMUP_STEPS) / (MAX_STEPS - WARMUP_STEPS)
                lr = LEARNING_RATE * 0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

            # Running averages
            running_loss += total_loss.item()
            running_recon += recon_loss.item()
            running_kl += kl_loss.item()

            step += 1
            pbar.update(1)

            # Logging
            if step % LOG_INTERVAL == 0:
                avg_loss = running_loss / LOG_INTERVAL
                avg_recon = running_recon / LOG_INTERVAL
                avg_kl = running_kl / LOG_INTERVAL

                pbar.set_postfix({
                    "loss": f"{avg_loss:.4f}",
                    "recon": f"{avg_recon:.4f}",
                    "kl": f"{avg_kl:.4f}",
                    "lr": f"{lr:.2e}",
                })

                writer.add_scalar("train/total_loss", avg_loss, step)
                writer.add_scalar("train/recon_loss", avg_recon, step)
                writer.add_scalar("train/kl_loss", avg_kl, step)
                writer.add_scalar("train/lr", lr, step)

                running_loss = 0
                running_recon = 0
                running_kl = 0

            # Save samples
            if step % 1000 == 0:
                vae.eval()
                with torch.no_grad():
                    comparison = torch.cat([images[:8], recon[:8]])
                    comparison = (comparison + 1) / 2
                    grid = make_grid(comparison, nrow=8)
                    save_image(grid, samples_dir / f"step_{step}.png")
                    writer.add_image("samples/reconstruction", grid, step)
                vae.train()

            # Save checkpoint
            if step % SAVE_EVERY == 0:
                elapsed = time.time() - start_time
                print(f"\nStep {step}/{MAX_STEPS} | Time: {elapsed/60:.1f}min")

                checkpoint_path = OUTPUT_DIR / f"vae_step_{step}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": vae.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "image_size": IMAGE_SIZE,
                        "base_channels": BASE_CHANNELS,
                        "latent_dim": LATENT_DIM,
                    },
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

    pbar.close()

    # Save final model
    final_path = OUTPUT_DIR / "vae_final.pt"
    torch.save({
        "step": step,
        "model_state_dict": vae.state_dict(),
        "config": {
            "image_size": IMAGE_SIZE,
            "base_channels": BASE_CHANNELS,
            "latent_dim": LATENT_DIM,
        },
    }, final_path)

    elapsed = time.time() - start_time
    print(f"\nTraining complete!")
    print(f"Total time: {elapsed/60:.1f}min")
    print(f"Final model: {final_path}")

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
    parser.add_argument("--max_steps", type=int, default=100000,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--kl_weight", type=float, default=1e-6,
                        help="KL divergence weight")
    parser.add_argument("--save_every", type=int, default=10000,
                        help="Save checkpoint every N steps")

    args = parser.parse_args()

    # Override globals with args
    global BATCH_SIZE, IMAGE_SIZE, MAX_STEPS, LEARNING_RATE, KL_WEIGHT, SAVE_EVERY
    BATCH_SIZE = args.batch_size
    IMAGE_SIZE = args.image_size
    MAX_STEPS = args.max_steps
    LEARNING_RATE = args.lr
    KL_WEIGHT = args.kl_weight
    SAVE_EVERY = args.save_every

    train_vae(args)


if __name__ == "__main__":
    main()
