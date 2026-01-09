#!/usr/bin/env python3
"""
Train INL-DiT for text-to-image diffusion.

Requires:
- Trained INL-VAE for image encoding
- Text encoder (INL-LLM or CLIP)

Usage:
    python train_dit.py --vae_path vae_checkpoints/vae_final.pt --dataset laion/laion-art
    python train_dit.py --vae_path vae_final.pt --dit_size XL --batch_size 16
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

from inl_diffusion.vae import INLVAE
from inl_diffusion.dit import INLDiT
from inl_diffusion.pipeline.text_to_image import DDPMScheduler


# ============================================================================
# CONFIGURATION
# ============================================================================

# Training
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
IMAGE_SIZE = int(os.getenv("IMAGE_SIZE", "256"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "500000"))
LEARNING_RATE = float(os.getenv("LEARNING_RATE", "1e-4"))
WARMUP_STEPS = int(os.getenv("WARMUP_STEPS", "5000"))
SAVE_EVERY = int(os.getenv("SAVE_EVERY", "10000"))
LOG_INTERVAL = int(os.getenv("LOG_INTERVAL", "100"))
GRADIENT_ACCUMULATION = int(os.getenv("GRADIENT_ACCUMULATION", "4"))

# Model
DIT_SIZE = os.getenv("DIT_SIZE", "L")  # S, B, L, XL, XXL

# Output
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "dit_checkpoints"))


# ============================================================================
# TEXT ENCODER
# ============================================================================

class DummyTextEncoder(nn.Module):
    """Dummy text encoder for testing (replace with INL-LLM or CLIP)."""

    def __init__(self, vocab_size: int = 50000, d_model: int = 2048, max_length: int = 77):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)

        # Simple transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=4)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.encoder(x)
        return x


class SimpleTokenizer:
    """Simple tokenizer for testing."""

    def __init__(self, vocab_size: int = 50000):
        self.vocab_size = vocab_size

    def __call__(self, texts, **kwargs):
        max_length = kwargs.get("max_length", 77)
        batch_size = len(texts) if isinstance(texts, list) else 1

        # Random tokens for testing
        input_ids = torch.randint(0, self.vocab_size, (batch_size, max_length))

        return {"input_ids": input_ids}


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
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class TextImageDataset(torch.utils.data.IterableDataset):
    """Streaming text-image dataset."""

    def __init__(self, dataset_name: str, image_size: int = 256, max_samples: int = None):
        self.transform = get_transform(image_size)
        self.max_samples = max_samples

        print(f"Loading dataset: {dataset_name}")

        self.dataset = load_dataset(dataset_name, split="train", streaming=True)

        if max_samples:
            self.dataset = self.dataset.take(max_samples)
            print(f"Limited to {max_samples} samples")

        # Find keys
        self.image_key = "image"
        self.text_key = "text"

    def _load_image_from_url(self, url: str):
        """Download image from URL."""
        import requests
        from io import BytesIO
        from PIL import Image

        try:
            response = requests.get(url, timeout=5)
            response.raise_for_status()
            return Image.open(BytesIO(response.content))
        except Exception:
            return None

    def __iter__(self):
        for item in self.dataset:
            try:
                # Try multiple common image keys (case insensitive for LAION)
                image = item.get(self.image_key) or item.get("img") or item.get("pixel_values")
                text = item.get(self.text_key) or item.get("TEXT") or item.get("caption") or item.get("label", "")

                # Handle URL-based datasets (like LAION)
                if image is None:
                    url = item.get("URL") or item.get("url") or item.get("image_url")
                    if url:
                        image = self._load_image_from_url(url)

                # Convert label to string if it's an int (e.g., CIFAR classes)
                if isinstance(text, int):
                    text = f"class {text}"

                if image is None:
                    continue

                if hasattr(image, "convert"):
                    image = image.convert("RGB")

                image = self.transform(image)

                yield {"image": image, "text": text}
            except Exception:
                continue


# ============================================================================
# TRAINING
# ============================================================================

def train_dit(args):
    """Train DiT."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create output directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    samples_dir = OUTPUT_DIR / "samples"
    samples_dir.mkdir(exist_ok=True)

    # TensorBoard
    run_name = f"dit_{DIT_SIZE}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(OUTPUT_DIR / "runs" / run_name)

    # Load VAE
    print(f"\nLoading VAE from {args.vae_path}...")
    vae_checkpoint = torch.load(args.vae_path, map_location="cpu")
    vae_config = vae_checkpoint.get("config", {})

    vae = INLVAE(
        image_size=vae_config.get("image_size", IMAGE_SIZE),
        base_channels=vae_config.get("base_channels", 128),
        latent_dim=vae_config.get("latent_dim", 4),
    ).to(device)
    vae.load_state_dict(vae_checkpoint["model_state_dict"])
    vae.eval()
    for param in vae.parameters():
        param.requires_grad = False
    print(f"VAE loaded ({vae.get_num_params() / 1e6:.2f}M params)")

    # Latent dimensions
    latent_size = IMAGE_SIZE // 8
    latent_channels = vae_config.get("latent_dim", 4)

    # Create DiT
    print(f"\nCreating INL-DiT ({DIT_SIZE})...")
    dit = INLDiT.from_config(
        DIT_SIZE,
        img_size=latent_size,
        patch_size=2,
        in_channels=latent_channels,
        context_dim=2048,  # Text encoder dim
    ).to(device)
    print(f"DiT Parameters: {dit.get_num_params() / 1e6:.2f}M")

    # Text encoder (placeholder - replace with INL-LLM)
    text_encoder = DummyTextEncoder(d_model=2048).to(device)
    tokenizer = SimpleTokenizer()
    print(f"Text encoder: {sum(p.numel() for p in text_encoder.parameters()) / 1e6:.2f}M params")

    # Noise scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    # Dataset
    if args.local_data:
        # Use torchvision ImageFolder with dummy captions
        from torchvision.datasets import ImageFolder

        class LocalDataset(ImageFolder):
            def __getitem__(self, idx):
                image, _ = super().__getitem__(idx)
                return {"image": image, "text": "a photo"}

        transform = get_transform(IMAGE_SIZE)
        dataset = LocalDataset(args.local_data, transform=transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    else:
        dataset = TextImageDataset(args.dataset, IMAGE_SIZE, max_samples=args.max_samples)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE)

    # Optimizer
    optimizer = torch.optim.AdamW(dit.parameters(), lr=LEARNING_RATE, weight_decay=0.01)

    # Training loop
    print(f"\n{'='*60}")
    print("TRAINING START")
    print(f"{'='*60}")
    print(f"DiT size: {DIT_SIZE}")
    print(f"Batch size: {BATCH_SIZE} (grad accum: {GRADIENT_ACCUMULATION})")
    print(f"Image size: {IMAGE_SIZE} → Latent: {latent_size}x{latent_size}")
    print(f"Max steps: {MAX_STEPS}")
    print(f"{'='*60}\n")

    step = 0
    start_time = time.time()
    total_loss = 0

    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while step < MAX_STEPS:
        for batch in dataloader:
            images = batch["image"].to(device)
            texts = batch["text"]

            # Encode images to latents
            with torch.no_grad():
                latents = vae.tokenize(images)

            # Encode text
            tokens = tokenizer(texts, max_length=77, padding="max_length", truncation=True)
            input_ids = tokens["input_ids"].to(device)

            with torch.no_grad():
                text_embeddings = text_encoder(input_ids)

            # Sample noise
            noise = torch.randn_like(latents)

            # Sample timesteps
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=device)

            # Add noise
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Predict noise
            noise_pred = dit(noisy_latents, timesteps, text_embeddings)

            # Loss
            loss = F.mse_loss(noise_pred, noise)
            loss = loss / GRADIENT_ACCUMULATION

            loss.backward()
            total_loss += loss.item() * GRADIENT_ACCUMULATION  # Unscale for logging
            current_loss = loss.item() * GRADIENT_ACCUMULATION

            # Compute gradient norm before clipping
            grad_norm = 0.0
            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0).item()
                optimizer.step()
                optimizer.zero_grad()

                # Learning rate warmup
                if step < WARMUP_STEPS:
                    lr = LEARNING_RATE * (step + 1) / WARMUP_STEPS
                else:
                    lr = LEARNING_RATE

                for param_group in optimizer.param_groups:
                    param_group["lr"] = lr

            step += 1

            # Update progress bar with stats every step
            pbar.set_postfix({
                "loss": f"{current_loss:.4f}",
                "lr": f"{lr:.2e}",
                "grad": f"{grad_norm:.2f}" if grad_norm > 0 else "acc",
            })
            pbar.update(1)

            # TensorBoard logging
            if step % LOG_INTERVAL == 0:
                avg_loss = total_loss / LOG_INTERVAL

                writer.add_scalar("train/loss", avg_loss, step)
                writer.add_scalar("train/lr", lr, step)
                if grad_norm > 0:
                    writer.add_scalar("train/grad_norm", grad_norm, step)

                total_loss = 0

            # Generate samples
            if step % SAVE_EVERY == 0:
                print(f"\nGenerating samples at step {step}...")
                dit.eval()

                with torch.no_grad():
                    # Sample from noise
                    sample_latents = torch.randn(4, latent_channels, latent_size, latent_size, device=device)

                    # Dummy text embeddings
                    sample_text = text_encoder(torch.randint(0, 50000, (4, 77), device=device))

                    # Denoise (few steps for speed)
                    for t in reversed(range(0, 1000, 50)):
                        t_tensor = torch.full((4,), t, device=device, dtype=torch.long)
                        noise_pred = dit(sample_latents, t_tensor, sample_text)
                        sample_latents = scheduler.step(noise_pred, t, sample_latents)

                    # Decode to images
                    samples = vae.detokenize(sample_latents)
                    samples = torch.clamp(samples, 0, 1)

                    # Save
                    grid = make_grid(samples, nrow=2)
                    save_image(grid, samples_dir / f"step_{step}.png")
                    writer.add_image("samples/generated", grid, step)

                dit.train()

                # Save checkpoint
                checkpoint_path = OUTPUT_DIR / f"dit_step_{step}.pt"
                torch.save({
                    "step": step,
                    "model_state_dict": dit.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": {
                        "dit_size": DIT_SIZE,
                        "img_size": latent_size,
                        "latent_channels": latent_channels,
                    },
                }, checkpoint_path)
                print(f"Saved checkpoint: {checkpoint_path}")

            if step >= MAX_STEPS:
                break

    pbar.close()

    # Save final model
    final_path = OUTPUT_DIR / "dit_final.pt"
    torch.save({
        "model_state_dict": dit.state_dict(),
        "config": {
            "dit_size": DIT_SIZE,
            "img_size": latent_size,
            "latent_channels": latent_channels,
        },
    }, final_path)
    print(f"\nTraining complete! Final model: {final_path}")

    writer.close()


def main():
    parser = argparse.ArgumentParser(description="Train INL-DiT")
    parser.add_argument("--vae_path", type=str, required=True,
                        help="Path to trained VAE checkpoint")
    parser.add_argument("--dataset", type=str, default="laion/laion-art",
                        help="HuggingFace dataset name")
    parser.add_argument("--local_data", type=str, default=None,
                        help="Path to local image folder")
    parser.add_argument("--dit_size", type=str, default="L",
                        choices=["S", "B", "L", "XL", "XXL"],
                        help="DiT model size")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--max_steps", type=int, default=500000,
                        help="Maximum training steps")
    parser.add_argument("--lr", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Limit dataset to N samples (faster loading)")

    args = parser.parse_args()

    global BATCH_SIZE, DIT_SIZE, MAX_STEPS, LEARNING_RATE
    BATCH_SIZE = args.batch_size
    DIT_SIZE = args.dit_size
    MAX_STEPS = args.max_steps
    LEARNING_RATE = args.lr

    train_dit(args)


if __name__ == "__main__":
    main()
