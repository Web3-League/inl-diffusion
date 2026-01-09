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
from inl_diffusion.tokenizer import INLTokenizer


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
CFG_DROPOUT = float(os.getenv("CFG_DROPOUT", "0.1"))  # 10% unconditional training for CFG
USE_BF16 = os.getenv("USE_BF16", "1") == "1"  # Enable bf16 by default

# Model
DIT_SIZE = os.getenv("DIT_SIZE", "L")  # S, B, L, XL, XXL

# Tokenizer
TOKENIZER_PATH = os.getenv("TOKENIZER_PATH", "checkpoints/tokenizer/tokenizer.json")

# Output
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "dit_checkpoints"))


# ============================================================================
# TEXT ENCODER
# ============================================================================

class INLTextEncoder(nn.Module):
    """
    INL Text Encoder for diffusion model conditioning.

    Uses the INL tokenizer vocabulary (100K tokens) and encodes text
    into embeddings for cross-attention in DiT.
    """

    def __init__(self, vocab_size: int = 100000, d_model: int = 2048, max_length: int = 77, num_layers: int = 6):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, max_length, d_model) * 0.02)

        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=8,
            dim_feedforward=d_model * 4,
            batch_first=True,
            dropout=0.1,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Layer norm for output
        self.ln_final = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Encode token IDs to embeddings.

        Args:
            input_ids: [B, seq_len] token IDs

        Returns:
            [B, seq_len, d_model] text embeddings
        """
        x = self.embedding(input_ids)
        x = x + self.pos_embedding[:, :x.size(1)]
        x = self.encoder(x)
        x = self.ln_final(x)
        return x


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
    """Train DiT with bf16 mixed precision and CFG support."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Setup bf16 mixed precision
    use_amp = USE_BF16 and torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    amp_dtype = torch.bfloat16 if use_amp else torch.float32
    print(f"Mixed precision: {'bf16' if use_amp else 'fp32'}")

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

    # INL Tokenizer (from Pacific-Prime/pacific-tiny or local)
    tokenizer_path = args.tokenizer_path if hasattr(args, 'tokenizer_path') and args.tokenizer_path else TOKENIZER_PATH
    print(f"\nLoading INL tokenizer from {tokenizer_path}...")
    tokenizer = INLTokenizer(tokenizer_path=tokenizer_path)
    vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizer vocab size: {vocab_size:,}")

    # INL Text Encoder
    text_encoder = INLTextEncoder(vocab_size=vocab_size, d_model=2048, num_layers=6).to(device)
    print(f"Text encoder: {sum(p.numel() for p in text_encoder.parameters()) / 1e6:.2f}M params")

    # Noise scheduler
    scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule="scaled_linear",
        prediction_type="epsilon",
    )

    # Dataset
    if args.local_dataset:
        # Use pre-downloaded HuggingFace dataset (much faster!)
        from datasets import load_from_disk

        print(f"Loading local dataset from {args.local_dataset}...")
        hf_dataset = load_from_disk(args.local_dataset)
        print(f"Dataset size: {len(hf_dataset)} samples")

        transform = get_transform(IMAGE_SIZE)

        class LocalHFDataset(torch.utils.data.Dataset):
            def __init__(self, hf_ds, transform):
                self.hf_ds = hf_ds
                self.transform = transform

            def __len__(self):
                return len(self.hf_ds)

            def __getitem__(self, idx):
                item = self.hf_ds[idx]
                image = item.get("image") or item.get("img")
                text = item.get("text") or item.get("caption") or item.get("artist", "art")

                if hasattr(image, "convert"):
                    image = image.convert("RGB")
                image = self.transform(image)

                return {"image": image, "text": str(text)}

        dataset = LocalHFDataset(hf_dataset, transform)
        dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=True)

    elif args.local_data:
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
    print(f"CFG dropout: {CFG_DROPOUT*100:.0f}%")
    print(f"Mixed precision: {'bf16' if use_amp else 'fp32'}")
    print(f"{'='*60}\n")

    # Create null text embedding for CFG (unconditional)
    with torch.no_grad():
        null_tokens = torch.zeros(1, 77, dtype=torch.long, device=device)
        null_text_embedding = text_encoder(null_tokens)  # [1, 77, 2048]

    step = 0
    start_time = time.time()
    total_loss = 0
    lr = 0.0  # Will be set during warmup

    pbar = tqdm(total=MAX_STEPS, desc="Training")

    while step < MAX_STEPS:
        for batch in dataloader:
            images = batch["image"].to(device)
            texts = batch["text"]

            # Encode images to latents
            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    latents = vae.tokenize(images)

            # Encode text
            tokens = tokenizer(texts, max_length=77, padding="max_length", truncation=True)
            input_ids = tokens["input_ids"].to(device)

            with torch.no_grad():
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    text_embeddings = text_encoder(input_ids)

            # CFG: randomly drop text conditioning (use null embedding)
            batch_size = text_embeddings.shape[0]
            cfg_mask = torch.rand(batch_size, device=device) < CFG_DROPOUT
            if cfg_mask.any():
                # Replace with null embedding for CFG training
                null_expanded = null_text_embedding.expand(batch_size, -1, -1)
                text_embeddings = torch.where(
                    cfg_mask.view(-1, 1, 1),
                    null_expanded,
                    text_embeddings
                )

            # Sample noise
            noise = torch.randn_like(latents)

            # Sample timesteps
            timesteps = torch.randint(0, scheduler.num_train_timesteps, (latents.shape[0],), device=device)

            # Add noise
            noisy_latents = scheduler.add_noise(latents, noise, timesteps)

            # Forward pass with mixed precision
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                # Predict noise
                noise_pred = dit(noisy_latents, timesteps, text_embeddings)

                # Loss
                loss = F.mse_loss(noise_pred, noise)
                loss = loss / GRADIENT_ACCUMULATION

            # Backward with scaler
            scaler.scale(loss).backward()
            total_loss += loss.item() * GRADIENT_ACCUMULATION  # Unscale for logging
            current_loss = loss.item() * GRADIENT_ACCUMULATION

            # Compute gradient norm before clipping
            grad_norm = 0.0
            if (step + 1) % GRADIENT_ACCUMULATION == 0:
                # Unscale before clipping
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(dit.parameters(), 1.0).item()

                # Optimizer step with scaler
                scaler.step(optimizer)
                scaler.update()
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
                    num_samples = 4
                    cfg_scale = 7.5  # Classifier-free guidance scale

                    # Sample from noise
                    sample_latents = torch.randn(num_samples, latent_channels, latent_size, latent_size, device=device)

                    # Text embeddings (conditioned)
                    sample_tokens = torch.randint(0, 50000, (num_samples, 77), device=device)
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        cond_embeddings = text_encoder(sample_tokens)
                        # Null embeddings (unconditioned)
                        uncond_embeddings = null_text_embedding.expand(num_samples, -1, -1)

                    # Denoise with CFG (few steps for speed)
                    for t in reversed(range(0, 1000, 50)):
                        t_tensor = torch.full((num_samples,), t, device=device, dtype=torch.long)

                        with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                            # Predict noise for both conditioned and unconditioned
                            noise_pred_uncond = dit(sample_latents, t_tensor, uncond_embeddings)
                            noise_pred_cond = dit(sample_latents, t_tensor, cond_embeddings)

                            # CFG: combine predictions
                            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)

                        sample_latents = scheduler.step(noise_pred.float(), t, sample_latents)

                    # Decode to images
                    with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        samples = vae.detokenize(sample_latents)
                    samples = torch.clamp(samples.float(), 0, 1)

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
    parser.add_argument("--local_dataset", type=str, default=None,
                        help="Path to pre-downloaded HuggingFace dataset (from download_dataset.py)")
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
    parser.add_argument("--tokenizer_path", type=str, default=None,
                        help="Path to INL tokenizer.json (default: checkpoints/tokenizer/tokenizer.json)")

    args = parser.parse_args()

    global BATCH_SIZE, DIT_SIZE, MAX_STEPS, LEARNING_RATE
    BATCH_SIZE = args.batch_size
    DIT_SIZE = args.dit_size
    MAX_STEPS = args.max_steps
    LEARNING_RATE = args.lr

    train_dit(args)


if __name__ == "__main__":
    main()
