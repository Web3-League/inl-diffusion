#!/usr/bin/env python3
"""Test VAE reconstruction."""

import torch
from PIL import Image
from torchvision import transforms
from pathlib import Path

from inl_diffusion import INLVAE


def test_reconstruction(image_path: str, vae_path: str = "checkpoints/vae-safetensors"):
    """Test VAE encode/decode on an image."""

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load VAE
    print(f"Loading VAE from {vae_path}...")
    vae = INLVAE.from_pretrained(vae_path)
    vae = vae.to(device).eval()

    # Load and preprocess image
    print(f"Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    original_size = image.size

    # Resize to 256x256
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])  # [-1, 1]
    ])

    x = transform(image).unsqueeze(0).to(device)

    # Encode -> Decode
    print("Encoding...")
    with torch.no_grad():
        latent = vae.encode(x)
        print(f"Latent shape: {latent.shape}")  # Should be [1, 4, 32, 32]

        print("Decoding...")
        reconstructed = vae.decode(latent)

    # Convert back to image
    reconstructed = reconstructed.squeeze(0).cpu()
    reconstructed = (reconstructed * 0.5 + 0.5).clamp(0, 1)  # [0, 1]
    reconstructed = transforms.ToPILImage()(reconstructed)

    # Save comparison
    output_dir = Path("test_outputs")
    output_dir.mkdir(exist_ok=True)

    # Resize original to 256x256 for comparison
    original_resized = Image.open(image_path).convert("RGB").resize((256, 256))

    # Side by side comparison
    comparison = Image.new("RGB", (512, 256))
    comparison.paste(original_resized, (0, 0))
    comparison.paste(reconstructed, (256, 0))

    comparison_path = output_dir / "comparison.png"
    comparison.save(comparison_path)
    print(f"Saved comparison: {comparison_path}")

    # Also save individual files
    original_resized.save(output_dir / "original.png")
    reconstructed.save(output_dir / "reconstructed.png")
    print(f"Saved original and reconstructed to {output_dir}/")

    # Calculate reconstruction error
    orig_tensor = transforms.ToTensor()(original_resized)
    recon_tensor = transforms.ToTensor()(reconstructed)
    mse = torch.mean((orig_tensor - recon_tensor) ** 2).item()
    print(f"\nReconstruction MSE: {mse:.6f}")

    return comparison_path


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        # Download a test image if none provided
        print("Usage: python test_vae.py <image_path>")
        print("\nNo image provided, downloading test image...")

        import urllib.request
        test_url = "https://upload.wikimedia.org/wikipedia/commons/thumb/e/ea/Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg/300px-Van_Gogh_-_Starry_Night_-_Google_Art_Project.jpg"
        test_path = "test_image.jpg"
        urllib.request.urlretrieve(test_url, test_path)
        print(f"Downloaded: {test_path}")
        image_path = test_path
    else:
        image_path = sys.argv[1]

    test_reconstruction(image_path)
