#!/usr/bin/env python3
"""
Convert INL-Diffusion checkpoints to safetensors format.

Usage:
    python convert_to_safetensors.py --vae vae_checkpoints/vae_final.pt --output vae_model
    python convert_to_safetensors.py --dit dit_checkpoints/dit_final.pt --output dit_model
"""

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def convert_vae(checkpoint_path: str, output_dir: str):
    """Convert VAE checkpoint to safetensors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading VAE from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})
    step = checkpoint.get("step", 0)

    # Convert to float16
    print("Converting to float16...")
    converted = {}
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            converted[key] = tensor.half()
        else:
            converted[key] = tensor

    # Save safetensors
    safetensors_path = output_dir / "model.safetensors"
    save_file(converted, safetensors_path)
    print(f"Saved: {safetensors_path} ({safetensors_path.stat().st_size / 1e6:.2f} MB)")

    # Save config
    vae_config = {
        "model_type": "inl-vae",
        "image_size": config.get("image_size", 256),
        "base_channels": config.get("base_channels", 128),
        "latent_dim": config.get("latent_dim", 4),
        "channel_mult": [1, 2, 4, 4],
        "num_res_blocks": 2,
        "training_steps": step,
        "torch_dtype": "float16",
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(vae_config, f, indent=2)
    print(f"Saved: {config_path}")

    print(f"\nVAE converted successfully to {output_dir}")


def convert_dit(checkpoint_path: str, output_dir: str):
    """Convert DiT checkpoint to safetensors."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading DiT from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    state_dict = checkpoint["model_state_dict"]
    config = checkpoint.get("config", {})
    step = checkpoint.get("step", 0)

    # Convert to float16
    print("Converting to float16...")
    converted = {}
    for key, tensor in state_dict.items():
        if tensor.dtype == torch.float32:
            converted[key] = tensor.half()
        else:
            converted[key] = tensor

    # Save safetensors
    safetensors_path = output_dir / "model.safetensors"
    save_file(converted, safetensors_path)
    print(f"Saved: {safetensors_path} ({safetensors_path.stat().st_size / 1e6:.2f} MB)")

    # DiT configs
    DIT_CONFIGS = {
        "S": {"d_model": 384, "num_layers": 12, "num_heads": 6, "num_kv_heads": 2},
        "B": {"d_model": 768, "num_layers": 12, "num_heads": 12, "num_kv_heads": 4},
        "L": {"d_model": 1024, "num_layers": 24, "num_heads": 16, "num_kv_heads": 4},
        "XL": {"d_model": 1152, "num_layers": 28, "num_heads": 16, "num_kv_heads": 4},
        "XXL": {"d_model": 1536, "num_layers": 32, "num_heads": 24, "num_kv_heads": 6},
    }

    dit_size = config.get("dit_size", "L")
    dit_params = DIT_CONFIGS.get(dit_size, DIT_CONFIGS["L"])

    dit_config = {
        "model_type": "inl-dit",
        "dit_size": dit_size,
        "img_size": config.get("img_size", 32),
        "patch_size": 2,
        "in_channels": config.get("latent_channels", 4),
        "d_model": dit_params["d_model"],
        "num_layers": dit_params["num_layers"],
        "num_heads": dit_params["num_heads"],
        "num_kv_heads": dit_params["num_kv_heads"],
        "context_dim": 2048,
        "mlp_ratio": 4.0,
        "num_integrator_iterations": 2,
        "training_steps": step,
        "torch_dtype": "float16",
    }

    config_path = output_dir / "config.json"
    with open(config_path, "w") as f:
        json.dump(dit_config, f, indent=2)
    print(f"Saved: {config_path}")

    print(f"\nDiT converted successfully to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Convert INL-Diffusion to safetensors")
    parser.add_argument("--vae", type=str, help="VAE checkpoint path")
    parser.add_argument("--dit", type=str, help="DiT checkpoint path")
    parser.add_argument("--output", "-o", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    if args.vae:
        convert_vae(args.vae, args.output)
    elif args.dit:
        convert_dit(args.dit, args.output)
    else:
        print("Error: Specify --vae or --dit")
        exit(1)


if __name__ == "__main__":
    main()
