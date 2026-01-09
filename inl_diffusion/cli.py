#!/usr/bin/env python3
"""INL-Diffusion CLI for image generation."""

import argparse
import sys
from pathlib import Path

import torch
from PIL import Image


def generate_command(args):
    """Generate images from text prompts."""
    from inl_diffusion import INLDiffusionPipeline

    print(f"Loading pipeline from {args.model_dir}...")
    pipeline = INLDiffusionPipeline.load(args.model_dir)

    if args.device:
        pipeline.to(args.device)

    print(f"Generating: {args.prompt}")

    for i in range(args.num_images):
        seed = args.seed + i if args.seed else None

        result = pipeline.generate(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            seed=seed,
        )

        # Save image
        if args.num_images == 1:
            output_path = Path(args.output)
        else:
            output_path = Path(args.output).with_stem(f"{Path(args.output).stem}_{i}")

        result.image.save(output_path)
        print(f"Saved: {output_path} (seed: {result.seed})")


def info_command(args):
    """Show model information."""
    from safetensors import safe_open
    import json

    model_dir = Path(args.model_dir)

    # Load config
    config_path = model_dir / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print("Config:")
        for k, v in config.items():
            print(f"  {k}: {v}")

    # Check model files
    for name in ["vae.safetensors", "dit.safetensors", "model.safetensors"]:
        model_path = model_dir / name
        if model_path.exists():
            size_mb = model_path.stat().st_size / 1e6
            print(f"\n{name}: {size_mb:.2f} MB")

            with safe_open(model_path, framework="pt") as f:
                keys = list(f.keys())
                print(f"  Tensors: {len(keys)}")


def main():
    parser = argparse.ArgumentParser(
        prog="inl-diffusion",
        description="INL-Diffusion: Image Generation with Integrator Neurons",
    )
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate images from text")
    gen_parser.add_argument("prompt", type=str, help="Text prompt")
    gen_parser.add_argument("--model-dir", "-m", type=str, required=True, help="Model directory")
    gen_parser.add_argument("--output", "-o", type=str, default="output.png", help="Output path")
    gen_parser.add_argument("--negative-prompt", type=str, default="", help="Negative prompt")
    gen_parser.add_argument("--steps", type=int, default=50, help="Inference steps")
    gen_parser.add_argument("--guidance", type=float, default=7.5, help="Guidance scale")
    gen_parser.add_argument("--seed", type=int, default=None, help="Random seed")
    gen_parser.add_argument("--num-images", "-n", type=int, default=1, help="Number of images")
    gen_parser.add_argument("--device", type=str, default=None, help="Device (cuda/cpu)")
    gen_parser.set_defaults(func=generate_command)

    # Info command
    info_parser = subparsers.add_parser("info", help="Show model information")
    info_parser.add_argument("model_dir", type=str, help="Model directory")
    info_parser.set_defaults(func=info_command)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
