#!/usr/bin/env python3
"""Download dataset locally for faster training."""

import argparse
from datasets import load_dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="huggan/wikiart")
    parser.add_argument("--output", type=str, default="/workspace/wikiart_local")
    args = parser.parse_args()

    print(f"Downloading {args.dataset}...")
    print("This may take a while...")

    ds = load_dataset(args.dataset, split="train")

    print(f"Dataset size: {len(ds)} samples")
    print(f"Saving to {args.output}...")

    ds.save_to_disk(args.output)

    print("Done!")
    print(f"\nTo use: python train_dit.py --local_dataset {args.output} ...")

if __name__ == "__main__":
    main()
