# INL-Diffusion

Image generation with Integrator Neurons - Text-to-image diffusion model.

## Architecture

```
Text Prompt → [INL-LLM Encoder] → [INL-DiT] → [INL-VAE Decoder] → Image
```

### Components

1. **INL-VAE** - Custom Variational Autoencoder for image tokenization
   - 8x spatial compression (256x256 → 32x32 latents)
   - 4-channel latent space
   - Trained on image reconstruction

2. **INL-DiT** - Diffusion Transformer with Integrator Neurons
   - Adaptive computation per image region
   - 2D RoPE positional encoding
   - GQA (Grouped Query Attention)
   - Cross-attention for text conditioning

3. **Pipeline** - Complete text-to-image generation
   - DDPM/DDIM schedulers
   - Classifier-free guidance

## Installation

```bash
pip install inl-diffusion
```

Or from source:

```bash
git clone https://github.com/Web3-League/inl-diffusion.git
cd inl-diffusion
pip install -e .
```

## Training

### Step 1: Train VAE (image tokenizer)

```bash
python train_vae.py --dataset laion/laion-art --image_size 256 --batch_size 32 --epochs 100
```

### Step 2: Train DiT (diffusion model)

```bash
python train_dit.py --vae_path vae_checkpoints/vae_final.pt --dataset laion/laion-art --dit_size L
```

### Model Sizes

| Size | Parameters | d_model | Layers | Heads |
|------|------------|---------|--------|-------|
| S    | ~100M      | 384     | 12     | 6     |
| B    | ~250M      | 768     | 12     | 12    |
| L    | ~500M      | 1024    | 24     | 16    |
| XL   | ~700M      | 1152    | 28     | 16    |
| XXL  | ~1.5B      | 1536    | 32     | 24    |

## Usage

```python
from inl_diffusion import INLVAE, INLDiT, INLDiffusionPipeline

# Load models
vae = INLVAE.load("vae_checkpoints/vae_final.pt")
dit = INLDiT.load("dit_checkpoints/dit_final.pt")

# Create pipeline
pipeline = INLDiffusionPipeline(vae=vae, dit=dit)

# Generate images
images = pipeline(
    prompt="A beautiful sunset over mountains",
    num_inference_steps=50,
    guidance_scale=7.5,
)
```

## License

Apache 2.0

## Citation

```bibtex
@misc{inl-diffusion,
  title={INL-Diffusion: Image Generation with Integrator Neurons},
  author={Pacific Prime},
  year={2025},
  url={https://github.com/Web3-League/inl-diffusion}
}
```
