# INL-Diffusion

**Text-to-Image Generation with Integrator Neurons**

[![PyPI](https://img.shields.io/pypi/v/inl-diffusion)](https://pypi.org/project/inl-diffusion/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

INL-Diffusion is a custom text-to-image diffusion model built from scratch, featuring **Integrator Neurons** for adaptive computation. The architecture combines a custom VAE for image tokenization with a Diffusion Transformer (DiT) for generation.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        INL-Diffusion                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐         │
│  │  INL-VAE    │    │  INL-DiT    │    │   Text      │         │
│  │  (Encoder)  │───▶│ (Denoiser)  │◀───│  Encoder    │         │
│  │             │    │             │    │             │         │
│  │ 256x256x3   │    │ Transformer │    │ INL-LLM or  │         │
│  │    ↓        │    │ + Integrator│    │   CLIP      │         │
│  │ 32x32x4     │    │   Neurons   │    │             │         │
│  └─────────────┘    └─────────────┘    └─────────────┘         │
│         │                  │                                    │
│         ▼                  ▼                                    │
│  ┌─────────────┐    ┌─────────────┐                            │
│  │  INL-VAE    │    │  Generated  │                            │
│  │  (Decoder)  │◀───│   Latent    │                            │
│  │             │    │             │                            │
│  │ 32x32x4     │    │  32x32x4    │                            │
│  │    ↓        │    │             │                            │
│  │ 256x256x3   │    └─────────────┘                            │
│  └─────────────┘                                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Key Features

### INL-VAE (Image Tokenizer)
- **Custom architecture** - No dependency on external VAEs (Stable Diffusion, etc.)
- **KL-regularized** latent space for stable training
- **8x compression** - 256x256x3 → 32x32x4 latents
- **Residual blocks** with GroupNorm and Swish activation
- **Self-attention** at bottleneck resolution

### INL-DiT (Diffusion Transformer)
- **Integrator Neurons** - Adaptive computation for complex patches
- **GQA (Grouped Query Attention)** - Efficient attention with fewer KV heads
- **2D RoPE** - Rotary position embeddings for image patches
- **AdaLN-Zero** - Timestep conditioning via adaptive layer norm
- **Cross-attention** - Text conditioning from encoder

### Model Sizes

| Config | Parameters | Layers | d_model | Heads | KV Heads |
|--------|------------|--------|---------|-------|----------|
| S      | ~100M      | 12     | 384     | 6     | 2        |
| B      | ~250M      | 12     | 768     | 12    | 4        |
| L      | ~500M      | 24     | 1024    | 16    | 4        |
| XL     | ~700M      | 28     | 1152    | 16    | 4        |
| XXL    | ~1.5B      | 32     | 1536    | 24    | 6        |

## Installation

```bash
pip install inl-diffusion
```

Or install from source:

```bash
git clone https://github.com/Web3-League/inl-diffusion.git
cd inl-diffusion
pip install -e ".[train]"
```

## Quick Start

### Inference

```python
from inl_diffusion import INLVAE, INLDiT, INLDiffusionPipeline

# Load models
vae = INLVAE.load("vae_checkpoints/vae_final.pt")
dit = INLDiT.load("dit_checkpoints/dit_final.pt")

# Create pipeline
pipeline = INLDiffusionPipeline(vae=vae, dit=dit)

# Generate image
image = pipeline(
    prompt="A beautiful sunset over mountains, oil painting style",
    num_inference_steps=50,
    guidance_scale=7.5,
)

image.save("output.png")
```

## Training

### Step 1: Train the VAE (Image Tokenizer)

The VAE compresses images to a latent space. Train on any image dataset:

```bash
python train_vae.py --dataset cifar10 --epochs 100
```

Or with custom settings:

```bash
python train_vae.py \
    --dataset_path /path/to/images \
    --image_size 256 \
    --batch_size 32 \
    --max_steps 50000
```

**Expected VAE training:**
- ~15K steps for good reconstruction
- Loss ~0.07-0.08 (total), recon ~0.01, KL ~8.0

### Step 2: Train the DiT (Diffusion Model)

First, download your dataset locally for faster training:

```bash
python download_dataset.py --dataset huggan/wikiart --output /workspace/wikiart_local
```

Then train:

```bash
GRADIENT_ACCUMULATION=1 SAVE_EVERY=10000 python train_dit.py \
    --vae_path vae_checkpoints/vae_step_15000.pt \
    --local_dataset /workspace/wikiart_local \
    --dit_size XL \
    --batch_size 64 \
    --max_steps 50000 \
    --lr 1e-4
```

### Training Configuration

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--vae_path` | Path to trained VAE checkpoint | Required |
| `--dataset` | HuggingFace dataset name | `laion/laion-art` |
| `--local_dataset` | Pre-downloaded dataset path | None |
| `--dit_size` | Model size (S/B/L/XL/XXL) | `L` |
| `--batch_size` | Training batch size | 16 |
| `--max_steps` | Maximum training steps | 500000 |
| `--lr` | Learning rate | 1e-4 |

**Environment variables:**

```bash
GRADIENT_ACCUMULATION=4   # Effective batch = batch_size * grad_accum
SAVE_EVERY=10000         # Checkpoint frequency
LOG_INTERVAL=100         # TensorBoard logging frequency
IMAGE_SIZE=256           # Training image resolution
WARMUP_STEPS=5000        # LR warmup steps
```

## Project Structure

```
inl-diffusion/
├── inl_diffusion/
│   ├── __init__.py
│   ├── vae/
│   │   ├── __init__.py
│   │   └── inl_vae.py          # VAE architecture
│   ├── dit/
│   │   ├── __init__.py
│   │   └── inl_dit.py          # DiT architecture
│   └── pipeline/
│       ├── __init__.py
│       └── text_to_image.py    # Inference pipeline & schedulers
├── train_vae.py                # VAE training script
├── train_dit.py                # DiT training script
├── download_dataset.py         # Dataset download utility
├── convert_to_safetensors.py   # Model conversion
└── setup.py
```

## Integrator Neurons

The key innovation in INL-Diffusion is the **Integrator Neuron** module, which allows the model to allocate more computation to complex image patches:

```python
class IntegratorNeuron(nn.Module):
    """
    Integrator Neuron for adaptive computation in diffusion.
    Allows the model to allocate more computation to complex patches.
    """

    def __init__(self, d_model: int, num_iterations: int = 2):
        super().__init__()
        self.d_model = d_model
        self.num_iterations = num_iterations

        # Gating mechanism to decide iteration count
        self.gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Sigmoid(),
        )

        # Refinement MLP
        self.refine = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
        )
```

This is inspired by [INL-LLM v3](https://github.com/Web3-League/inl-llm-v3), adapted for diffusion models.

## Performance

Training on NVIDIA H200 (141GB VRAM):

| Model | Batch Size | Speed | Memory |
|-------|------------|-------|--------|
| DiT-S | 128 | ~0.3s/it | ~20GB |
| DiT-B | 64 | ~0.5s/it | ~35GB |
| DiT-L | 64 | ~1.2s/it | ~55GB |
| DiT-XL | 64 | ~2.5s/it | ~80GB |
| DiT-XXL | 32 | ~4.0s/it | ~120GB |

## Monitoring

TensorBoard logs are saved during training:

```bash
tensorboard --logdir dit_checkpoints/runs/
```

**Metrics tracked:**
- `train/loss` - MSE loss on noise prediction
- `train/lr` - Learning rate (with warmup)
- `train/grad_norm` - Gradient norm (clipped to 1.0)
- `samples/generated` - Sample images every N steps

## Datasets

### VAE Training (images only)

| Dataset | Size | Link |
|---------|------|------|
| ImageNet-1K | 1.2M | [huggingface.co/datasets/imagenet-1k](https://huggingface.co/datasets/imagenet-1k) |
| WikiArt | 81K | [huggingface.co/datasets/huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart) |
| CelebA-HQ | 30K | [huggingface.co/datasets/huggan/CelebA-HQ](https://huggingface.co/datasets/huggan/CelebA-HQ) |
| FFHQ | 70K | [huggingface.co/datasets/huggan/FFHQ](https://huggingface.co/datasets/huggan/FFHQ) |

### DiT Training (images + captions)

| Dataset | Size | Link |
|---------|------|------|
| LAION-Art | 8M | [huggingface.co/datasets/laion/laion-art](https://huggingface.co/datasets/laion/laion-art) |
| LAION-Aesthetics | 120M | [huggingface.co/datasets/laion/laion2B-en-aesthetic](https://huggingface.co/datasets/laion/laion2B-en-aesthetic) |
| Conceptual Captions | 3M | [huggingface.co/datasets/conceptual_captions](https://huggingface.co/datasets/conceptual_captions) |
| DiffusionDB | 14M | [huggingface.co/datasets/poloclub/diffusiondb](https://huggingface.co/datasets/poloclub/diffusiondb) |
| JourneyDB | 4M | [huggingface.co/datasets/JourneyDB/JourneyDB](https://huggingface.co/datasets/JourneyDB/JourneyDB) |

### Large-Scale

| Dataset | Size | Link |
|---------|------|------|
| LAION-2B | 2B | [huggingface.co/datasets/laion/laion2B-en](https://huggingface.co/datasets/laion/laion2B-en) |
| LAION-5B | 5B | [laion.ai/blog/laion-5b](https://laion.ai/blog/laion-5b/) |
| DataComp | 12B | [github.com/mlfoundations/datacomp](https://github.com/mlfoundations/datacomp) |

## Roadmap

- [x] Custom VAE architecture
- [x] DiT with Integrator Neurons
- [x] Text conditioning (cross-attention)
- [x] Multi-size model configs (S/B/L/XL/XXL)
- [x] Local dataset support for fast training
- [x] TensorBoard logging with samples
- [ ] bf16 mixed precision training
- [ ] Multi-GPU training (DDP/FSDP)
- [ ] Classifier-free guidance
- [ ] LoRA fine-tuning support
- [ ] SDXL-style dual text encoders
- [ ] ControlNet support

## Citation

```bibtex
@software{inl_diffusion,
  title = {INL-Diffusion: Text-to-Image Generation with Integrator Neurons},
  author = {Pacific Prime},
  year = {2025},
  url = {https://github.com/Web3-League/inl-diffusion}
}
```

## License

Apache 2.0 License - see [LICENSE](LICENSE) for details.

## Related Projects

- [INL-LLM v3](https://github.com/Web3-League/inl-llm-v3) - Language model with Integrator Neurons
- [DiT](https://github.com/facebookresearch/DiT) - Original Diffusion Transformer paper
