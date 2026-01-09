"""
INL Diffusion Pipeline for Text-to-Image Generation

Complete pipeline that combines:
- INL-LLM for text encoding
- INL-VAE for image encoding/decoding
- INL-DiT for diffusion denoising
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Union, Callable
from tqdm import tqdm
import math


class DDPMScheduler:
    """
    DDPM (Denoising Diffusion Probabilistic Models) noise scheduler.

    Handles the forward (noising) and reverse (denoising) diffusion process.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        # Create beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)

        # Calculations for diffusion q(x_t | x_0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        self.posterior_variance = betas * (1.0 - self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_log_variance_clipped = torch.log(
            torch.cat([self.posterior_variance[1:2], self.posterior_variance[1:]])
        )
        self.posterior_mean_coef1 = betas * torch.sqrt(self.alphas_cumprod_prev) / (1.0 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1.0 - self.alphas_cumprod_prev) * torch.sqrt(self.alphas) / (1.0 - self.alphas_cumprod)

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples (forward diffusion)."""
        # Index on CPU then move to device
        timesteps_cpu = timesteps.cpu()
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps_cpu].view(-1, 1, 1, 1).to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps_cpu].view(-1, 1, 1, 1).to(original_samples.device)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """Denoise step (reverse diffusion)."""
        t = timestep

        # Predict x_0
        if self.prediction_type == "epsilon":
            alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)

            pred_original_sample = (sample - sqrt_one_minus_alpha_prod_t * model_output) / sqrt_alpha_prod_t
        elif self.prediction_type == "v_prediction":
            alpha_prod_t = self.alphas_cumprod[t].to(sample.device)
            sqrt_alpha_prod_t = torch.sqrt(alpha_prod_t)
            sqrt_one_minus_alpha_prod_t = torch.sqrt(1 - alpha_prod_t)

            pred_original_sample = sqrt_alpha_prod_t * sample - sqrt_one_minus_alpha_prod_t * model_output
        else:
            pred_original_sample = model_output

        # Clip prediction
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # Compute posterior mean
        posterior_mean = (
            self.posterior_mean_coef1[t].to(sample.device) * pred_original_sample
            + self.posterior_mean_coef2[t].to(sample.device) * sample
        )

        # Add noise for all steps except the last
        if t > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            variance = self.posterior_variance[t].to(sample.device)
            prev_sample = posterior_mean + torch.sqrt(variance) * noise
        else:
            prev_sample = posterior_mean

        return prev_sample


class DDIMScheduler:
    """
    DDIM (Denoising Diffusion Implicit Models) scheduler for faster sampling.
    """

    def __init__(
        self,
        num_train_timesteps: int = 1000,
        beta_start: float = 0.0001,
        beta_end: float = 0.02,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
    ):
        self.num_train_timesteps = num_train_timesteps
        self.prediction_type = prediction_type

        # Create beta schedule
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_train_timesteps)
        elif beta_schedule == "cosine":
            steps = num_train_timesteps + 1
            x = torch.linspace(0, num_train_timesteps, steps)
            alphas_cumprod = torch.cos(((x / num_train_timesteps) + 0.008) / 1.008 * math.pi / 2) ** 2
            alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
            betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
            betas = torch.clip(betas, 0.0001, 0.9999)
        elif beta_schedule == "scaled_linear":
            betas = torch.linspace(beta_start**0.5, beta_end**0.5, num_train_timesteps) ** 2
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")

        self.betas = betas
        self.alphas = 1.0 - betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

        self.timesteps = None

    def set_timesteps(self, num_inference_steps: int):
        """Set timesteps for inference."""
        step_ratio = self.num_train_timesteps // num_inference_steps
        timesteps = (torch.arange(0, num_inference_steps) * step_ratio).long()
        timesteps = torch.flip(timesteps, [0])
        self.timesteps = timesteps

    def add_noise(
        self,
        original_samples: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
    ) -> torch.Tensor:
        """Add noise to samples."""
        sqrt_alpha_prod = self.sqrt_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(original_samples.device)
        sqrt_one_minus_alpha_prod = self.sqrt_one_minus_alphas_cumprod[timesteps].view(-1, 1, 1, 1).to(original_samples.device)

        noisy_samples = sqrt_alpha_prod * original_samples + sqrt_one_minus_alpha_prod * noise
        return noisy_samples

    def step(
        self,
        model_output: torch.Tensor,
        timestep: int,
        sample: torch.Tensor,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """DDIM denoising step."""
        # Get current and previous timestep
        prev_timestep = timestep - self.num_train_timesteps // len(self.timesteps)
        prev_timestep = max(prev_timestep, 0)

        alpha_prod_t = self.alphas_cumprod[timestep].to(sample.device)
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep].to(sample.device) if prev_timestep >= 0 else torch.tensor(1.0)

        # Predict x_0
        if self.prediction_type == "epsilon":
            pred_original_sample = (sample - torch.sqrt(1 - alpha_prod_t) * model_output) / torch.sqrt(alpha_prod_t)
        elif self.prediction_type == "v_prediction":
            pred_original_sample = torch.sqrt(alpha_prod_t) * sample - torch.sqrt(1 - alpha_prod_t) * model_output
        else:
            pred_original_sample = model_output

        # Clip prediction
        pred_original_sample = torch.clamp(pred_original_sample, -1, 1)

        # Compute variance
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)
        std_dev_t = eta * torch.sqrt(variance)

        # Direction pointing to x_t
        pred_sample_direction = torch.sqrt(1 - alpha_prod_t_prev - std_dev_t**2) * model_output

        # Compute x_{t-1}
        prev_sample = torch.sqrt(alpha_prod_t_prev) * pred_original_sample + pred_sample_direction

        if eta > 0:
            noise = torch.randn(sample.shape, generator=generator, device=sample.device, dtype=sample.dtype)
            prev_sample = prev_sample + std_dev_t * noise

        return prev_sample


class INLDiffusionPipeline:
    """
    Complete text-to-image generation pipeline.

    Combines INL-LLM (text encoder), INL-VAE (image codec), and INL-DiT (diffusion).
    """

    def __init__(
        self,
        vae: nn.Module,
        dit: nn.Module,
        text_encoder: Optional[nn.Module] = None,
        tokenizer: Optional[object] = None,
        scheduler: Optional[object] = None,
        device: str = "cuda",
    ):
        """
        Initialize pipeline.

        Args:
            vae: INL-VAE for encoding/decoding images
            dit: INL-DiT for diffusion
            text_encoder: Text encoder (INL-LLM or CLIP)
            tokenizer: Tokenizer for text
            scheduler: Noise scheduler (DDPM or DDIM)
            device: Device to run on
        """
        self.vae = vae
        self.dit = dit
        self.text_encoder = text_encoder
        self.tokenizer = tokenizer
        self.scheduler = scheduler or DDIMScheduler()
        self.device = device

        # Move models to device
        self.vae.to(device)
        self.dit.to(device)
        if self.text_encoder is not None:
            self.text_encoder.to(device)

    @torch.no_grad()
    def encode_prompt(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = None,
        max_length: int = 77,
    ) -> torch.Tensor:
        """
        Encode text prompt to embeddings.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s) for classifier-free guidance
            max_length: Maximum sequence length

        Returns:
            Text embeddings [B, S, D] or [2*B, S, D] with negative prompts
        """
        if isinstance(prompt, str):
            prompt = [prompt]

        batch_size = len(prompt)

        if self.text_encoder is None or self.tokenizer is None:
            # Return dummy embeddings if no text encoder
            return torch.zeros(batch_size, max_length, self.dit.d_model, device=self.device)

        # Tokenize
        tokens = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(self.device)

        # Encode
        text_embeddings = self.text_encoder(input_ids)

        # Handle negative prompts for classifier-free guidance
        if negative_prompt is not None:
            if isinstance(negative_prompt, str):
                negative_prompt = [negative_prompt] * batch_size

            neg_tokens = self.tokenizer(
                negative_prompt,
                padding="max_length",
                max_length=max_length,
                truncation=True,
                return_tensors="pt",
            )
            neg_input_ids = neg_tokens["input_ids"].to(self.device)
            neg_embeddings = self.text_encoder(neg_input_ids)

            # Concatenate [negative, positive]
            text_embeddings = torch.cat([neg_embeddings, text_embeddings], dim=0)

        return text_embeddings

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[str, List[str]],
        negative_prompt: Optional[Union[str, List[str]]] = "",
        height: int = 256,
        width: int = 256,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        num_images_per_prompt: int = 1,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.Tensor] = None,
        callback: Optional[Callable[[int, int, torch.Tensor], None]] = None,
        callback_steps: int = 1,
    ) -> List[torch.Tensor]:
        """
        Generate images from text prompts.

        Args:
            prompt: Text prompt(s)
            negative_prompt: Negative prompt(s)
            height: Image height
            width: Image width
            num_inference_steps: Number of denoising steps
            guidance_scale: Classifier-free guidance scale
            num_images_per_prompt: Number of images per prompt
            generator: Random number generator
            latents: Initial latents (optional)
            callback: Progress callback
            callback_steps: Call callback every N steps

        Returns:
            List of generated images as tensors
        """
        if isinstance(prompt, str):
            prompt = [prompt]
        batch_size = len(prompt)

        # Encode prompts
        do_classifier_free_guidance = guidance_scale > 1.0
        text_embeddings = self.encode_prompt(
            prompt,
            negative_prompt=negative_prompt if do_classifier_free_guidance else None,
        )

        # Repeat for multiple images per prompt
        if num_images_per_prompt > 1:
            text_embeddings = text_embeddings.repeat_interleave(num_images_per_prompt, dim=0)

        actual_batch_size = batch_size * num_images_per_prompt

        # Get latent dimensions
        latent_height = height // 8
        latent_width = width // 8
        latent_channels = self.vae.latent_dim

        # Initialize latents
        if latents is None:
            latents = torch.randn(
                (actual_batch_size, latent_channels, latent_height, latent_width),
                generator=generator,
                device=self.device,
                dtype=torch.float32,
            )

        # Set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps.to(self.device)

        # Scale initial noise
        latents = latents * self.scheduler.sqrt_one_minus_alphas_cumprod[timesteps[0]]

        # Denoising loop
        for i, t in enumerate(tqdm(timesteps, desc="Generating")):
            # Expand latents for classifier-free guidance
            if do_classifier_free_guidance:
                latent_model_input = torch.cat([latents] * 2)
                timestep = torch.cat([t.unsqueeze(0)] * (actual_batch_size * 2))
            else:
                latent_model_input = latents
                timestep = t.unsqueeze(0).expand(actual_batch_size)

            # Predict noise
            noise_pred = self.dit(latent_model_input, timestep, text_embeddings)

            # Classifier-free guidance
            if do_classifier_free_guidance:
                noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

            # Denoise step
            latents = self.scheduler.step(noise_pred, t.item(), latents, generator=generator)

            # Callback
            if callback is not None and i % callback_steps == 0:
                callback(i, len(timesteps), latents)

        # Decode latents to images
        images = self.vae.detokenize(latents)

        # Clamp to valid range
        images = torch.clamp(images, 0, 1)

        return images

    def to(self, device: str):
        """Move pipeline to device."""
        self.device = device
        self.vae.to(device)
        self.dit.to(device)
        if self.text_encoder is not None:
            self.text_encoder.to(device)
        return self

    @classmethod
    def from_pretrained(
        cls,
        vae_path: str,
        dit_path: str,
        tokenizer_path: Optional[str] = None,
        device: str = "cuda",
        scheduler_type: str = "ddim",
    ) -> "INLDiffusionPipeline":
        """
        Load pipeline from pretrained checkpoints.

        Args:
            vae_path: Path to VAE checkpoint (.pt or .safetensors)
            dit_path: Path to DiT checkpoint (.pt or .safetensors)
            tokenizer_path: Path to INL tokenizer.json (optional)
            device: Device to run on
            scheduler_type: "ddpm" or "ddim"

        Returns:
            Loaded pipeline ready for inference
        """
        from ..vae import INLVAE
        from ..dit import INLDiT
        from ..tokenizer import INLTokenizer

        # Load VAE
        print(f"Loading VAE from {vae_path}...")
        vae_checkpoint = torch.load(vae_path, map_location="cpu")
        vae_config = vae_checkpoint.get("config", {})

        vae = INLVAE(
            image_size=vae_config.get("image_size", 256),
            base_channels=vae_config.get("base_channels", 128),
            latent_dim=vae_config.get("latent_dim", 4),
        )
        vae.load_state_dict(vae_checkpoint["model_state_dict"])
        vae.eval()

        # Load DiT
        print(f"Loading DiT from {dit_path}...")
        dit_checkpoint = torch.load(dit_path, map_location="cpu")
        dit_config = dit_checkpoint.get("config", {})

        dit = INLDiT.from_config(
            dit_config.get("dit_size", "L"),
            img_size=dit_config.get("img_size", 32),
            patch_size=2,
            in_channels=dit_config.get("latent_channels", 4),
            context_dim=2048,
        )
        dit.load_state_dict(dit_checkpoint["model_state_dict"])
        dit.eval()

        # Load tokenizer
        print(f"Loading INL tokenizer...")
        tokenizer = INLTokenizer(tokenizer_path=tokenizer_path)
        vocab_size = tokenizer.get_vocab_size()
        print(f"Tokenizer vocab size: {vocab_size:,}")

        # Create text encoder
        # Import here to avoid circular imports
        import torch.nn as nn

        class INLTextEncoder(nn.Module):
            def __init__(self, vocab_size: int, d_model: int = 2048, num_layers: int = 6):
                super().__init__()
                self.d_model = d_model
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_embedding = nn.Parameter(torch.randn(1, 77, d_model) * 0.02)
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model=d_model, nhead=8, dim_feedforward=d_model * 4,
                    batch_first=True, dropout=0.1
                )
                self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
                self.ln_final = nn.LayerNorm(d_model)

            def forward(self, input_ids):
                x = self.embedding(input_ids)
                x = x + self.pos_embedding[:, :x.size(1)]
                x = self.encoder(x)
                return self.ln_final(x)

        text_encoder = INLTextEncoder(vocab_size=vocab_size)

        # Create scheduler
        if scheduler_type == "ddpm":
            scheduler = DDPMScheduler(
                num_train_timesteps=1000,
                beta_schedule="scaled_linear",
                prediction_type="epsilon",
            )
        else:
            scheduler = DDIMScheduler(
                num_train_timesteps=1000,
                beta_schedule="scaled_linear",
                prediction_type="epsilon",
            )

        # Create pipeline
        pipeline = cls(
            vae=vae,
            dit=dit,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            device=device,
        )

        print(f"Pipeline loaded on {device}")
        return pipeline
