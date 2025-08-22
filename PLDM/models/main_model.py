"""
Main Underwater Image Enhancement Diffusion Model
Integrates VAE, Diffusion UNet, and Water-Net preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# --- 의존하는 다른 모델 파일들을 먼저 정의해야 합니다 ---
# 아래 코드를 실행하기 전에 vae_encoder.py, vae_decoder.py, diffusion_unet.py,
# water_physics.py, loss_functions.py 파일이 올바르게 정의되어 있어야 합니다.
# 여기서는 설명을 위해 하나의 파일에 통합된 형태로 가정합니다.
# 실제로는 각 파일을 올바르게 import 해야 합니다.

from .vae_encoder import EnhancedVAEEncoder, reparameterize
from .vae_decoder import VAEDecoder
from .diffusion_unet import ConditionalDiffusionUNet
from .water_physics import WaterNetPreprocessor
from .loss_functions import CombinedLoss


class UnderwaterEnhancementDiffusion(nn.Module):
    """Complete Underwater Image Enhancement Diffusion Model"""
    def __init__(self,
                 img_size=256,
                 in_channels=3,
                 latent_channels=4,
                 base_channels=128,
                 time_steps=1000,
                 physics_dim=128):
        super().__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.time_steps = time_steps

        # VAE components
        self.encoder = EnhancedVAEEncoder(in_channels, latent_channels, base_channels)
        self.decoder = VAEDecoder(latent_channels, in_channels, base_channels)

        # Diffusion UNet
        self.diffusion_unet = ConditionalDiffusionUNet(
            latent_channels, latent_channels, base_channels,
            physics_dim=physics_dim
        )

        # Noise schedule (cosine schedule for better stability)
        self.register_buffer('betas', self._cosine_beta_schedule(time_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        # Loss function
        self.loss_fn = CombinedLoss()

        # Preprocessor
        self.preprocessor = WaterNetPreprocessor()

    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule for better stability"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def encode_to_latent(self, x, preprocessed_x=None):
        """Encode image to latent space with physics conditioning"""
        mean, logvar, physics_cond = self.encoder(x, preprocessed_x)
        z = reparameterize(mean, logvar)
        return z, mean, logvar, physics_cond

    def decode_from_latent(self, z):
        """Decode latent to image space"""
        return self.decoder(z)

    def forward_diffusion(self, x0, t):
        """Add noise for training"""
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(0, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).view(-1, 1, 1, 1)
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    @torch.no_grad()
    def sample(self, degraded_img, num_steps=50):
        """Generate enhanced image through reverse diffusion"""
        device = degraded_img.device
        batch_size = degraded_img.shape[0]

        preprocessed = self.preprocessor.preprocess_batch(degraded_img)
        _, _, _, physics_cond = self.encode_to_latent(degraded_img, preprocessed)

        z = torch.randn((batch_size, self.latent_channels, self.img_size // 8, self.img_size // 8), device=device)

        step_size = self.time_steps // num_steps
        timesteps = torch.arange(self.time_steps - 1, -1, -step_size, device=device)[:num_steps]

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = self.diffusion_unet(z, t_batch, physics_cond)

            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t - step_size] if t >= step_size else torch.tensor(1.0, device=device)

            pred_x0 = (z - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            pred_x0 = torch.clamp(pred_x0, -3, 3)

            direction_xt = (1 - alpha_t_prev).sqrt() * predicted_noise
            z = alpha_t_prev.sqrt() * pred_x0 + direction_xt

        enhanced_img = self.decode_from_latent(z)
        return enhanced_img

    def training_step(self, degraded_img, enhanced_img, preprocessed_img):
        """Single training step"""
        batch_size = degraded_img.shape[0]
        device = degraded_img.device

        # 1. VAE-related loss
        z_enhanced, mean, logvar, physics_cond = self.encode_to_latent(enhanced_img, preprocessed_img)
        reconstructed = self.decode_from_latent(z_enhanced)
        recon_loss, _ = self.loss_fn(reconstructed, enhanced_img)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size

        # 2. Diffusion-related loss
        t = torch.randint(0, self.time_steps, (batch_size,), device=device, dtype=torch.long)
        z_noisy, noise = self.forward_diffusion(z_enhanced.detach(), t)
        predicted_noise = self.diffusion_unet(z_noisy, t, physics_cond.detach())
        diffusion_loss = F.mse_loss(predicted_noise, noise)

        # 3. Total Loss
        total_loss = recon_loss + 0.0001 * kl_loss + diffusion_loss

        return total_loss, {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'kl': kl_loss.item(),
            'diffusion': diffusion_loss.item()
        }

def create_model(config):
    """Factory function to create model from config"""
    model = UnderwaterEnhancementDiffusion(
        img_size=config.get('img_size', 256),
        in_channels=config.get('in_channels', 3),
        latent_channels=config.get('latent_channels', 4),
        base_channels=config.get('base_channels', 128),
        time_steps=config.get('time_steps', 1000),
        physics_dim=config.get('physics_dim', 128)
    )
    return model