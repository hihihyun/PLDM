"""
Main Underwater Image Enhancement Diffusion Model
Integrates VAE, Diffusion UNet, and Water-Net preprocessing
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .vae_encoder import EnhancedVAEEncoder, reparameterize
from .vae_decoder import VAEDecoder
from .diffusion_unet import ConditionalDiffusionUNet
from .water_physics import WaterNetPreprocessor
from .loss_functions import CombinedLoss


class UnderwaterEnhancementDiffusion(nn.Module):
    """주석: VAE, Diffusion UNet, Water-Net 전처리를 통합한 메인 모델 클래스"""
    def __init__(self,
                 img_size=256,
                 in_channels=3,
                 latent_channels=4,
                 base_channels=128,
                 time_steps=1000,
                 physics_dim=128,
                 device='cuda'):
        super().__init__()

        self.device = device
        self.img_size = img_size
        self.in_channels = in_channels
        self.latent_channels = latent_channels
        self.time_steps = time_steps

        self.encoder = EnhancedVAEEncoder(in_channels, latent_channels, base_channels)
        self.decoder = VAEDecoder(latent_channels, in_channels, base_channels)
        self.diffusion_unet = ConditionalDiffusionUNet(
            latent_channels, latent_channels, base_channels,
            physics_dim=physics_dim
        )

        self.register_buffer('betas', self._cosine_beta_schedule(time_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        self.loss_fn = CombinedLoss(device=self.device)
        self.preprocessor = WaterNetPreprocessor()
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def encode_to_latent(self, x, preprocessed_x):
        mean, logvar, physics_cond = self.encoder(x, preprocessed_x)
        z = reparameterize(mean, logvar)
        return z, mean, logvar, physics_cond

    def decode_from_latent(self, z):
        return self.decoder(z)

    def forward_diffusion(self, x0, t):
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(0, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).view(-1, 1, 1, 1)
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    @torch.no_grad()
    def sample(self, degraded_img, num_steps=50):
        device = degraded_img.device
        batch_size = degraded_img.shape[0]

        # VAE 입력 범위인 [-1, 1]을 전처리기 입력 범위인 [0, 1]로 변환
        degraded_for_preprocess = (degraded_img + 1.0) / 2.0
        preprocessed = self.preprocessor.preprocess_batch(degraded_for_preprocess)
        
        # 물리 컨디셔닝은 저하된 이미지로부터 추출
        _, _, _, physics_cond = self.encode_to_latent(degraded_img, preprocessed)

        # Assuming latent size based on common downsampling factor of 32 (256 -> 8)
        latent_size = self.img_size // 32 
        z = torch.randn((batch_size, self.latent_channels, latent_size, latent_size), device=device)

        step_size = self.time_steps // num_steps
        timesteps = torch.arange(self.time_steps - 1, -1, -step_size, device=device)[:num_steps]

        for t in timesteps:
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = self.diffusion_unet(z, t_batch, physics_cond)

            alpha_t = self.alphas_cumprod[t]
            alpha_t_prev = self.alphas_cumprod[t - step_size] if t >= step_size else torch.tensor(1.0, device=device)

            pred_x0 = (z - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            pred_x0 = torch.clamp(pred_x0, -3.0, 3.0)

            direction_xt = (1 - alpha_t_prev).sqrt() * predicted_noise
            z = alpha_t_prev.sqrt() * pred_x0 + direction_xt

        enhanced_img = self.decode_from_latent(z)
        return torch.clamp(enhanced_img, -1.0, 1.0)

    # 👇 [수정] preprocessed_img 인자 제거
    def training_step(self, degraded_img, enhanced_img):
        batch_size = degraded_img.shape[0]

        # 👇 [수정] 모델 내부에서 직접 전처리 수행
        enhanced_for_preprocess = (enhanced_img + 1.0) / 2.0
        preprocessed_enhanced = self.preprocessor.preprocess_batch(enhanced_for_preprocess)

        z_enhanced, mean, logvar, physics_cond = self.encode_to_latent(enhanced_img, preprocessed_enhanced)
        
        reconstructed = self.decode_from_latent(z_enhanced)
        recon_loss, recon_loss_dict = self.loss_fn(reconstructed, enhanced_img)
        
        # 👇 [수정] KL 손실을 잠재 공간의 전체 크기로 정규화하여 안정성 확보
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        kl_loss = kl_loss / (batch_size * self.latent_channels * z_enhanced.shape[2] * z_enhanced.shape[3])

        t = torch.randint(0, self.time_steps, (batch_size,), device=degraded_img.device, dtype=torch.long)
        z_noisy, noise = self.forward_diffusion(z_enhanced.detach(), t)
        
        # 물리 컨디셔닝은 저하된 이미지로부터 추출
        degraded_for_preprocess = (degraded_img + 1.0) / 2.0
        preprocessed_degraded = self.preprocessor.preprocess_batch(degraded_for_preprocess)
        _, _, _, physics_cond_degraded = self.encode_to_latent(degraded_img, preprocessed_degraded)
        
        predicted_noise = self.diffusion_unet(z_noisy, t, physics_cond_degraded.detach())
        diffusion_loss = F.mse_loss(predicted_noise, noise)

        kl_weight = 1e-4
        total_loss = recon_loss + (kl_weight * kl_loss) + (0.1 * diffusion_loss)

        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'kl': (kl_weight * kl_loss).item(),
            'diffusion': diffusion_loss.item()
        }
        loss_dict.update(recon_loss_dict)
        return total_loss, loss_dict


def create_model(config, device='cuda'):
    """주석: config와 device 정보를 받아 모델 인스턴스를 생성하는 팩토리 함수"""
    model = UnderwaterEnhancementDiffusion(
        img_size=config.get('img_size', 256),
        in_channels=config.get('in_channels', 3),
        latent_channels=config.get('latent_channels', 4),
        base_channels=config.get('base_channels', 128),
        time_steps=config.get('time_steps', 1000),
        physics_dim=config.get('physics_dim', 128),
        device=device
    )
    return model