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

        # VAE 구성요소
        self.encoder = EnhancedVAEEncoder(in_channels, latent_channels, base_channels)
        self.decoder = VAEDecoder(latent_channels, in_channels, base_channels)

        # Diffusion UNet
        self.diffusion_unet = ConditionalDiffusionUNet(
            latent_channels, latent_channels, base_channels,
            physics_dim=physics_dim
        )

        # Noise schedule (코사인 스케줄)
        self.register_buffer('betas', self._cosine_beta_schedule(time_steps))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - self.alphas_cumprod))

        # 손실 함수
        self.loss_fn = CombinedLoss(device=self.device)
        
        # 전처리기
        self.preprocessor = WaterNetPreprocessor()
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        # 주석: 안정적인 학습을 위해 코사인 노이즈 스케줄을 생성합니다.
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps, dtype=torch.float32)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)

    def encode_to_latent(self, x, preprocessed_x=None):
        # 주석: 이미지를 받아 VAE 인코더를 통과시킨 후, 잠재 변수 z를 계산하여 반환합니다. (총 4개 값 반환)
        mean, logvar, physics_cond = self.encoder(x, preprocessed_x)
        z = reparameterize(mean, logvar)
        return z, mean, logvar, physics_cond

    def decode_from_latent(self, z):
        # 주석: 잠재 변수 z를 받아 VAE 디코더를 통해 이미지로 복원합니다.
        return self.decoder(z)

    def forward_diffusion(self, x0, t):
        # 주석: 원본 잠재 변수 x0에 t 타임스텝만큼 노이즈를 추가합니다.
        noise = torch.randn_like(x0)
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod.gather(0, t).view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod.gather(0, t).view(-1, 1, 1, 1)
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        return xt, noise

    @torch.no_grad()
    def sample(self, degraded_img, num_steps=50):
        # 주석: 훈련된 모델을 사용하여 저하된 이미지를 개선합니다. (추론 과정)
        device = degraded_img.device
        batch_size = degraded_img.shape[0]

        degraded_for_preprocess = (degraded_img * 0.5) + 0.5
        preprocessed = self.preprocessor.preprocess_batch(degraded_for_preprocess)
        
        _, _, _, physics_cond = self.encode_to_latent(degraded_img, preprocessed.to(device))

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
            pred_x0 = torch.clamp(pred_x0, -3, 3)

            direction_xt = (1 - alpha_t_prev).sqrt() * predicted_noise
            z = alpha_t_prev.sqrt() * pred_x0 + direction_xt

        enhanced_img = self.decode_from_latent(z)
        return enhanced_img

    def training_step(self, degraded_img, enhanced_img, preprocessed_img):
        # 주석: 단일 훈련 배치에 대한 손실을 계산합니다.
        batch_size = degraded_img.shape[0]

        # 👇 [수정 완료] self.encoder 대신 self.encode_to_latent를 호출하여 4개의 값을 받습니다.
        z_enhanced, mean, logvar, physics_cond = self.encode_to_latent(enhanced_img, preprocessed_img)
        
        # 1. VAE 관련 손실
        reconstructed = self.decode_from_latent(z_enhanced)
        recon_loss, recon_loss_dict = self.loss_fn(reconstructed, enhanced_img)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - (logvar + 1e-8).exp()) / batch_size

        # 2. Diffusion 관련 손실
        t = torch.randint(0, self.time_steps, (batch_size,), device=degraded_img.device, dtype=torch.long)
        z_noisy, noise = self.forward_diffusion(z_enhanced.detach(), t)
        predicted_noise = self.diffusion_unet(z_noisy, t, physics_cond.detach())
        diffusion_loss = F.mse_loss(predicted_noise, noise)

        # 3. 최종 손실 (가중치 조절로 학습 안정화)
        total_loss = recon_loss + (1e-6 * kl_loss) + (0.1 * diffusion_loss)

        loss_dict = {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'kl': kl_loss.item(),
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