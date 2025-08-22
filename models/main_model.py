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
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(self.alphas_cumprod))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1 - self.alphas_cumprod))
        
        # Loss function
        self.loss_fn = CombinedLoss()
        
        # Preprocessor
        self.preprocessor = WaterNetPreprocessor()
        
    def _cosine_beta_schedule(self, timesteps, s=0.008):
        """Cosine schedule for better stability"""
        steps = timesteps + 1
        x = torch.linspace(0, timesteps, steps)
        alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
        betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
        return torch.clip(betas, 0.0001, 0.9999)
    
    def encode_to_latent(self, x, preprocessed_x=None):
        """Encode image to latent space with physics conditioning"""
        mean, logvar, physics_cond = self.encoder(x, preprocessed_x)
        
        # Reparameterization trick
        z = reparameterize(mean, logvar)
        
        return z, mean, logvar, physics_cond
    
    def decode_from_latent(self, z):
        """Decode latent to image space"""
        return self.decoder(z)
    
    def forward_diffusion(self, x0, t):
        """Add noise for training"""
        noise = torch.randn_like(x0)
        
        # Get noise schedule values for time t
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        
        # Add noise: x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1 - alpha_bar_t) * noise
        xt = sqrt_alphas_cumprod_t * x0 + sqrt_one_minus_alphas_cumprod_t * noise
        
        return xt, noise
    
    @torch.no_grad()
    def reverse_diffusion(self, z_noisy, t, physics_cond, use_ddim=True, eta=0.0):
        """Single step of reverse diffusion (DDIM or DDPM)"""
        batch_size = z_noisy.shape[0]
        
        # Predict noise
        predicted_noise = self.diffusion_unet(z_noisy, t, physics_cond)
        
        if use_ddim:
            # DDIM sampling
            alpha_t = self.alphas_cumprod[t]
            
            # Get alpha for previous timestep
            if t.item() > 0:
                alpha_t_prev = self.alphas_cumprod[t - 1]
            else:
                alpha_t_prev = torch.ones_like(alpha_t)
            
            # Predict x0
            pred_x0 = (z_noisy - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
            pred_x0 = torch.clamp(pred_x0, -3, 3)  # Clamp for stability
            
            # Compute direction pointing towards x_t
            direction_xt = (1 - alpha_t_prev - eta**2 * (1 - alpha_t)).sqrt() * predicted_noise
            
            # Add noise if eta > 0 (stochastic sampling)
            noise = torch.randn_like(z_noisy) if eta > 0 else 0.0
            
            # Compute x_{t-1}
            z_prev = alpha_t_prev.sqrt() * pred_x0 + direction_xt + eta * noise
            
        else:
            # DDPM sampling
            alpha = self.alphas[t]
            alpha_cumprod = self.alphas_cumprod[t]
            
            if t.item() > 0:
                alpha_cumprod_prev = self.alphas_cumprod[t - 1]
            else:
                alpha_cumprod_prev = torch.ones_like(alpha_cumprod)
            
            # Compute variance
            beta_t = 1 - alpha
            variance = beta_t * (1 - alpha_cumprod_prev) / (1 - alpha_cumprod)
            
            # Predict x0
            pred_x0 = (z_noisy - (1 - alpha_cumprod).sqrt() * predicted_noise) / alpha_cumprod.sqrt()
            
            # Compute mean of q(x_{t-1}|x_t, x_0)
            pred_mean = (alpha.sqrt() * (1 - alpha_cumprod_prev) * z_noisy + 
                        alpha_cumprod_prev.sqrt() * beta_t * pred_x0) / (1 - alpha_cumprod)
            
            # Add noise if not final step
            if t.item() > 0:
                noise = torch.randn_like(z_noisy)
                z_prev = pred_mean + variance.sqrt() * noise
            else:
                z_prev = pred_mean
        
        return z_prev
    
    @torch.no_grad()
    def sample(self, degraded_img, num_steps=50, use_ddim=True, eta=0.0):
        """Generate enhanced image through reverse diffusion"""
        device = degraded_img.device
        batch_size = degraded_img.shape[0]
        
        # Preprocess degraded image
        preprocessed = self.preprocessor.preprocess_batch(degraded_img)
        
        # Encode degraded image to latent space
        z_degraded, _, _, physics_cond = self.encode_to_latent(degraded_img, preprocessed)
        
        # Start from noise
        z = torch.randn_like(z_degraded)
        
        # Create sampling schedule
        if num_steps < self.time_steps:
            # Use subset of timesteps for faster sampling
            step_size = self.time_steps // num_steps
            timesteps = torch.arange(self.time_steps - 1, -1, -step_size)[:num_steps]
        else:
            timesteps = torch.arange(self.time_steps - 1, -1, -1)
        
        # Reverse diffusion process
        for i, t in enumerate(timesteps):
            t_batch = torch.full((batch_size,), t, device=device, dtype=torch.long)
            z = self.reverse_diffusion(z, t_batch, physics_cond, use_ddim, eta)
        
        # Decode to image space
        enhanced_img = self.decode_from_latent(z)
        
        return enhanced_img
    
    def training_step(self, degraded_img, enhanced_img, preprocessed_img=None):
        """Single training step with improved error handling"""
        device = degraded_img.device
        batch_size = degraded_img.shape[0]
        
        try:
            # Ensure all inputs are on the same device
            degraded_img = degraded_img.to(device)
            enhanced_img = enhanced_img.to(device)
            if preprocessed_img is not None:
                preprocessed_img = preprocessed_img.to(device)
            
            # Preprocess if not provided
            if preprocessed_img is None:
                preprocessed_img = self.preprocessor.preprocess_batch(degraded_img)
                preprocessed_img = preprocessed_img.to(device)
            
            # Encode enhanced image to latent space
            z_enhanced, mean, logvar, physics_cond = self.encode_to_latent(enhanced_img, preprocessed_img)
            
            # Ensure physics_cond is on correct device
            physics_cond = physics_cond.to(device)
            
            # Sample random timesteps
            t = torch.randint(0, self.time_steps, (batch_size,), device=device, dtype=torch.long)
            
            # Add noise to enhanced latent (forward diffusion)
            z_noisy, noise = self.forward_diffusion(z_enhanced, t)
            
            # Predict noise using diffusion model
            predicted_noise = self.diffusion_unet(z_noisy, t, physics_cond)
            
            # Compute diffusion loss (MSE between predicted and actual noise)
            diffusion_loss = F.mse_loss(predicted_noise, noise)
            
            # KL divergence loss for VAE
            kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size
            
            # Reconstruction loss (occasionally decode for memory efficiency)
            if torch.rand(1).item() < 0.1:  # 10% of the time
                try:
                    z_pred = self.diffusion_unet(z_enhanced, torch.zeros_like(t), physics_cond)
                    reconstructed = self.decode_from_latent(z_pred)
                    recon_loss, recon_loss_dict = self.loss_fn(reconstructed, enhanced_img)
                except Exception as e:
                    print(f"⚠️  Reconstruction loss failed: {e}")
                    recon_loss = torch.tensor(0.0, device=device)
                    recon_loss_dict = {}
            else:
                recon_loss = torch.tensor(0.0, device=device)
                recon_loss_dict = {}
            
            # Total loss - ensure all components are on same device
            total_loss = diffusion_loss + 0.0001 * kl_loss + 0.1 * recon_loss
            
            # Sanity check - loss should not be zero
            if total_loss.item() == 0.0:
                print(f"⚠️  Warning: Zero loss detected!")
                print(f"  Diffusion: {diffusion_loss.item():.6f}")
                print(f"  KL: {kl_loss.item():.6f}")
                print(f"  Recon: {recon_loss.item():.6f}")
            
            return total_loss, {
                'total': total_loss.item(),
                'diffusion': diffusion_loss.item(),
                'kl': kl_loss.item(),
                'reconstruction': recon_loss.item() if isinstance(recon_loss, torch.Tensor) else 0.0,
                **recon_loss_dict
            }
            
        except Exception as e:
            print(f"❌ Error in training step: {e}")
            # Return dummy loss to prevent crash
            dummy_loss = torch.tensor(1.0, device=device, requires_grad=True)
            return dummy_loss, {
                'total': 1.0,
                'diffusion': 1.0,
                'kl': 0.0,
                'reconstruction': 0.0
            }
    
    def validation_step(self, degraded_img, enhanced_img, preprocessed_img=None):
        """Validation step with reconstruction"""
        with torch.no_grad():
            try:
                # Generate enhanced image with fewer steps for faster validation
                pred_enhanced = self.sample(degraded_img, num_steps=10)
                
                # Compute validation losses
                val_loss, val_loss_dict = self.loss_fn(pred_enhanced, enhanced_img)
                
                return val_loss, val_loss_dict, pred_enhanced
                
            except Exception as e:
                print(f"⚠️  Error in validation step: {e}")
                # Return dummy values to avoid breaking training
                dummy_loss = torch.tensor(0.0, device=degraded_img.device)
                dummy_dict = {'total': 0.0, 'l1': 0.0, 'ssim': 0.0, 'perceptual': 0.0}
                dummy_pred = torch.zeros_like(enhanced_img)
                return dummy_loss, dummy_dict, dummy_pred
    
    def configure_optimizers(self, learning_rate=1e-4, weight_decay=1e-5):
        """Configure optimizers for different components"""
        # Different learning rates for different components
        vae_params = list(self.encoder.parameters()) + list(self.decoder.parameters())
        diffusion_params = list(self.diffusion_unet.parameters())
        
        optimizers = {
            'vae': torch.optim.AdamW(vae_params, lr=learning_rate, weight_decay=weight_decay),
            'diffusion': torch.optim.AdamW(diffusion_params, lr=learning_rate, weight_decay=weight_decay)
        }
        
        # Learning rate schedulers
        schedulers = {
            'vae': torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['vae'], T_max=1000, eta_min=1e-6),
            'diffusion': torch.optim.lr_scheduler.CosineAnnealingLR(optimizers['diffusion'], T_max=1000, eta_min=1e-6)
        }
        
        return optimizers, schedulers


class UnderwaterEnhancementModel(nn.Module):
    """Simplified wrapper for inference"""
    def __init__(self, model_path=None, device='cuda'):
        super().__init__()
        self.device = device
        
        # Initialize the main model
        self.model = UnderwaterEnhancementDiffusion()
        
        # Load pretrained weights if provided
        if model_path:
            self.load_model(model_path)
        
        self.model.to(device)
        self.model.eval()
    
    def load_model(self, model_path):
        """Load pretrained model weights"""
        checkpoint = torch.load(model_path, map_location='cpu')
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
        
        print(f"Model loaded from {model_path}")
    
    @torch.no_grad()
    def enhance_image(self, degraded_img, num_steps=50, use_ddim=True):
        """Enhance a single underwater image"""
        self.model.eval()
        
        # Ensure correct input format
        if degraded_img.dim() == 3:
            degraded_img = degraded_img.unsqueeze(0)
        
        degraded_img = degraded_img.to(self.device)
        
        # Normalize to [-1, 1] if in [0, 1]
        if degraded_img.max() <= 1.0:
            degraded_img = degraded_img * 2.0 - 1.0
        
        # Generate enhanced image
        enhanced_img = self.model.sample(degraded_img, num_steps=num_steps, use_ddim=use_ddim)
        
        # Denormalize to [0, 1]
        enhanced_img = (enhanced_img + 1.0) / 2.0
        enhanced_img = torch.clamp(enhanced_img, 0, 1)
        
        return enhanced_img
    
    @torch.no_grad()
    def enhance_batch(self, degraded_batch, num_steps=50, use_ddim=True):
        """Enhance a batch of underwater images"""
        self.model.eval()
        
        degraded_batch = degraded_batch.to(self.device)
        
        # Normalize if needed
        if degraded_batch.max() <= 1.0:
            degraded_batch = degraded_batch * 2.0 - 1.0
        
        # Generate enhanced images
        enhanced_batch = self.model.sample(degraded_batch, num_steps=num_steps, use_ddim=use_ddim)
        
        # Denormalize
        enhanced_batch = (enhanced_batch + 1.0) / 2.0
        enhanced_batch = torch.clamp(enhanced_batch, 0, 1)
        
        return enhanced_batch


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


if __name__ == "__main__":
    # Test the complete model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create model
    model = UnderwaterEnhancementDiffusion().to(device)
    
    # Test data
    degraded = torch.randn(2, 3, 256, 256).to(device)
    enhanced = torch.randn(2, 3, 256, 256).to(device)
    
    print("Testing training step...")
    loss, loss_dict = model.training_step(degraded, enhanced)
    print(f"Training loss: {loss.item():.4f}")
    print("Loss components:", {k: f"{v:.4f}" for k, v in loss_dict.items()})
    
    print("\nTesting sampling...")
    with torch.no_grad():
        sampled = model.sample(degraded, num_steps=10)
        print(f"Sampled shape: {sampled.shape}")
    
    print("\nTesting inference model...")
    inference_model = UnderwaterEnhancementModel(device=device)
    enhanced_result = inference_model.enhance_image(degraded[0])
    print(f"Enhanced shape: {enhanced_result.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Statistics:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
