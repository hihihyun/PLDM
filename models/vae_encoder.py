"""
VAE Encoder with integrated Water-Physics module for underwater image enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

from .water_physics import WaterPhysicsModule, ConditionalFusionModule
from .basic_modules import ResBlock, AttentionBlock


class EnhancedVAEEncoder(nn.Module):
    """VAE Encoder with integrated Water-Physics module"""
    def __init__(self, in_channels=3, latent_channels=4, base_channels=128):
        super().__init__()
        self.water_physics = WaterPhysicsModule(in_channels, base_channels//2)
        self.fusion_module = ConditionalFusionModule(12, base_channels//2)
        
        # Calculate total input channels: original(3) + physics(64) + fusion(64) = 131
        total_input_channels = in_channels + (base_channels//2) + (base_channels//2)
        
        # Progressive downsampling with attention
        self.encoder_blocks = ModuleList([
            ResBlock(total_input_channels, base_channels, downsample=True),  # 131 -> 128 channels
            ResBlock(base_channels, base_channels*2, downsample=True),  # 128x128 -> 64x64
            AttentionBlock(base_channels*2),
            ResBlock(base_channels*2, base_channels*4, downsample=True),  # 64x64 -> 32x32
            AttentionBlock(base_channels*4),
            ResBlock(base_channels*4, base_channels*8, downsample=True),  # 32x32 -> 16x16
            ResBlock(base_channels*8, base_channels*8, downsample=True),  # 16x16 -> 8x8
        ])
        
        # Final layers for mean and logvar
        self.norm_out = nn.GroupNorm(32, base_channels*8)
        self.conv_out_mean = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        self.conv_out_logvar = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        
    def forward(self, x, preprocessed_x=None):
        """
        x: original image [B, 3, H, W]
        preprocessed_x: preprocessed features [B, 9, H, W] (optional, will compute if not provided)
        """
        batch_size = x.shape[0]
        
        # Clamp input to prevent extreme values
        x = torch.clamp(x, -10, 10)
        
        # Get physics-based features - same spatial size as input
        physics_features, transmission, backscatter = self.water_physics(x)  # [B, 64, H, W]
        
        # Get preprocessed features if not provided
        if preprocessed_x is None:
            from .water_physics import WaterNetPreprocessor
            preprocessed_x = WaterNetPreprocessor.preprocess_batch(x)  # [B, 9, H, W]
        
        # Fusion of original and preprocessed features
        fusion_features = self.fusion_module(x, preprocessed_x)  # [B, 64, H, W]
        
        # Ensure all tensors have the same spatial dimensions
        assert x.shape[2:] == physics_features.shape[2:], f"x shape {x.shape} vs physics shape {physics_features.shape}"
        assert x.shape[2:] == fusion_features.shape[2:], f"x shape {x.shape} vs fusion shape {fusion_features.shape}"
        
        # Combine all features: [B, 3+64+64=131, H, W]
        h = torch.cat([x, physics_features, fusion_features], dim=1)
        
        # Progressive encoding with gradient clipping
        for i, block in enumerate(self.encoder_blocks):
            h = block(h)
            # Clamp intermediate values to prevent NaN
            h = torch.clamp(h, -50, 50)
            
            # Check for NaN during forward pass
            if torch.isnan(h).any():
                print(f"⚠️  NaN detected in encoder block {i}!")
                print(f"Input stats: min={h.min():.3f}, max={h.max():.3f}, mean={h.mean():.3f}")
                # Replace NaN with zeros to continue
                h = torch.where(torch.isnan(h), torch.zeros_like(h), h)
        
        # Output mean and logvar with numerical stability
        h = self.norm_out(h)
        h = F.silu(h)
        
        # Clamp before final conv to prevent extreme values
        h = torch.clamp(h, -10, 10)
        
        mean = self.conv_out_mean(h)
        logvar = self.conv_out_logvar(h)
        
        # Clamp outputs to prevent extreme values and NaN
        mean = torch.clamp(mean, -10, 10)
        logvar = torch.clamp(logvar, -10, 10)  # Prevent exp(logvar) from exploding
        
        # Check for NaN in outputs
        if torch.isnan(mean).any() or torch.isnan(logvar).any():
            print("⚠️  NaN in VAE outputs! Using fallback values.")
            mean = torch.zeros_like(mean)
            logvar = torch.zeros_like(logvar)
        
        # Return physics features for conditioning
        physics_cond = torch.cat([transmission, backscatter], dim=1)
        
        return mean, logvar, physics_cond


class ResidualVAEEncoder(nn.Module):
    """Alternative VAE Encoder with residual connections and multi-scale features"""
    def __init__(self, in_channels=3, latent_channels=4, base_channels=128):
        super().__init__()
        self.water_physics = WaterPhysicsModule(in_channels, base_channels//2)
        self.fusion_module = ConditionalFusionModule(12, base_channels//2)
        
        # Multi-scale feature extraction
        self.scale1 = nn.Sequential(
            nn.Conv2d(in_channels + base_channels, base_channels, 3, padding=1),
            nn.GroupNorm(32, base_channels),
            nn.SiLU()
        )
        
        self.scale2 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            ResBlock(base_channels*2, base_channels*2),
            ResBlock(base_channels*2, base_channels*2),
            nn.GroupNorm(32, base_channels*2),
            nn.SiLU()
        )
        
        self.scale3 = nn.Sequential(
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            ResBlock(base_channels*4, base_channels*4),
            AttentionBlock(base_channels*4),
            ResBlock(base_channels*4, base_channels*4),
            nn.GroupNorm(32, base_channels*4),
            nn.SiLU()
        )
        
        self.scale4 = nn.Sequential(
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1),
            ResBlock(base_channels*8, base_channels*8),
            AttentionBlock(base_channels*8),
            ResBlock(base_channels*8, base_channels*8),
            nn.GroupNorm(32, base_channels*8),
            nn.SiLU()
        )
        
        # Final encoding
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels*8, base_channels*8, 3, stride=2, padding=1),
            ResBlock(base_channels*8, base_channels*8),
            nn.GroupNorm(32, base_channels*8),
            nn.SiLU()
        )
        
        # Output layers
        self.conv_out_mean = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        self.conv_out_logvar = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        
    def forward(self, x, preprocessed_x=None):
        """
        x: original image [B, 3, H, W]
        preprocessed_x: preprocessed features [B, 9, H, W]
        """
        # Get physics-based features
        physics_features, transmission, backscatter = self.water_physics(x)
        
        # Get preprocessed features if not provided
        if preprocessed_x is None:
            from .water_physics import WaterNetPreprocessor
            preprocessed_x = WaterNetPreprocessor.preprocess_batch(x)
        
        # Fusion of original and preprocessed features
        fusion_features = self.fusion_module(x, preprocessed_x)
        
        # Combine all features
        h = torch.cat([x, physics_features, fusion_features], dim=1)
        
        # Multi-scale encoding
        h1 = self.scale1(h)  # [B, 128, H, W]
        h2 = self.scale2(h1)  # [B, 256, H/2, W/2]
        h3 = self.scale3(h2)  # [B, 512, H/4, W/4]
        h4 = self.scale4(h3)  # [B, 1024, H/8, W/8]
        
        # Final encoding
        h_final = self.final_conv(h4)  # [B, 1024, H/16, W/16]
        
        # Output
        mean = self.conv_out_mean(h_final)
        logvar = self.conv_out_logvar(h_final)
        
        # Physics conditioning
        physics_cond = torch.cat([transmission, backscatter], dim=1)
        
        return mean, logvar, physics_cond


def reparameterize(mean, logvar):
    """VAE reparameterization trick"""
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


if __name__ == "__main__":
    # Test the encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test EnhancedVAEEncoder
    encoder = EnhancedVAEEncoder().to(device)
    x = torch.randn(2, 3, 256, 256).to(device)
    
    mean, logvar, physics_cond = encoder(x)
    print(f"Mean shape: {mean.shape}")
    print(f"Logvar shape: {logvar.shape}")
    print(f"Physics condition shape: {physics_cond.shape}")
    
    # Test reparameterization
    z = reparameterize(mean, logvar)
    print(f"Latent shape: {z.shape}")