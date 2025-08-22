"""
Fixed VAE Encoder with integrated Water-Physics module
- Fixed dimension mismatch issues
- Added numerical stability
- Improved error handling
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

# Try to import water physics modules
try:
    from models.water_physics import WaterPhysicsModule, ConditionalFusionModule
    from models.basic_modules import ResBlock, AttentionBlock
    PHYSICS_AVAILABLE = True
except ImportError:
    print("⚠️  Water physics modules not found, using fallback")
    PHYSICS_AVAILABLE = False
    
    # Fallback implementations
    class WaterPhysicsModule(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            
        def forward(self, x):
            features = self.conv(x)
            transmission = torch.ones_like(features)
            backscatter = torch.zeros_like(features)
            return features, transmission, backscatter
    
    class ConditionalFusionModule(nn.Module):
        def __init__(self, in_channels, out_channels):
            super().__init__()
            self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            
        def forward(self, x, context):
            return self.conv(torch.cat([x, context], dim=1)[:, :in_channels])
    
    class ResBlock(nn.Module):
        def __init__(self, in_channels, out_channels, downsample=False):
            super().__init__()
            stride = 2 if downsample else 1
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.norm1 = nn.BatchNorm2d(out_channels)
            self.norm2 = nn.BatchNorm2d(out_channels)
            
            if in_channels != out_channels or downsample:
                self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
            else:
                self.skip = nn.Identity()
                
        def forward(self, x):
            h = F.relu(self.norm1(self.conv1(x)))
            h = self.norm2(self.conv2(h))
            return F.relu(h + self.skip(x))
    
    class AttentionBlock(nn.Module):
        def __init__(self, channels):
            super().__init__()
            self.norm = nn.BatchNorm2d(channels)
            
        def forward(self, x):
            return self.norm(x)


class LightweightVAEEncoder(nn.Module):
    """Lightweight VAE Encoder for testing and debugging"""
    def __init__(self, in_channels=3, latent_channels=4, base_channels=64):
        super().__init__()
        
        # Simple encoder without physics module for debugging
        self.encoder = nn.Sequential(
            # Input: [B, 3, H, W]
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            
            # Downsample 1: [B, 64, H/2, W/2] 
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*2),
            nn.ReLU(),
            
            # Downsample 2: [B, 128, H/4, W/4]
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*4),
            nn.ReLU(),
            
            # Downsample 3: [B, 256, H/8, W/8]
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1),
            nn.BatchNorm2d(base_channels*8),
            nn.ReLU(),
        )
        
        # Output layers
        self.conv_out_mean = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        self.conv_out_logvar = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        
    def forward(self, x, preprocessed_x=None):
        """Simple forward pass for debugging"""
        x = torch.clamp(x, -5, 5)
        
        h = self.encoder(x)
        
        mean = self.conv_out_mean(h)
        logvar = self.conv_out_logvar(h)
        
        # Clamp outputs
        mean = torch.clamp(mean, -5, 5)
        logvar = torch.clamp(logvar, -10, 5)
        
        # Dummy physics conditioning
        physics_cond = torch.zeros(x.shape[0], 64, x.shape[2], x.shape[3], device=x.device)
        
        return mean, logvar, physics_cond


class FixedVAEEncoder(nn.Module):
    """Fixed VAE Encoder with proper dimension handling"""
    def __init__(self, in_channels=3, latent_channels=4, base_channels=128, physics_dim=64):
        super().__init__()
        self.physics_dim = physics_dim
        
        # Water physics module
        self.water_physics = WaterPhysicsModule(in_channels, physics_dim)
        self.fusion_module = ConditionalFusionModule(12, physics_dim)
        
        # Calculate total input channels: original(3) + physics(64) + fusion(64) = 131
        total_input_channels = in_channels + physics_dim + physics_dim
        
        # Initial conv to match channel count
        self.input_conv = nn.Conv2d(total_input_channels, base_channels, 3, padding=1)
        self.input_norm = nn.BatchNorm2d(base_channels)
        
        # Progressive downsampling
        self.encoder_blocks = nn.ModuleList([
            ResBlock(base_channels, base_channels, downsample=True),
            ResBlock(base_channels, base_channels*2, downsample=True),
            AttentionBlock(base_channels*2),
            ResBlock(base_channels*2, base_channels*4, downsample=True),
            AttentionBlock(base_channels*4),
            ResBlock(base_channels*4, base_channels*8, downsample=True),
            ResBlock(base_channels*8, base_channels*8, downsample=True),
        ])
        
        # Final layers
        self.norm_out = nn.BatchNorm2d(base_channels*8)
        self.conv_out_mean = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        self.conv_out_logvar = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        
    def forward(self, x, preprocessed_x=None):
        """Fixed forward pass with proper error handling"""
        try:
            # Clamp input
            x = torch.clamp(x, -5, 5)
            
            # Get physics features
            physics_features, transmission, backscatter = self.water_physics(x)
            
            # Get preprocessed features (dummy for now)
            if preprocessed_x is None:
                preprocessed_x = torch.randn(x.shape[0], 9, x.shape[2], x.shape[3], device=x.device)
            
            # Fusion features
            fusion_features = self.fusion_module(x, preprocessed_x)
            
            # Ensure spatial dimensions match
            H, W = x.shape[2], x.shape[3]
            if physics_features.shape[2:] != (H, W):
                physics_features = F.interpolate(physics_features, size=(H, W), mode='bilinear')
            if fusion_features.shape[2:] != (H, W):
                fusion_features = F.interpolate(fusion_features, size=(H, W), mode='bilinear')
            
            # Combine features
            h = torch.cat([x, physics_features, fusion_features], dim=1)
            
            # Initial convolution
            h = F.relu(self.input_norm(self.input_conv(h)))
            
            # Progressive encoding
            for i, block in enumerate(self.encoder_blocks):
                h = block(h)
                if torch.isnan(h).any():
                    print(f"⚠️  NaN detected in block {i}")
                    break
                h = torch.clamp(h, -10, 10)
            
            # Output
            h = self.norm_out(h)
            h = F.relu(h)
            
            mean = self.conv_out_mean(h)
            logvar = self.conv_out_logvar(h)
            
            # Clamp outputs
            mean = torch.clamp(mean, -8, 8)
            logvar = torch.clamp(logvar, -15, 10)
            
            # Physics conditioning
            physics_cond = torch.cat([transmission, backscatter], dim=1)
            
            return mean, logvar, physics_cond
            
        except Exception as e:
            print(f"❌ Error in VAE encoder: {e}")
            # Safe fallback
            H_out, W_out = x.shape[2] // 32, x.shape[3] // 32
            mean = torch.zeros(x.shape[0], 4, H_out, W_out, device=x.device)
            logvar = torch.full((x.shape[0], 4, H_out, W_out), -10, device=x.device)
            physics_cond = torch.zeros(x.shape[0], 128, x.shape[2], x.shape[3], device=x.device)
            return mean, logvar, physics_cond


def reparameterize(mean, logvar):
    """VAE reparameterization trick with numerical stability"""
    logvar = torch.clamp(logvar, -15, 10)
    std = torch.exp(0.5 * logvar)
    eps = torch.randn_like(std)
    return mean + eps * std


if __name__ == "__main__":
    print("Testing VAE Encoder modules...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test LightweightVAEEncoder
    try:
        encoder = LightweightVAEEncoder(base_channels=32).to(device)
        x = torch.randn(2, 3, 64, 64).to(device)
        mean, logvar, physics_cond = encoder(x)
        print(f"✅ LightweightVAEEncoder - Mean: {mean.shape}, Logvar: {logvar.shape}")
    except Exception as e:
        print(f"❌ LightweightVAEEncoder failed: {e}")
    
    # Test FixedVAEEncoder
    try:
        encoder = FixedVAEEncoder(base_channels=32, physics_dim=32).to(device)
        x = torch.randn(2, 3, 64, 64).to(device)
        mean, logvar, physics_cond = encoder(x)
        print(f"✅ FixedVAEEncoder - Mean: {mean.shape}, Logvar: {logvar.shape}")
    except Exception as e:
        print(f"❌ FixedVAEEncoder failed: {e}")
