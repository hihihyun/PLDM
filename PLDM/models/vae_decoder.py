"""
VAE Decoder for underwater image enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .basic_modules import ResBlock, AttentionBlock


class VAEDecoder(nn.Module):
    """Decoder from latent to image space"""
    def __init__(self, latent_channels=4, out_channels=3, base_channels=128):
        super().__init__()
        
        # Initial projection
        self.conv_in = nn.Conv2d(latent_channels, base_channels*8, 3, padding=1)
        
        # Progressive upsampling blocks
        self.decoder_blocks = nn.Sequential(
            ResBlock(base_channels*8, base_channels*8),
            ResBlock(base_channels*8, base_channels*8),
            AttentionBlock(base_channels*8),
            
            # Upsample 1: 8x8 -> 16x16
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResBlock(base_channels*8, base_channels*4),
            ResBlock(base_channels*4, base_channels*4),
            AttentionBlock(base_channels*4),
            
            # Upsample 2: 16x16 -> 32x32
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResBlock(base_channels*4, base_channels*2),
            ResBlock(base_channels*2, base_channels*2),
            AttentionBlock(base_channels*2),
            
            # Upsample 3: 32x32 -> 64x64
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResBlock(base_channels*2, base_channels),
            ResBlock(base_channels, base_channels),
            
            # Upsample 4: 64x64 -> 128x128
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResBlock(base_channels, base_channels),
            ResBlock(base_channels, base_channels),
            
            # Upsample 5: 128x128 -> 256x256
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            ResBlock(base_channels, base_channels),
        )
        
        # Final output layer
        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh()  # Output in [-1, 1] range
        )
        
    def forward(self, z):
        """
        z: latent tensor [B, latent_channels, H_latent, W_latent]
        """
        h = self.conv_in(z)
        h = self.decoder_blocks(h)
        output = self.out_conv(h)
        return output


class SkipConnectionVAEDecoder(nn.Module):
    """VAE Decoder with skip connections for better detail preservation"""
    def __init__(self, latent_channels=4, out_channels=3, base_channels=128):
        super().__init__()
        
        # Initial projection
        self.conv_in = nn.Conv2d(latent_channels, base_channels*8, 3, padding=1)
        
        # Decoder blocks with skip connection preparation
        self.block1 = nn.Sequential(
            ResBlock(base_channels*8, base_channels*8),
            ResBlock(base_channels*8, base_channels*8),
            AttentionBlock(base_channels*8)
        )
        
        # Level 1: 8x8 -> 16x16
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv1 = nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1)
        self.block2 = nn.Sequential(
            ResBlock(base_channels*4, base_channels*4),
            ResBlock(base_channels*4, base_channels*4),
            AttentionBlock(base_channels*4)
        )
        
        # Level 2: 16x16 -> 32x32
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv2 = nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1)
        self.block3 = nn.Sequential(
            ResBlock(base_channels*2, base_channels*2),
            ResBlock(base_channels*2, base_channels*2),
            AttentionBlock(base_channels*2)
        )
        
        # Level 3: 32x32 -> 64x64
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(base_channels*2, base_channels, 3, padding=1)
        self.block4 = nn.Sequential(
            ResBlock(base_channels, base_channels),
            ResBlock(base_channels, base_channels)
        )
        
        # Level 4: 64x64 -> 128x128
        self.upsample4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.block5 = nn.Sequential(
            ResBlock(base_channels, base_channels),
            ResBlock(base_channels, base_channels)
        )
        
        # Level 5: 128x128 -> 256x256
        self.upsample5 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv5 = nn.Conv2d(base_channels, base_channels, 3, padding=1)
        self.block6 = nn.Sequential(
            ResBlock(base_channels, base_channels),
            ResBlock(base_channels, base_channels)
        )
        
        # Output layer
        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1),
            nn.Tanh()
        )
        
    def forward(self, z, skip_connections=None):
        """
        z: latent tensor [B, latent_channels, H_latent, W_latent]
        skip_connections: list of tensors from encoder (optional)
        """
        # Initial processing
        h = self.conv_in(z)
        h = self.block1(h)
        
        # Progressive upsampling
        h = self.upsample1(h)
        h = self.conv1(h)
        h = self.block2(h)
        
        h = self.upsample2(h)
        h = self.conv2(h)
        h = self.block3(h)
        
        h = self.upsample3(h)
        h = self.conv3(h)
        h = self.block4(h)
        
        h = self.upsample4(h)
        h = self.conv4(h)
        h = self.block5(h)
        
        h = self.upsample5(h)
        h = self.conv5(h)
        h = self.block6(h)
        
        # Final output
        output = self.out_conv(h)
        return output


class ProgressiveVAEDecoder(nn.Module):
    """Progressive VAE Decoder that can generate multiple resolution outputs"""
    def __init__(self, latent_channels=4, out_channels=3, base_channels=128):
        super().__init__()
        
        self.conv_in = nn.Conv2d(latent_channels, base_channels*8, 3, padding=1)
        
        # Progressive blocks
        self.block_8x8 = nn.Sequential(
            ResBlock(base_channels*8, base_channels*8),
            AttentionBlock(base_channels*8)
        )
        
        self.to_rgb_8x8 = nn.Sequential(
            nn.GroupNorm(32, base_channels*8),
            nn.SiLU(),
            nn.Conv2d(base_channels*8, out_channels, 1),
            nn.Tanh()
        )
        
        # 8x8 -> 16x16
        self.upsample_16x16 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1),
            ResBlock(base_channels*4, base_channels*4),
            AttentionBlock(base_channels*4)
        )
        
        self.to_rgb_16x16 = nn.Sequential(
            nn.GroupNorm(32, base_channels*4),
            nn.SiLU(),
            nn.Conv2d(base_channels*4, out_channels, 1),
            nn.Tanh()
        )
        
        # 16x16 -> 32x32
        self.upsample_32x32 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            ResBlock(base_channels*2, base_channels*2),
            AttentionBlock(base_channels*2)
        )
        
        self.to_rgb_32x32 = nn.Sequential(
            nn.GroupNorm(32, base_channels*2),
            nn.SiLU(),
            nn.Conv2d(base_channels*2, out_channels, 1),
            nn.Tanh()
        )
        
        # 32x32 -> 64x64
        self.upsample_64x64 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*2, base_channels, 3, padding=1),
            ResBlock(base_channels, base_channels)
        )
        
        self.to_rgb_64x64 = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Tanh()
        )
        
        # 64x64 -> 128x128
        self.upsample_128x128 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            ResBlock(base_channels, base_channels)
        )
        
        self.to_rgb_128x128 = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Tanh()
        )
        
        # 128x128 -> 256x256
        self.upsample_256x256 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels, base_channels, 3, padding=1),
            ResBlock(base_channels, base_channels)
        )
        
        self.to_rgb_256x256 = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 1),
            nn.Tanh()
        )
        
    def forward(self, z, return_multiscale=False):
        """
        z: latent tensor [B, latent_channels, H_latent, W_latent]
        return_multiscale: if True, return outputs at all scales
        """
        outputs = {}
        
        # Initial processing
        h = self.conv_in(z)
        h = self.block_8x8(h)
        
        if return_multiscale:
            outputs['8x8'] = self.to_rgb_8x8(h)
        
        # Progressive upsampling
        h = self.upsample_16x16(h)
        if return_multiscale:
            outputs['16x16'] = self.to_rgb_16x16(h)
        
        h = self.upsample_32x32(h)
        if return_multiscale:
            outputs['32x32'] = self.to_rgb_32x32(h)
        
        h = self.upsample_64x64(h)
        if return_multiscale:
            outputs['64x64'] = self.to_rgb_64x64(h)
        
        h = self.upsample_128x128(h)
        if return_multiscale:
            outputs['128x128'] = self.to_rgb_128x128(h)
        
        h = self.upsample_256x256(h)
        final_output = self.to_rgb_256x256(h)
        
        if return_multiscale:
            outputs['256x256'] = final_output
            return outputs
        else:
            return final_output


if __name__ == "__main__":
    # Test the decoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test VAEDecoder
    decoder = VAEDecoder().to(device)
    z = torch.randn(2, 4, 8, 8).to(device)  # Latent size 8x8
    
    output = decoder(z)
    print(f"Output shape: {output.shape}")
    
    # Test ProgressiveVAEDecoder
    prog_decoder = ProgressiveVAEDecoder().to(device)
    
    # Single scale output
    output_single = prog_decoder(z)
    print(f"Progressive decoder single output shape: {output_single.shape}")
    
    # Multi-scale output
    outputs_multi = prog_decoder(z, return_multiscale=True)
    for scale, tensor in outputs_multi.items():
        print(f"Scale {scale}: {tensor.shape}")
