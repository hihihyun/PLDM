"""
Conditional Diffusion UNet for underwater image enhancement
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import ModuleList

from .basic_modules import (
    ConditionalResBlock, 
    AttentionBlock, 
    CrossAttentionBlock, 
    SinusoidalPositionEmbedding
)


class ConditionalDiffusionUNet(nn.Module):
    """U-Net with time and physics conditioning for diffusion process"""
    def __init__(self, in_channels=4, out_channels=4, base_channels=128, 
                 time_dim=256, physics_dim=128, num_res_blocks=2):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.base_channels = base_channels
        self.time_dim = time_dim
        self.physics_dim = physics_dim
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim // 4),
            nn.Linear(time_dim // 4, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Physics conditioning projection
        self.physics_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(physics_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Initial convolution
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder blocks
        self.encoder_blocks = ModuleList()
        self.encoder_attentions = ModuleList()
        self.encoder_cross_attentions = ModuleList()
        
        # Level 1: base_channels -> base_channels*2
        encoder_block1 = ModuleList()
        for _ in range(num_res_blocks):
            encoder_block1.append(ConditionalResBlock(base_channels, base_channels, time_dim))
        encoder_block1.append(ConditionalResBlock(base_channels, base_channels*2, time_dim, downsample=True))
        self.encoder_blocks.append(encoder_block1)
        self.encoder_attentions.append(None)  # No attention at first level
        self.encoder_cross_attentions.append(None)
        
        # Level 2: base_channels*2 -> base_channels*4
        encoder_block2 = ModuleList()
        for _ in range(num_res_blocks):
            encoder_block2.append(ConditionalResBlock(base_channels*2, base_channels*2, time_dim))
        encoder_block2.append(ConditionalResBlock(base_channels*2, base_channels*4, time_dim, downsample=True))
        self.encoder_blocks.append(encoder_block2)
        self.encoder_attentions.append(AttentionBlock(base_channels*2))
        self.encoder_cross_attentions.append(CrossAttentionBlock(base_channels*2, physics_dim))
        
        # Level 3: base_channels*4 -> base_channels*8
        encoder_block3 = ModuleList()
        for _ in range(num_res_blocks):
            encoder_block3.append(ConditionalResBlock(base_channels*4, base_channels*4, time_dim))
        encoder_block3.append(ConditionalResBlock(base_channels*4, base_channels*8, time_dim, downsample=True))
        self.encoder_blocks.append(encoder_block3)
        self.encoder_attentions.append(AttentionBlock(base_channels*4))
        self.encoder_cross_attentions.append(CrossAttentionBlock(base_channels*4, physics_dim))
        
        # Bottleneck
        self.bottleneck = ModuleList([
            ConditionalResBlock(base_channels*8, base_channels*8, time_dim),
            AttentionBlock(base_channels*8),
            CrossAttentionBlock(base_channels*8, physics_dim),
            ConditionalResBlock(base_channels*8, base_channels*8, time_dim),
            AttentionBlock(base_channels*8),
            CrossAttentionBlock(base_channels*8, physics_dim),
        ])
        
        # Decoder blocks
        self.decoder_blocks = ModuleList()
        self.decoder_attentions = ModuleList()
        self.decoder_cross_attentions = ModuleList()
        
        # Level 1: base_channels*8 -> base_channels*4
        decoder_block1 = ModuleList()
        decoder_block1.append(ConditionalResBlock(base_channels*8 + base_channels*8, base_channels*8, time_dim))
        for _ in range(num_res_blocks):
            decoder_block1.append(ConditionalResBlock(base_channels*8, base_channels*8, time_dim))
        decoder_block1.append(ConditionalResBlock(base_channels*8, base_channels*4, time_dim, upsample=True))
        self.decoder_blocks.append(decoder_block1)
        self.decoder_attentions.append(AttentionBlock(base_channels*8))
        self.decoder_cross_attentions.append(CrossAttentionBlock(base_channels*8, physics_dim))
        
        # Level 2: base_channels*4 -> base_channels*2
        decoder_block2 = ModuleList()
        decoder_block2.append(ConditionalResBlock(base_channels*4 + base_channels*4, base_channels*4, time_dim))
        for _ in range(num_res_blocks):
            decoder_block2.append(ConditionalResBlock(base_channels*4, base_channels*4, time_dim))
        decoder_block2.append(ConditionalResBlock(base_channels*4, base_channels*2, time_dim, upsample=True))
        self.decoder_blocks.append(decoder_block2)
        self.decoder_attentions.append(AttentionBlock(base_channels*4))
        self.decoder_cross_attentions.append(CrossAttentionBlock(base_channels*4, physics_dim))
        
        # Level 3: base_channels*2 -> base_channels
        decoder_block3 = ModuleList()
        decoder_block3.append(ConditionalResBlock(base_channels*2 + base_channels*2, base_channels*2, time_dim))
        for _ in range(num_res_blocks):
            decoder_block3.append(ConditionalResBlock(base_channels*2, base_channels*2, time_dim))
        decoder_block3.append(ConditionalResBlock(base_channels*2, base_channels, time_dim, upsample=True))
        self.decoder_blocks.append(decoder_block3)
        self.decoder_attentions.append(AttentionBlock(base_channels*2))
        self.decoder_cross_attentions.append(CrossAttentionBlock(base_channels*2, physics_dim))
        
        # Final output
        self.final_block = ModuleList([
            ConditionalResBlock(base_channels + base_channels, base_channels, time_dim),
            ConditionalResBlock(base_channels, base_channels, time_dim),
        ])
        
        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, out_channels, 3, padding=1)
        )
        
    def forward(self, x, t, physics_context):
        """
        x: noisy latent [B, in_channels, H, W]
        t: time steps [B]
        physics_context: physics features [B, physics_dim, H_p, W_p]
        """
        # Time embedding
        t_emb = self.time_embedding(t)
        
        # Physics embedding
        physics_emb = self.physics_proj(physics_context)
        
        # Combine time and physics embeddings
        combined_emb = t_emb + physics_emb
        
        # Initial convolution
        h = self.conv_in(x)
        
        # Store skip connections
        skip_connections = [h]
        
        # Encoder
        for i, (blocks, attn, cross_attn) in enumerate(zip(
            self.encoder_blocks, self.encoder_attentions, self.encoder_cross_attentions
        )):
            for j, block in enumerate(blocks):
                h = block(h, combined_emb)
                
                # Apply attention after first few blocks
                if j == len(blocks) - 2:  # Before the downsampling block
                    if attn is not None:
                        h = attn(h)
                    if cross_attn is not None:
                        h = cross_attn(h, physics_context)
                
                # Store skip connection before downsampling
                if j == len(blocks) - 1:
                    skip_connections.append(h)
        
        # Bottleneck
        for i, block in enumerate(self.bottleneck):
            if isinstance(block, ConditionalResBlock):
                h = block(h, combined_emb)
            elif isinstance(block, CrossAttentionBlock):
                h = block(h, physics_context)
            else:  # AttentionBlock
                h = block(h)
        
        # Decoder
        skip_idx = len(skip_connections) - 1
        for i, (blocks, attn, cross_attn) in enumerate(zip(
            self.decoder_blocks, self.decoder_attentions, self.decoder_cross_attentions
        )):
            for j, block in enumerate(blocks):
                # Concatenate skip connection at the beginning of each level
                if j == 0:
                    h = torch.cat([h, skip_connections[skip_idx]], dim=1)
                    skip_idx -= 1
                
                h = block(h, combined_emb)
                
                # Apply attention after first few blocks
                if j == 1:  # After the first block with skip connection
                    if attn is not None:
                        h = attn(h)
                    if cross_attn is not None:
                        h = cross_attn(h, physics_context)
        
        # Final blocks
        h = torch.cat([h, skip_connections[0]], dim=1)  # Last skip connection
        for block in self.final_block:
            h = block(h, combined_emb)
        
        # Output
        output = self.out_conv(h)
        
        return output


class LightweightDiffusionUNet(nn.Module):
    """Lightweight version of the UNet for faster inference"""
    def __init__(self, in_channels=4, out_channels=4, base_channels=64, 
                 time_dim=128, physics_dim=64):
        super().__init__()
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            SinusoidalPositionEmbedding(time_dim // 4),
            nn.Linear(time_dim // 4, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Physics conditioning
        self.physics_proj = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(physics_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Simplified U-Net structure
        self.conv_in = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Encoder
        self.enc1 = ConditionalResBlock(base_channels, base_channels*2, time_dim, downsample=True)
        self.enc2 = ConditionalResBlock(base_channels*2, base_channels*4, time_dim, downsample=True)
        self.enc3 = ConditionalResBlock(base_channels*4, base_channels*8, time_dim, downsample=True)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            ConditionalResBlock(base_channels*8, base_channels*8, time_dim),
            AttentionBlock(base_channels*8),
        )
        
        # Decoder
        self.dec3 = ConditionalResBlock(base_channels*8 + base_channels*8, base_channels*4, time_dim, upsample=True)
        self.dec2 = ConditionalResBlock(base_channels*4 + base_channels*4, base_channels*2, time_dim, upsample=True)
        self.dec1 = ConditionalResBlock(base_channels*2 + base_channels*2, base_channels, time_dim, upsample=True)
        
        # Output
        self.out_conv = nn.Sequential(
            nn.GroupNorm(32, base_channels + base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels + base_channels, out_channels, 3, padding=1)
        )
        
    def forward(self, x, t, physics_context):
        # Embeddings
        t_emb = self.time_embedding(t)
        physics_emb = self.physics_proj(physics_context)
        combined_emb = t_emb + physics_emb
        
        # Initial conv
        h0 = self.conv_in(x)
        
        # Encoder
        h1 = self.enc1(h0, combined_emb)
        h2 = self.enc2(h1, combined_emb)
        h3 = self.enc3(h2, combined_emb)
        
        # Bottleneck
        h = h3
        for block in self.bottleneck:
            if isinstance(block, ConditionalResBlock):
                h = block(h, combined_emb)
            else:
                h = block(h)
        
        # Decoder with skip connections
        h = self.dec3(torch.cat([h, h3], dim=1), combined_emb)
        h = self.dec2(torch.cat([h, h2], dim=1), combined_emb)
        h = self.dec1(torch.cat([h, h1], dim=1), combined_emb)
        
        # Output
        h = torch.cat([h, h0], dim=1)
        output = self.out_conv(h)
        
        return output


if __name__ == "__main__":
    # Test the UNet
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test ConditionalDiffusionUNet
    unet = ConditionalDiffusionUNet().to(device)
    
    x = torch.randn(2, 4, 64, 64).to(device)  # Latent input
    t = torch.randint(0, 1000, (2,)).to(device)  # Time steps
    physics_context = torch.randn(2, 128, 32, 32).to(device)  # Physics features
    
    output = unet(x, t, physics_context)
    print(f"UNet output shape: {output.shape}")
    
    # Test LightweightDiffusionUNet
    light_unet = LightweightDiffusionUNet().to(device)
    physics_context_light = torch.randn(2, 64, 32, 32).to(device)
    
    output_light = light_unet(x, t, physics_context_light)
    print(f"Lightweight UNet output shape: {output_light.shape}")
    
    # Count parameters
    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Full UNet parameters: {count_parameters(unet):,}")
    print(f"Lightweight UNet parameters: {count_parameters(light_unet):,}")
