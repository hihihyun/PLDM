"""
Fixed Basic Modules with proper dimension handling and numerical stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FixedResBlock(nn.Module):
    """Fixed ResNet block with proper dimension handling"""
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        
        stride = 2 if downsample else 1
        
        # Main path
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection
        if in_channels != out_channels or downsample:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip = nn.Identity()
    
    def forward(self, x):
        residual = self.skip(x)
        
        h = F.relu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        
        return F.relu(h + residual)


class FixedAttentionBlock(nn.Module):
    """Fixed attention block"""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = min(num_heads, channels // 32)  # Reasonable head count
        
        if self.num_heads == 0:
            self.num_heads = 1
            
        self.head_dim = channels // self.num_heads
        
        self.norm = nn.BatchNorm2d(channels)
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(channels, channels, 1)
        self.v = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        h = self.norm(x)
        
        # Compute attention
        q = self.q(h).view(B, self.num_heads, self.head_dim, H * W)
        k = self.k(h).view(B, self.num_heads, self.head_dim, H * W)
        v = self.v(h).view(B, self.num_heads, self.head_dim, H * W)
        
        # Attention weights
        attn = torch.matmul(q.transpose(-2, -1), k) / (self.head_dim ** 0.5)
        attn = torch.clamp(attn, -50, 50)  # Prevent extreme values
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v.transpose(-2, -1))
        out = out.transpose(-2, -1).contiguous().view(B, C, H, W)
        
        out = self.proj(out)
        
        return residual + out


# Aliases for compatibility
ResBlock = FixedResBlock
AttentionBlock = FixedAttentionBlock


if __name__ == "__main__":
    print("Testing basic modules...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test ResBlock
    try:
        block = FixedResBlock(64, 128, downsample=True).to(device)
        x = torch.randn(2, 64, 32, 32).to(device)
        out = block(x)
        print(f"✅ ResBlock - Input: {x.shape}, Output: {out.shape}")
    except Exception as e:
        print(f"❌ ResBlock failed: {e}")
    
    # Test AttentionBlock
    try:
        attn = FixedAttentionBlock(128).to(device)
        x = torch.randn(2, 128, 16, 16).to(device)
        out = attn(x)
        print(f"✅ AttentionBlock - Input: {x.shape}, Output: {out.shape}")
    except Exception as e:
        print(f"❌ AttentionBlock failed: {e}")
