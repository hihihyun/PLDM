"""
Basic building blocks for the underwater image enhancement model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ResBlock(nn.Module):
    """Residual block with optional downsampling"""
    def __init__(self, in_channels, out_channels, downsample=False):
        super().__init__()
        self.downsample = downsample
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        if in_channels != out_channels or downsample:
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=stride)
        else:
            self.skip = nn.Identity()
        
    def forward(self, x):
        residual = self.skip(x)
        
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        
        return F.silu(h + residual)


class AttentionBlock(nn.Module):
    """Memory-efficient Self-attention block for feature refinement"""
    def __init__(self, channels, num_heads=4):  # Reduced from 8 to 4 heads
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x):
        B, C, H, W = x.shape
        residual = x
        
        # Skip attention for very large feature maps to save memory
        if H * W > 64 * 64:
            # Just apply normalization and return
            return self.proj_out(self.norm(x)) + residual
        
        # Normalize
        h = self.norm(x)
        
        # Get Q, K, V
        qkv = self.qkv(h)
        q, k, v = qkv.chunk(3, dim=1)
        
        # Reshape for attention - use reshape instead of view for safety
        q = q.reshape(B, self.num_heads, self.head_dim, H * W)
        k = k.reshape(B, self.num_heads, self.head_dim, H * W)
        v = v.reshape(B, self.num_heads, self.head_dim, H * W)
        
        # Memory-efficient attention computation
        scale = 1.0 / math.sqrt(self.head_dim)
        
        # For small feature maps, use regular attention
        if H * W <= 32 * 32:
            attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
            attn = F.softmax(attn, dim=-1)
            out = torch.einsum('bhij,bhdj->bhdi', attn, v)
        else:
            # For larger feature maps, use chunked attention
            chunk_size = 512
            out_chunks = []
            
            for i in range(0, H * W, chunk_size):
                end_i = min(i + chunk_size, H * W)
                q_chunk = q[:, :, :, i:end_i]
                
                attn_chunk = torch.einsum('bhdi,bhdj->bhij', q_chunk, k) * scale
                attn_chunk = F.softmax(attn_chunk, dim=-1)
                out_chunk = torch.einsum('bhij,bhdj->bhdi', attn_chunk, v)
                out_chunks.append(out_chunk)
            
            out = torch.cat(out_chunks, dim=-1)
        
        out = out.reshape(B, C, H, W)
        
        # Project and residual
        out = self.proj_out(out)
        return out + residual


class ConditionalResBlock(nn.Module):
    """Residual block with time embedding conditioning"""
    def __init__(self, in_channels, out_channels, time_dim, downsample=False, upsample=False):
        super().__init__()
        self.downsample = downsample
        self.upsample = upsample
        
        stride = 2 if downsample else 1
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.norm1 = nn.GroupNorm(8, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_channels)
        
        # Time embedding projection
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        # Skip connection
        if in_channels != out_channels or downsample or upsample:
            skip_stride = stride
            self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=skip_stride)
        else:
            self.skip = nn.Identity()
            
        # Upsample layer
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        
    def forward(self, x, t_emb):
        if self.upsample:
            x = self.upsample_layer(x)
            
        residual = self.skip(x)
        
        # First conv
        h = self.conv1(x)
        h = self.norm1(h)
        h = F.silu(h)
        
        # Add time embedding
        t_proj = self.time_mlp(t_emb)[:, :, None, None]
        h = h + t_proj
        
        # Second conv
        h = self.conv2(h)
        h = self.norm2(h)
        
        return F.silu(h + residual)


class CrossAttentionBlock(nn.Module):
    """Memory-efficient Cross-attention block with fixed dimensions"""
    def __init__(self, channels, context_dim, num_heads=4):
        super().__init__()
        self.channels = channels
        self.context_dim = context_dim
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        
        assert channels % num_heads == 0
        
        self.norm = nn.GroupNorm(32, channels)
        
        # Use conv layers instead of linear for spatial context
        self.q = nn.Conv2d(channels, channels, 1)
        self.k = nn.Conv2d(context_dim, channels, 1)  # Changed from Linear to Conv2d
        self.v = nn.Conv2d(context_dim, channels, 1)  # Changed from Linear to Conv2d
        self.proj_out = nn.Conv2d(channels, channels, 1)
        
    def forward(self, x, context):
        """
        Memory-efficient cross-attention with proper dimension handling
        x: [B, C, H, W]
        context: [B, context_dim, H_ctx, W_ctx]
        """
        B, C, H, W = x.shape
        residual = x
        
        # Normalize input
        h = self.norm(x)
        
        # Handle context dimensions
        if context.dim() == 4:
            # Spatial context - resize to match x if needed
            if context.shape[2:] != x.shape[2:]:
                context = F.interpolate(context, size=(H, W), mode='bilinear', align_corners=False)
        elif context.dim() == 2:
            # Global context - expand to spatial
            context = context.unsqueeze(2).unsqueeze(3).expand(-1, -1, H, W)
        else:
            print(f"⚠️  Unexpected context dimension: {context.shape}")
            return residual  # Skip attention if context format is unexpected
        
        # Skip attention for very large feature maps to save memory
        if H * W > 64 * 64:
            # Simple projection without attention
            return self.proj_out(h) + residual
        
        try:
            # Get Q, K, V using conv layers
            q = self.q(h)  # [B, C, H, W]
            k = self.k(context)  # [B, C, H, W] 
            v = self.v(context)  # [B, C, H, W]
            
            # Reshape for attention computation
            q = q.reshape(B, self.num_heads, self.head_dim, H * W)
            k = k.reshape(B, self.num_heads, self.head_dim, H * W)
            v = v.reshape(B, self.num_heads, self.head_dim, H * W)
            
            # Memory-efficient attention computation
            scale = 1.0 / math.sqrt(self.head_dim)
            
            # For small feature maps, use regular attention
            if H * W <= 32 * 32:
                attn = torch.einsum('bhdi,bhdj->bhij', q, k) * scale
                attn = F.softmax(attn, dim=-1)
                out = torch.einsum('bhij,bhdj->bhdi', attn, v)
            else:
                # For larger feature maps, use chunked attention
                chunk_size = 512
                out_chunks = []
                
                for i in range(0, H * W, chunk_size):
                    end_i = min(i + chunk_size, H * W)
                    q_chunk = q[:, :, :, i:end_i]
                    
                    attn_chunk = torch.einsum('bhdi,bhdj->bhij', q_chunk, k) * scale
                    attn_chunk = F.softmax(attn_chunk, dim=-1)
                    out_chunk = torch.einsum('bhij,bhdj->bhdi', attn_chunk, v)
                    out_chunks.append(out_chunk)
                
                out = torch.cat(out_chunks, dim=-1)
            
            out = out.reshape(B, C, H, W)
            
            # Project and residual
            out = self.proj_out(out)
            return out + residual
            
        except Exception as e:
            print(f"⚠️  CrossAttention failed: {e}, using skip connection")
            return self.proj_out(h) + residual


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time steps"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        device = t.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = t[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.channels = channels
        self.reduction = reduction
        
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        y = self.fc(x)
        return x * y


class ConvBlock(nn.Module):
    """Basic convolution block with normalization and activation"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1,
                 norm_type='group', activation='silu'):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
        # Normalization
        if norm_type == 'group':
            num_groups = min(32, out_channels // 4) if out_channels >= 4 else 1
            self.norm = nn.GroupNorm(num_groups, out_channels)
        elif norm_type == 'batch':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm_type == 'instance':
            self.norm = nn.InstanceNorm2d(out_channels)
        else:
            self.norm = nn.Identity()
            
        # Activation
        if activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.Identity()
            
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution for efficiency"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, 
                                 stride, padding, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1)
        self.norm1 = nn.GroupNorm(min(32, in_channels // 4) if in_channels >= 4 else 1, in_channels)
        self.norm2 = nn.GroupNorm(min(32, out_channels // 4) if out_channels >= 4 else 1, out_channels)
        
    def forward(self, x):
        x = self.depthwise(x)
        x = self.norm1(x)
        x = F.silu(x)
        x = self.pointwise(x)
        x = self.norm2(x)
        x = F.silu(x)
        return x


if __name__ == "__main__":
    # Test basic modules
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Test ResBlock
    x = torch.randn(2, 64, 32, 32).to(device)
    resblock = ResBlock(64, 128, downsample=True).to(device)
    out = resblock(x)
    print(f"ResBlock output shape: {out.shape}")
    
    # Test AttentionBlock
    x = torch.randn(2, 128, 16, 16).to(device)
    attn = AttentionBlock(128).to(device)
    out = attn(x)
    print(f"AttentionBlock output shape: {out.shape}")
    
    # Test ConditionalResBlock
    x = torch.randn(2, 128, 16, 16).to(device)
    t_emb = torch.randn(2, 256).to(device)
    cond_resblock = ConditionalResBlock(128, 256, 256, downsample=True).to(device)
    out = cond_resblock(x, t_emb)
    print(f"ConditionalResBlock output shape: {out.shape}")
    
    # Test CrossAttentionBlock
    x = torch.randn(2, 128, 16, 16).to(device)
    context = torch.randn(2, 64, 8, 8).to(device)
    cross_attn = CrossAttentionBlock(128, 64).to(device)
    out = cross_attn(x, context)
    print(f"CrossAttentionBlock output shape: {out.shape}")
