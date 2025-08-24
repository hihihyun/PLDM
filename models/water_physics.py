"""
Water-Net inspired physics-based feature extraction for underwater conditions
Based on the underwater image formation model and Water-Net preprocessing
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np


class WaterPhysicsModule(nn.Module):
    """Water-Net inspired physics-based feature extraction for underwater conditions"""
    def __init__(self, in_channels=3, out_channels=64):
        super().__init__()
        # Transmission map estimation branch - keep same spatial size
        self.transmission_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # Removed stride=2
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        
        # Backscatter estimation branch - keep same spatial size
        self.backscatter_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 5, padding=2),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, padding=1),  # Removed stride=2
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, padding=1)
        )
        
        # Attenuation coefficient estimation - upsample to original size
        self.attenuation_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, 16, 1),
            nn.SiLU(),
            nn.Conv2d(16, out_channels, 1)
        )
        
    def forward(self, x):
        # Get original spatial dimensions
        batch_size, channels, height, width = x.shape
        
        # Estimate underwater physics parameters
        transmission = torch.sigmoid(self.transmission_net(x))  # [B, out_channels, H, W]
        backscatter = self.backscatter_net(x)  # [B, out_channels, H, W]
        attenuation = self.attenuation_net(x)  # [B, out_channels, 1, 1]
        
        # Expand attenuation to match spatial dimensions
        attenuation_spatial = attenuation.expand(-1, -1, height, width)  # [B, out_channels, H, W]
        
        # Combine physics-based features
        physics_features = transmission * backscatter + (1 - transmission) * attenuation_spatial
        
        return physics_features, transmission, backscatter


class WaterNetPreprocessor:
    """Water-Net style preprocessing for generating conditional inputs"""
    
    @staticmethod
    def white_balance(image):
        """Simple white balance correction"""
        if isinstance(image, torch.Tensor):
            # Ensure the tensor is on CPU for numpy conversion
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
        
        # Simple white balance using gray world assumption
        avg_r = np.mean(image[:, :, 0])
        avg_g = np.mean(image[:, :, 1])
        avg_b = np.mean(image[:, :, 2])
        
        avg_gray = (avg_r + avg_g + avg_b) / 3
        
        # 👇 [수정] 0으로 나누는 것을 방지하기 위해 분모에 epsilon(1e-6) 추가
        epsilon = 1e-6
        scale_r = avg_gray / (avg_r + epsilon)
        scale_g = avg_gray / (avg_g + epsilon)
        scale_b = avg_gray / (avg_b + epsilon)
        
        balanced = image.copy().astype(np.float32)
        balanced[:, :, 0] = np.clip(balanced[:, :, 0] * scale_r, 0, 255)
        balanced[:, :, 1] = np.clip(balanced[:, :, 1] * scale_g, 0, 255)
        balanced[:, :, 2] = np.clip(balanced[:, :, 2] * scale_b, 0, 255)
        
        balanced = balanced / 255.0
        return torch.tensor(balanced).permute(2, 0, 1).float()
    
    @staticmethod
    def gamma_correction(image, gamma=0.7):
        """Gamma correction for brightness adjustment"""
        if isinstance(image, torch.Tensor):
            # Clamp image to be non-negative before power operation
            return torch.pow(torch.clamp(image, min=0), gamma)
        else:
            image_norm = image / 255.0
            corrected = np.power(image_norm, gamma)
            return (corrected * 255).astype(np.uint8)
    
    @staticmethod
    def histogram_equalization(image):
        """CLAHE histogram equalization"""
        if isinstance(image, torch.Tensor):
            image = image.permute(1, 2, 0).cpu().numpy()
            image = (image * 255).astype(np.uint8)
        
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        enhanced = enhanced / 255.0
        
        return torch.tensor(enhanced).permute(2, 0, 1).float()
    
    @classmethod
    def preprocess_batch(cls, batch_images):
        """Process a batch of images with all three methods"""
        device = batch_images.device # Remember original device
        batch_size = batch_images.shape[0]
        wb_batch = []
        gc_batch = []
        he_batch = []
        
        # Move to CPU for numpy operations
        batch_images_cpu = batch_images.cpu()

        for i in range(batch_size):
            img = batch_images_cpu[i]
            
            # White balance
            wb_img = cls.white_balance(img)
            wb_batch.append(wb_img)
            
            # Gamma correction
            gc_img = cls.gamma_correction(img)
            gc_batch.append(gc_img)
            
            # Histogram equalization
            he_img = cls.histogram_equalization(img)
            he_batch.append(he_img)
        
        wb_batch = torch.stack(wb_batch)
        gc_batch = torch.stack(gc_batch)
        he_batch = torch.stack(he_batch)
        
        # Concatenate all preprocessed versions as conditioning
        fusion_features = torch.cat([wb_batch, gc_batch, he_batch], dim=1)
        
        return fusion_features.to(device) # Move back to original device


class ConditionalFusionModule(nn.Module):
    """Fusion module for combining original and preprocessed features"""
    def __init__(self, in_channels=12, out_channels=64):  # 3 (original) + 9 (3 preprocessing methods)
        super().__init__()
        
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * 2, 3, padding=1),
            nn.GroupNorm(16, out_channels * 2),
            nn.SiLU(),
            nn.Conv2d(out_channels * 2, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.SiLU()
        )
        
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channels, out_channels // 4, 1),
            nn.SiLU(),
            nn.Conv2d(out_channels // 4, out_channels, 1),
            nn.Sigmoid()
        )
        
    def forward(self, original, preprocessed):
        """
        original: [B, 3, H, W]
        preprocessed: [B, 9, H, W] (3 preprocessing methods × 3 channels)
        """
        # Concatenate original and preprocessed
        combined = torch.cat([original, preprocessed], dim=1)  # [B, 12, H, W]
        
        # Fusion
        fused = self.fusion_conv(combined)
        
        # Channel attention
        attention_weights = self.attention(fused)
        fused = fused * attention_weights
        
        return fused