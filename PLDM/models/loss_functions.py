"""
Loss functions for underwater image enhancement
Including reconstruction, perceptual, SSIM, and frequency domain losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torchvision.transforms import Normalize
import math


class StructureAwareLoss(nn.Module):
    """Combined loss for structure preservation and perceptual quality"""
    def __init__(self, device='cuda'):
        super().__init__()
        self.device = device
        
        # Basic losses
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()
        
        # Loss weights
        self.l1_weight = 1.0
        self.l2_weight = 0.5
        self.grad_weight = 1.0
        self.perceptual_weight = 0.1
        self.ssim_weight = 0.5
        self.freq_weight = 0.05
        self.color_weight = 0.1
        
        # VGG for perceptual loss
        self.vgg = VGGPerceptualLoss().to(device)
        
    def gradient_loss(self, pred, target):
        """Compute gradient loss for structure preservation"""
        def gradient(x):
            h_diff = x[:, :, 1:, :] - x[:, :, :-1, :]
            w_diff = x[:, :, :, 1:] - x[:, :, :, :-1]
            return h_diff, w_diff
        
        pred_dy, pred_dx = gradient(pred)
        target_dy, target_dx = gradient(target)
        
        grad_loss = self.l1_loss(pred_dy, target_dy) + self.l1_loss(pred_dx, target_dx)
        return grad_loss
    
    def frequency_loss(self, pred, target):
        """FFT-based frequency domain loss for texture preservation"""
        # Ensure inputs are float32 to avoid ComplexHalf issues
        pred = pred.float()
        target = target.float()
        
        try:
            pred_fft = torch.fft.fft2(pred, norm='ortho')
            target_fft = torch.fft.fft2(target, norm='ortho')
            
            # Separate real and imaginary parts
            loss = self.l1_loss(pred_fft.real, target_fft.real) + \
                   self.l1_loss(pred_fft.imag, target_fft.imag)
            
            # High frequency emphasis
            _, _, h, w = pred.shape
            center_h, center_w = h // 2, w // 2
            
            # Create high frequency mask
            y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
            dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2).to(pred.device)
            high_freq_mask = (dist > min(h, w) * 0.1).float()
            
            # Apply mask and compute high frequency loss
            pred_fft_hf = pred_fft * high_freq_mask[None, None, :, :]
            target_fft_hf = target_fft * high_freq_mask[None, None, :, :]
            
            hf_loss = self.l1_loss(pred_fft_hf.real, target_fft_hf.real) + \
                      self.l1_loss(pred_fft_hf.imag, target_fft_hf.imag)
            
            return loss + 0.5 * hf_loss
            
        except Exception as e:
            print(f"⚠️  Frequency loss failed: {e}, using L1 fallback")
            return self.l1_loss(pred, target)
    
    def ssim_loss(self, pred, target, window_size=11):
        """SSIM loss for structural similarity"""
        def gaussian_kernel(size, sigma=1.5):
            coords = torch.arange(size, dtype=torch.float)
            coords -= size // 2
            g = torch.exp(-(coords**2) / (2 * sigma**2))
            g /= g.sum()
            return g.unsqueeze(0).unsqueeze(0) * g.unsqueeze(0).unsqueeze(1)
        
        # Create Gaussian kernel
        kernel = gaussian_kernel(window_size).expand(pred.shape[1], 1, window_size, window_size).to(pred.device)
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        # Compute local means
        mu_x = F.conv2d(pred, kernel, padding=window_size//2, groups=pred.shape[1])
        mu_y = F.conv2d(target, kernel, padding=window_size//2, groups=target.shape[1])
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        # Compute local variances and covariance
        sigma_x_sq = F.conv2d(pred**2, kernel, padding=window_size//2, groups=pred.shape[1]) - mu_x_sq
        sigma_y_sq = F.conv2d(target**2, kernel, padding=window_size//2, groups=target.shape[1]) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, kernel, padding=window_size//2, groups=pred.shape[1]) - mu_xy
        
        # SSIM calculation
        ssim = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / \
               ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        
        return 1 - ssim.mean()
    
    def color_consistency_loss(self, pred, target):
        """Color consistency loss for underwater images"""
        # RGB to LAB color space approximation
        def rgb_to_lab_approx(rgb):
            # Simplified RGB to LAB conversion
            r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
            
            # Approximate L channel
            l = 0.299 * r + 0.587 * g + 0.114 * b
            
            # Approximate a and b channels
            a = 0.5 * (r - g)
            b_chan = 0.5 * (g - b)
            
            return torch.cat([l, a, b_chan], dim=1)
        
        pred_lab = rgb_to_lab_approx(pred)
        target_lab = rgb_to_lab_approx(target)
        
        # Color consistency in LAB space
        color_loss = self.l1_loss(pred_lab, target_lab)
        
        # Additional chromaticity loss
        pred_chroma = torch.sqrt(pred_lab[:, 1:2]**2 + pred_lab[:, 2:3]**2)
        target_chroma = torch.sqrt(target_lab[:, 1:2]**2 + target_lab[:, 2:3]**2)
        chroma_loss = self.l1_loss(pred_chroma, target_chroma)
        
        return color_loss + 0.5 * chroma_loss
    
    def forward(self, pred, target):
        """Combined loss computation"""
        # Basic reconstruction losses
        l1 = self.l1_loss(pred, target) * self.l1_weight
        l2 = self.l2_loss(pred, target) * self.l2_weight
        
        # Structure preservation
        grad_loss = self.gradient_loss(pred, target) * self.grad_weight
        
        # Frequency domain loss
        freq_loss = self.frequency_loss(pred, target) * self.freq_weight
        
        # SSIM loss
        ssim_loss = self.ssim_loss(pred, target) * self.ssim_weight
        
        # Perceptual loss
        perceptual_loss = self.vgg(pred, target) * self.perceptual_weight
        
        # Color consistency loss
        color_loss = self.color_consistency_loss(pred, target) * self.color_weight
        
        # Total loss
        total_loss = l1 + l2 + grad_loss + freq_loss + ssim_loss + perceptual_loss + color_loss
        
        return total_loss, {
            'l1': l1.item(),
            'l2': l2.item(),
            'gradient': grad_loss.item(),
            'frequency': freq_loss.item(),
            'ssim': ssim_loss.item(),
            'perceptual': perceptual_loss.item(),
            'color': color_loss.item(),
            'total': total_loss.item()
        }


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']):
        super().__init__()
        
        # Load pre-trained VGG19
        vgg = models.vgg19(pretrained=True).features
        
        # Extract specified layers
        self.layers = layers
        self.features = nn.ModuleDict()
        
        layer_map = {
            'conv1_1': 0, 'conv1_2': 2,
            'conv2_1': 5, 'conv2_2': 7,
            'conv3_1': 10, 'conv3_2': 12, 'conv3_3': 14, 'conv3_4': 16,
            'conv4_1': 19, 'conv4_2': 21, 'conv4_3': 23, 'conv4_4': 25,
            'conv5_1': 28, 'conv5_2': 30, 'conv5_3': 32, 'conv5_4': 34
        }
        
        for layer_name in layers:
            if layer_name in layer_map:
                self.features[layer_name] = nn.Sequential(*list(vgg.children())[:layer_map[layer_name]+1])
        
        # Freeze VGG parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Normalization for ImageNet pre-trained models
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        # Loss weights for different layers
        self.layer_weights = {
            'conv1_2': 1.0,
            'conv2_2': 1.0,
            'conv3_3': 1.0,
            'conv4_3': 1.0
        }
    
    def forward(self, pred, target):
        # Normalize inputs
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        loss = 0.0
        
        for layer_name, model in self.features.items():
            pred_feat = model(pred_norm)
            target_feat = model(target_norm)
            
            # Compute feature loss
            feat_loss = F.mse_loss(pred_feat, target_feat)
            weight = self.layer_weights.get(layer_name, 1.0)
            loss += weight * feat_loss
        
        return loss


class UnderwaterSpecificLoss(nn.Module):
    """Underwater-specific loss functions"""
    def __init__(self):
        super().__init__()
        self.l1_loss = nn.L1Loss()
        
    def color_attenuation_loss(self, pred, target):
        """Loss for color attenuation correction in underwater images"""
        # Red channel typically attenuates most in underwater conditions
        red_weight = 2.0
        green_weight = 1.5
        blue_weight = 1.0
        
        red_loss = self.l1_loss(pred[:, 0:1], target[:, 0:1]) * red_weight
        green_loss = self.l1_loss(pred[:, 1:2], target[:, 1:2]) * green_weight
        blue_loss = self.l1_loss(pred[:, 2:3], target[:, 2:3]) * blue_weight
        
        return red_loss + green_loss + blue_loss
    
    def contrast_enhancement_loss(self, pred, target):
        """Loss for contrast enhancement"""
        # Compute local variance as a proxy for contrast
        kernel = torch.ones(1, 1, 3, 3) / 9.0
        kernel = kernel.to(pred.device)
        
        def local_variance(x):
            # Compute local mean
            mean = F.conv2d(x, kernel, padding=1, groups=1)
            # Compute local variance
            var = F.conv2d(x**2, kernel, padding=1, groups=1) - mean**2
            return var
        
        pred_var = local_variance(pred.mean(dim=1, keepdim=True))
        target_var = local_variance(target.mean(dim=1, keepdim=True))
        
        return self.l1_loss(pred_var, target_var)
    
    def underwater_color_correction_loss(self, pred, target):
        """Specific loss for underwater color correction"""
        # Blue-green dominance correction
        bg_ratio_pred = (pred[:, 1:2] + pred[:, 2:3]) / (pred[:, 0:1] + 1e-6)
        bg_ratio_target = (target[:, 1:2] + target[:, 2:3]) / (target[:, 0:1] + 1e-6)
        
        bg_loss = self.l1_loss(bg_ratio_pred, bg_ratio_target)
        
        # Overall color balance
        color_balance_loss = self.color_attenuation_loss(pred, target)
        
        return bg_loss + color_balance_loss


class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training (optional)"""
    def __init__(self, discriminator, loss_type='lsgan'):
        super().__init__()
        self.discriminator = discriminator
        self.loss_type = loss_type
        
        if loss_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif loss_type == 'vanilla':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(self, pred, real=True):
        """Compute adversarial loss"""
        pred_fake = self.discriminator(pred)
        
        if real:
            target = torch.ones_like(pred_fake)
        else:
            target = torch.zeros_like(pred_fake)
            
        return self.criterion(pred_fake, target)


class CombinedLoss(nn.Module):
    """Combined loss function for the complete model"""
    def __init__(self, device='cuda', use_adversarial=False):
        super().__init__()
        
        self.structure_loss = StructureAwareLoss(device)
        self.underwater_loss = UnderwaterSpecificLoss()
        
        # Loss weights
        self.structure_weight = 1.0
        self.underwater_weight = 0.5
        self.adversarial_weight = 0.1
        
        self.use_adversarial = use_adversarial
        if use_adversarial:
            # Initialize discriminator here if needed
            pass
    
    def forward(self, pred, target, discriminator=None):
        """Compute combined loss"""
        # Structure-aware loss
        structure_loss, structure_dict = self.structure_loss(pred, target)
        structure_loss *= self.structure_weight
        
        # Underwater-specific losses
        color_atten_loss = self.underwater_loss.color_attenuation_loss(pred, target)
        contrast_loss = self.underwater_loss.contrast_enhancement_loss(pred, target)
        underwater_color_loss = self.underwater_loss.underwater_color_correction_loss(pred, target)
        
        underwater_total = (color_atten_loss + contrast_loss + underwater_color_loss) * self.underwater_weight
        
        # Total loss
        total_loss = structure_loss + underwater_total
        
        loss_dict = structure_dict.copy()
        loss_dict.update({
            'color_attenuation': color_atten_loss.item(),
            'contrast': contrast_loss.item(),
            'underwater_color': underwater_color_loss.item(),
            'underwater_total': underwater_total.item(),
            'total_combined': total_loss.item()
        })
        
        # Add adversarial loss if specified
        if self.use_adversarial and discriminator is not None:
            adv_loss = AdversarialLoss(discriminator)(pred) * self.adversarial_weight
            total_loss += adv_loss
            loss_dict['adversarial'] = adv_loss.item()
        
        return total_loss, loss_dict


if __name__ == "__main__":
    # Test loss functions
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Generate test data
    pred = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    
    # Test StructureAwareLoss
    structure_loss = StructureAwareLoss(device)
    loss, loss_dict = structure_loss(pred, target)
    print(f"Structure loss: {loss.item():.4f}")
    print("Loss components:", {k: f"{v:.4f}" for k, v in loss_dict.items()})
    
    # Test CombinedLoss
    combined_loss = CombinedLoss(device)
    loss, loss_dict = combined_loss(pred, target)
    print(f"\nCombined loss: {loss.item():.4f}")
    print("Combined loss components:", {k: f"{v:.4f}" for k, v in loss_dict.items()})
