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
        pred = torch.clamp(pred, -1.0, 1.0)
        target = torch.clamp(target, -1.0, 1.0)

        pred = pred.float()
        target = target.float()
        
        try:
            pred_fft = torch.fft.fft2(pred, norm='ortho')
            target_fft = torch.fft.fft2(target, norm='ortho')
            
            loss = self.l1_loss(pred_fft.real, target_fft.real) + \
                   self.l1_loss(pred_fft.imag, target_fft.imag)
            
            _, _, h, w = pred.shape
            center_h, center_w = h // 2, w // 2
            
            y, x = torch.meshgrid(torch.arange(h, device=pred.device), torch.arange(w, device=pred.device), indexing='ij')
            dist = torch.sqrt((y - center_h)**2 + (x - center_w)**2)
            high_freq_mask = (dist > min(h, w) * 0.1).float()
            
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
        
        kernel = gaussian_kernel(window_size).expand(pred.shape[1], 1, window_size, window_size).to(pred.device)
        
        C1 = 0.01 ** 2
        C2 = 0.03 ** 2
        
        mu_x = F.conv2d(pred, kernel, padding=window_size//2, groups=pred.shape[1])
        mu_y = F.conv2d(target, kernel, padding=window_size//2, groups=target.shape[1])
        
        mu_x_sq = mu_x ** 2
        mu_y_sq = mu_y ** 2
        mu_xy = mu_x * mu_y
        
        sigma_x_sq = F.conv2d(pred**2, kernel, padding=window_size//2, groups=pred.shape[1]) - mu_x_sq
        sigma_y_sq = F.conv2d(target**2, kernel, padding=window_size//2, groups=target.shape[1]) - mu_y_sq
        sigma_xy = F.conv2d(pred * target, kernel, padding=window_size//2, groups=pred.shape[1]) - mu_xy
        
        ssim_numerator = ((2 * mu_xy + C1) * (2 * sigma_xy + C2))
        ssim_denominator = ((mu_x_sq + mu_y_sq + C1) * (sigma_x_sq + sigma_y_sq + C2))
        ssim = ssim_numerator / (ssim_denominator + 1e-6)
        
        return 1 - ssim.mean()
    
    def color_consistency_loss(self, pred, target):
        """Color consistency loss for underwater images"""
        def rgb_to_lab_approx(rgb):
            r, g, b = rgb[:, 0:1], rgb[:, 1:2], rgb[:, 2:3]
            l = 0.299 * r + 0.587 * g + 0.114 * b
            a = 0.5 * (r - g)
            b_chan = 0.5 * (g - b)
            return torch.cat([l, a, b_chan], dim=1)
        
        pred_lab = rgb_to_lab_approx(pred)
        target_lab = rgb_to_lab_approx(target)
        
        color_loss = self.l1_loss(pred_lab, target_lab)
        
        pred_chroma = torch.sqrt(pred_lab[:, 1:2]**2 + pred_lab[:, 2:3]**2 + 1e-6)
        target_chroma = torch.sqrt(target_lab[:, 1:2]**2 + target_lab[:, 2:3]**2 + 1e-6)
        chroma_loss = self.l1_loss(pred_chroma, target_chroma)
        
        return color_loss + 0.5 * chroma_loss
    
    def forward(self, pred, target):
        """Combined loss computation"""
        l1 = self.l1_loss(pred, target) * self.l1_weight
        l2 = self.l2_loss(pred, target) * self.l2_weight
        grad_loss = self.gradient_loss(pred, target) * self.grad_weight
        freq_loss = self.frequency_loss(pred, target) * self.freq_weight
        ssim_loss = self.ssim_loss(pred, target) * self.ssim_weight
        perceptual_loss = self.vgg(pred, target) * self.perceptual_weight
        color_loss = self.color_consistency_loss(pred, target) * self.color_weight
        
        total_loss = l1 + l2 + grad_loss + freq_loss + ssim_loss + perceptual_loss + color_loss
        
        return total_loss, {
            'l1': l1.item(),
            'l2': l2.item(),
            'gradient': grad_loss.item(),
            'frequency': freq_loss.item(),
            'ssim': ssim_loss.item(),
            'perceptual': perceptual_loss.item(),
            'color': color_loss.item(),
            'total_recon': total_loss.item()
        }


class VGGPerceptualLoss(nn.Module):
    """VGG-based perceptual loss"""
    def __init__(self, layers=['conv1_2', 'conv2_2', 'conv3_3', 'conv4_3']):
        super().__init__()
        
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1).features
        
        self.layers = layers
        self.features = nn.ModuleDict()
        
        layer_map = {
            'conv1_2': 4, 'conv2_2': 9, 'conv3_3': 16, 'conv4_3': 25
        }
        
        for layer_name in layers:
            if layer_name in layer_map:
                self.features[layer_name] = nn.Sequential(*list(vgg.children())[:layer_map[layer_name]+1])
        
        for param in self.parameters():
            param.requires_grad = False
        
        self.normalize = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        self.layer_weights = {
            'conv1_2': 1.0, 'conv2_2': 1.0, 'conv3_3': 1.0, 'conv4_3': 1.0
        }
    
    def forward(self, pred, target):
        pred_norm = self.normalize((pred + 1.0) / 2.0)
        target_norm = self.normalize((target + 1.0) / 2.0)
        
        loss = 0.0
        
        for layer_name, model in self.features.items():
            pred_feat = model(pred_norm)
            target_feat = model(target_norm)
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
        red_weight, green_weight, blue_weight = 2.0, 1.5, 1.0
        red_loss = self.l1_loss(pred[:, 0:1], target[:, 0:1]) * red_weight
        green_loss = self.l1_loss(pred[:, 1:2], target[:, 1:2]) * green_weight
        blue_loss = self.l1_loss(pred[:, 2:3], target[:, 2:3]) * blue_weight
        return red_loss + green_loss + blue_loss
    
    def contrast_enhancement_loss(self, pred, target):
        kernel = torch.ones(1, 1, 3, 3, device=pred.device) / 9.0
        def local_variance(x):
            mean = F.conv2d(x, kernel, padding=1, groups=1)
            var = F.conv2d(x**2, kernel, padding=1, groups=1) - mean**2
            return var
        pred_var = local_variance(pred.mean(dim=1, keepdim=True))
        target_var = local_variance(target.mean(dim=1, keepdim=True))
        return self.l1_loss(pred_var, target_var)

# 👇 --- 누락되었던 부분 시작 --- 👇
class AdversarialLoss(nn.Module):
    """Adversarial loss for GAN training (optional)"""
    def __init__(self, discriminator, loss_type='lsgan'):
        super().__init__()
        self.discriminator = discriminator
        self.loss_type = loss_type
        self.criterion = nn.MSELoss() if loss_type == 'lsgan' else nn.BCEWithLogitsLoss()
    
    def forward(self, pred, real=True):
        pred_fake = self.discriminator(pred)
        target = torch.ones_like(pred_fake) if real else torch.zeros_like(pred_fake)
        return self.criterion(pred_fake, target)


class CombinedLoss(nn.Module):
    """Combined loss function for the complete model"""
    def __init__(self, device='cuda', use_adversarial=False):
        super().__init__()
        self.structure_loss = StructureAwareLoss(device)
        self.underwater_loss = UnderwaterSpecificLoss()
        self.structure_weight = 1.0
        self.underwater_weight = 0.5
        self.adversarial_weight = 0.1
        self.use_adversarial = use_adversarial
    
    def forward(self, pred, target, discriminator=None):
        structure_loss, structure_dict = self.structure_loss(pred, target)
        structure_loss *= self.structure_weight
        
        color_atten_loss = self.underwater_loss.color_attenuation_loss(pred, target)
        contrast_loss = self.underwater_loss.contrast_enhancement_loss(pred, target)
        underwater_total = (color_atten_loss + contrast_loss) * self.underwater_weight
        
        total_loss = structure_loss + underwater_total
        
        loss_dict = structure_dict.copy()
        loss_dict.update({
            'color_attenuation': color_atten_loss.item(),
            'contrast': contrast_loss.item(),
            'underwater_total': underwater_total.item(),
            'total_combined': total_loss.item()
        })
        
        if self.use_adversarial and discriminator is not None:
            adv_loss = AdversarialLoss(discriminator)(pred) * self.adversarial_weight
            total_loss += adv_loss
            loss_dict['adversarial'] = adv_loss.item()
        
        return total_loss, loss_dict
# 👆 --- 누락되었던 부분 끝 --- 👆

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred = torch.randn(2, 3, 256, 256).to(device)
    target = torch.randn(2, 3, 256, 256).to(device)
    
    structure_loss_fn = StructureAwareLoss(device)
    loss, loss_dict = structure_loss_fn(pred, target)
    print(f"Structure loss: {loss.item():.4f}")
    print("Loss components:", {k: f"{v:.4f}" for k, v in loss_dict.items()})
    
    combined_loss_fn = CombinedLoss(device)
    loss, loss_dict = combined_loss_fn(pred, target)
    print(f"\nCombined loss: {loss.item():.4f}")
    print("Combined loss components:", {k: f"{v:.4f}" for k, v in loss_dict.items()})