"""
Utility functions for Underwater Image Enhancement Diffusion Model
"""

import os
import torch
import numpy as np
import random
from PIL import Image
import torchvision.transforms as transforms
import torch.nn.functional as F
from pathlib import Path
import matplotlib.pyplot as plt
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import cv2
import math


def set_seed(seed):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_checkpoint(state, checkpoint_dir, filename='checkpoint.pth', is_best=False):
    """Save model checkpoint"""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = checkpoint_dir / filename
    torch.save(state, filepath)
    
    if is_best:
        best_filepath = checkpoint_dir / 'best_model.pth'
        torch.save(state, best_filepath)
    
    return filepath


def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
    """Load model checkpoint"""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load model state
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model state loaded from {checkpoint_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Model loaded from {checkpoint_path}")
    
    # Load optimizer state
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print("Optimizer state loaded")
    
    # Load scheduler state
    if scheduler is not None and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        print("Scheduler state loaded")
    
    # Return additional info
    epoch = checkpoint.get('epoch', 0)
    best_loss = checkpoint.get('best_loss', float('inf'))
    
    return epoch, best_loss


def load_image(image_path, mode='RGB'):
    """Load image from path"""
    image = Image.open(image_path).convert(mode)
    return image


def save_single_image(tensor, save_path, normalize=True):
    """Save a single image tensor"""
    if normalize:
        # Assuming tensor is in [-1, 1], normalize to [0, 1]
        tensor = (tensor + 1.0) / 2.0
        tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL and save
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    
    pil_image = transforms.ToPILImage()(tensor.cpu())
    pil_image.save(save_path)


def save_images(degraded, enhanced_gt, enhanced_pred, save_path, normalize=True):
    """Save comparison of degraded, ground truth, and predicted images"""
    if normalize:
        degraded = torch.clamp(degraded, 0, 1)
        enhanced_gt = torch.clamp(enhanced_gt, 0, 1)
        enhanced_pred = torch.clamp(enhanced_pred, 0, 1)
    
    batch_size = degraded.size(0)
    
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4 * batch_size))
    if batch_size == 1:
        axes = axes.reshape(1, -1)
    
    titles = ['Degraded', 'Ground Truth', 'Enhanced']
    
    for i in range(batch_size):
        images = [degraded[i], enhanced_gt[i], enhanced_pred[i]]
        
        for j, (img, title) in enumerate(zip(images, titles)):
            img_np = img.permute(1, 2, 0).cpu().numpy()
            axes[i, j].imshow(img_np)
            axes[i, j].set_title(f'{title} (Sample {i+1})')
            axes[i, j].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def calculate_psnr(pred, target, max_val=1.0):
    """Calculate Peak Signal-to-Noise Ratio"""
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    # Convert to HWC if needed
    if pred.ndim == 3 and pred.shape[0] == 3:
        pred = pred.transpose(1, 2, 0)
    if target.ndim == 3 and target.shape[0] == 3:
        target = target.transpose(1, 2, 0)
    
    return peak_signal_noise_ratio(target, pred, data_range=max_val)


def calculate_ssim(pred, target, max_val=1.0):
    """Calculate Structural Similarity Index"""
    if torch.is_tensor(pred):
        pred = pred.cpu().numpy()
    if torch.is_tensor(target):
        target = target.cpu().numpy()
    
    # Convert to HWC if needed
    if pred.ndim == 3 and pred.shape[0] == 3:
        pred = pred.transpose(1, 2, 0)
    if target.ndim == 3 and target.shape[0] == 3:
        target = target.transpose(1, 2, 0)
    
    # Handle multichannel
    if pred.ndim == 3:
        return structural_similarity(target, pred, data_range=max_val, multichannel=True, channel_axis=2)
    else:
        return structural_similarity(target, pred, data_range=max_val)


def calculate_lpips(pred, target, lpips_model):
    """Calculate LPIPS (Learned Perceptual Image Patch Similarity)"""
    # Note: Requires lpips package: pip install lpips
    if torch.is_tensor(pred) and torch.is_tensor(target):
        # Ensure tensors are in [-1, 1] range
        pred = pred * 2.0 - 1.0 if pred.max() <= 1.0 else pred
        target = target * 2.0 - 1.0 if target.max() <= 1.0 else target
        
        with torch.no_grad():
            lpips_score = lpips_model(pred, target)
        
        return lpips_score.item()
    else:
        raise ValueError("LPIPS requires tensor inputs")


def calculate_metrics(pred, target, max_val=1.0):
    """Calculate common image quality metrics"""
    metrics = {}
    
    # PSNR
    try:
        metrics['psnr'] = calculate_psnr(pred, target, max_val)
    except Exception as e:
        print(f"PSNR calculation failed: {e}")
        metrics['psnr'] = 0.0
    
    # SSIM
    try:
        metrics['ssim'] = calculate_ssim(pred, target, max_val)
    except Exception as e:
        print(f"SSIM calculation failed: {e}")
        metrics['ssim'] = 0.0
    
    # MSE and MAE
    if torch.is_tensor(pred) and torch.is_tensor(target):
        mse = F.mse_loss(pred, target).item()
        mae = F.l1_loss(pred, target).item()
        metrics['mse'] = mse
        metrics['mae'] = mae
    
    return metrics


def underwater_color_metrics(pred, target):
    """Calculate underwater-specific color metrics"""
    def rgb_to_lab(rgb):
        """Simplified RGB to LAB conversion"""
        if torch.is_tensor(rgb):
            rgb = rgb.cpu().numpy()
        
        # Simple approximation
        if rgb.ndim == 3 and rgb.shape[0] == 3:
            rgb = rgb.transpose(1, 2, 0)
        
        # Normalize to [0, 1] if needed
        if rgb.max() > 1.0:
            rgb = rgb / 255.0
        
        # Approximate LAB conversion
        r, g, b = rgb[:, :, 0], rgb[:, :, 1], rgb[:, :, 2]
        l = 0.299 * r + 0.587 * g + 0.114 * b
        a = 0.5 * (r - g)
        b_chan = 0.5 * (g - b)
        
        return np.stack([l, a, b_chan], axis=2)
    
    pred_lab = rgb_to_lab(pred)
    target_lab = rgb_to_lab(target)
    
    # Color difference in LAB space
    color_diff = np.mean(np.sqrt(np.sum((pred_lab - target_lab)**2, axis=2)))
    
    # Underwater-specific metrics
    if torch.is_tensor(pred):
        pred_np = pred.cpu().numpy()
        target_np = target.cpu().numpy()
        
        if pred_np.ndim == 3 and pred_np.shape[0] == 3:
            pred_np = pred_np.transpose(1, 2, 0)
            target_np = target_np.transpose(1, 2, 0)
    else:
        pred_np = pred
        target_np = target
    
    # Blue-green dominance ratio
    pred_bg_ratio = (pred_np[:, :, 1] + pred_np[:, :, 2]) / (pred_np[:, :, 0] + 1e-6)
    target_bg_ratio = (target_np[:, :, 1] + target_np[:, :, 2]) / (target_np[:, :, 0] + 1e-6)
    bg_diff = np.mean(np.abs(pred_bg_ratio - target_bg_ratio))
    
    return {
        'color_diff_lab': color_diff,
        'bg_ratio_diff': bg_diff
    }


def visualize_training_progress(log_dir, save_path=None):
    """Visualize training progress from logs"""
    # This would read tensorboard logs and create plots
    # Implementation depends on specific logging format
    pass


def create_gif_from_images(image_paths, output_path, duration=0.5):
    """Create GIF from a list of image paths"""
    images = []
    for path in image_paths:
        img = Image.open(path)
        images.append(img)
    
    images[0].save(
        output_path,
        save_all=True,
        append_images=images[1:],
        duration=duration * 1000,  # Convert to milliseconds
        loop=0
    )


def resize_image_aspect_ratio(image, target_size, method='center_crop'):
    """Resize image while maintaining aspect ratio"""
    if isinstance(image, torch.Tensor):
        # Convert tensor to PIL for processing
        if image.dim() == 4:
            image = image.squeeze(0)
        pil_image = transforms.ToPILImage()(image)
        was_tensor = True
    else:
        pil_image = image
        was_tensor = False
    
    original_size = pil_image.size
    target_w, target_h = target_size
    
    if method == 'center_crop':
        # Calculate scaling factor to fit the smaller dimension
        scale = max(target_w / original_size[0], target_h / original_size[1])
        
        # Resize image
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        resized = pil_image.resize(new_size, Image.LANCZOS)
        
        # Center crop
        left = (new_size[0] - target_w) // 2
        top = (new_size[1] - target_h) // 2
        right = left + target_w
        bottom = top + target_h
        
        result = resized.crop((left, top, right, bottom))
    
    elif method == 'pad':
        # Resize to fit within target size
        scale = min(target_w / original_size[0], target_h / original_size[1])
        new_size = (int(original_size[0] * scale), int(original_size[1] * scale))
        resized = pil_image.resize(new_size, Image.LANCZOS)
        
        # Create new image with padding
        result = Image.new('RGB', target_size, (0, 0, 0))
        paste_x = (target_w - new_size[0]) // 2
        paste_y = (target_h - new_size[1]) // 2
        result.paste(resized, (paste_x, paste_y))
    
    else:
        # Simple resize (may distort aspect ratio)
        result = pil_image.resize(target_size, Image.LANCZOS)
    
    # Convert back to tensor if input was tensor
    if was_tensor:
        result = transforms.ToTensor()(result)
    
    return result


def count_parameters(model):
    """Count the number of parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }


def get_model_size_mb(model):
    """Get model size in MB"""
    param_size = 0
    buffer_size = 0
    
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    
    size_mb = (param_size + buffer_size) / 1024 / 1024
    return size_mb


def print_model_info(model):
    """Print comprehensive model information"""
    param_counts = count_parameters(model)
    size_mb = get_model_size_mb(model)
    
    print("=" * 50)
    print("MODEL INFORMATION")
    print("=" * 50)
    print(f"Total parameters: {param_counts['total']:,}")
    print(f"Trainable parameters: {param_counts['trainable']:,}")
    print(f"Non-trainable parameters: {param_counts['non_trainable']:,}")
    print(f"Model size: {size_mb:.2f} MB")
    print("=" * 50)


def setup_experiment_dir(exp_name, base_dir='experiments'):
    """Setup experiment directory structure"""
    exp_dir = Path(base_dir) / exp_name
    
    # Create subdirectories
    subdirs = ['checkpoints', 'logs', 'samples', 'configs', 'results']
    for subdir in subdirs:
        (exp_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    return exp_dir


class EarlyStopping:
    """Early stopping utility"""
    def __init__(self, patience=20, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.early_stop = True
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
    
    def save_checkpoint(self, model):
        """Save the best model weights"""
        self.best_weights = model.state_dict()


if __name__ == "__main__":
    # Test utility functions
    print("Testing utility functions...")
    
    # Test image operations
    dummy_image = torch.randn(3, 256, 256)
    dummy_target = torch.randn(3, 256, 256)
    
    # Test metrics
    metrics = calculate_metrics(dummy_image, dummy_target)
    print("Metrics:", metrics)
    
    # Test model info (would need an actual model)
    print("Utility functions tested successfully!")
