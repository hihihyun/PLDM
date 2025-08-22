"""
Training script for Underwater Image Enhancement Diffusion Model
"""

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import argparse
import json
import time
from pathlib import Path
import numpy as np
from tqdm import tqdm
import wandb

from models.main_model import UnderwaterEnhancementDiffusion, create_model
from data.dataset import create_dataloaders
from config import get_config
from utils import AverageMeter, save_checkpoint, load_checkpoint, set_seed, save_images


class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        set_seed(config['seed'])
        
        # Create directories
        self.setup_directories()
        
        # Initialize model
        self.model = create_model(config['model']).to(self.device)
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'diffusion_unet'):
            self.model.diffusion_unet.gradient_checkpointing = True
        
        # Setup data loaders
        self.train_loader, self.val_loader = create_dataloaders(config['data'])
        
        # Setup optimizers and schedulers
        self.setup_optimizers()
        
        # Setup logging
        self.setup_logging()
        
        # Setup mixed precision training
        self.use_amp = config.get('use_amp', True)
        if self.use_amp:
            self.scaler = torch.cuda.amp.GradScaler()
            print("✅ Mixed precision training enabled")
        
        # Training state
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.global_step = 0
        
        print(f"Trainer initialized. Device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Mixed precision: {self.use_amp}")
        
        # Clear cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def setup_directories(self):
        """Create necessary directories"""
        self.exp_dir = Path(self.config['exp_dir'])
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.log_dir = self.exp_dir / 'logs'
        self.sample_dir = self.exp_dir / 'samples'
        
        for dir_path in [self.exp_dir, self.checkpoint_dir, self.log_dir, self.sample_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.exp_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def setup_optimizers(self):
        """Setup optimizers and schedulers"""
        # Get model parameters
        vae_params = list(self.model.encoder.parameters()) + list(self.model.decoder.parameters())
        diffusion_params = list(self.model.diffusion_unet.parameters())
        
        # Optimizers
        self.optimizers = {
            'vae': torch.optim.AdamW(
                vae_params, 
                lr=self.config['training']['vae_lr'],
                weight_decay=self.config['training']['weight_decay'],
                betas=(0.9, 0.999)
            ),
            'diffusion': torch.optim.AdamW(
                diffusion_params, 
                lr=self.config['training']['diffusion_lr'],
                weight_decay=self.config['training']['weight_decay'],
                betas=(0.9, 0.999)
            )
        }
        
        # Schedulers
        total_steps = len(self.train_loader) * self.config['training']['num_epochs']
        
        self.schedulers = {
            'vae': torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers['vae'], 
                T_max=total_steps,
                eta_min=self.config['training']['min_lr']
            ),
            'diffusion': torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizers['diffusion'], 
                T_max=total_steps,
                eta_min=self.config['training']['min_lr']
            )
        }
    
    def setup_logging(self):
        """Setup logging with TensorBoard and WandB"""
        # TensorBoard
        self.writer = SummaryWriter(self.log_dir)
        
        # WandB (optional)
        if self.config.get('use_wandb', False):
            wandb.init(
                project=self.config['wandb_project'],
                name=self.config['exp_name'],
                config=self.config
            )
    
    def train_epoch(self):
        """Train for one epoch with memory optimization"""
        self.model.train()
        
        # Metrics
        metrics = {
            'total_loss': AverageMeter(),
            'diffusion_loss': AverageMeter(),
            'kl_loss': AverageMeter(),
            'reconstruction_loss': AverageMeter()
        }
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Clear cache periodically
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Move data to device
            degraded = batch['degraded'].to(self.device, non_blocking=True)
            enhanced = batch['enhanced'].to(self.device, non_blocking=True)
            
            # Normalize to [-1, 1]
            degraded = degraded * 2.0 - 1.0
            enhanced = enhanced * 2.0 - 1.0
            
            # Get preprocessed features if available
            preprocessed = batch.get('preprocessed', None)
            if preprocessed is not None:
                preprocessed = preprocessed.to(self.device, non_blocking=True)
            
            # Zero gradients
            for optimizer in self.optimizers.values():
                optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    loss, loss_dict = self.model.training_step(degraded, enhanced, preprocessed)
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip'] > 0:
                    self.scaler.unscale_(self.optimizers['diffusion'])
                    self.scaler.unscale_(self.optimizers['vae'])
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                
                # Optimizer step with scaling
                for optimizer in self.optimizers.values():
                    self.scaler.step(optimizer)
                self.scaler.update()
                
            else:
                # Regular training
                loss, loss_dict = self.model.training_step(degraded, enhanced, preprocessed)
                
                # Backward pass
                loss.backward()
                
                # Gradient clipping
                if self.config['training']['grad_clip'] > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['training']['grad_clip'])
                
                # Optimizer step
                for optimizer in self.optimizers.values():
                    optimizer.step()
            
            # Scheduler step
            for scheduler in self.schedulers.values():
                scheduler.step()
            
            # Update metrics
            batch_size = degraded.size(0)
            for key, value in loss_dict.items():
                if key in metrics:
                    metrics[key].update(value, batch_size)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{metrics['total_loss'].avg:.4f}",
                'diffusion': f"{metrics['diffusion_loss'].avg:.4f}",
                'lr': f"{self.optimizers['diffusion'].param_groups[0]['lr']:.2e}",
                'mem': f"{torch.cuda.memory_allocated()/1e9:.1f}GB" if torch.cuda.is_available() else "N/A"
            })
            
            # Log to tensorboard
            if self.global_step % self.config['training']['log_freq'] == 0:
                self.log_metrics('train', loss_dict, self.global_step)
            
            # Validation and sampling
            if self.global_step % self.config['training']['val_freq'] == 0:
                val_loss = self.validate()
                self.save_samples(degraded[:2], enhanced[:2])  # Reduced from 4 to 2 samples
                self.model.train()  # Return to training mode
            
            self.global_step += 1
        
        return {key: meter.avg for key, meter in metrics.items()}
    
    @torch.no_grad()
    def validate(self):
        """Validation step with robust error handling"""
        self.model.eval()
        
        val_metrics = {
            'total_loss': AverageMeter(),
            'l1': AverageMeter(),
            'ssim': AverageMeter(),
            'perceptual': AverageMeter()
        }
        
        valid_batches = 0
        
        for batch_idx, batch in enumerate(self.val_loader):
            if batch_idx >= self.config['training']['max_val_batches']:
                break
            
            # Check if batch has both degraded and enhanced images
            if 'enhanced' not in batch:
                print(f"⚠️  Validation batch {batch_idx} missing 'enhanced' key, skipping...")
                continue
                
            try:
                degraded = batch['degraded'].to(self.device, non_blocking=True)
                enhanced = batch['enhanced'].to(self.device, non_blocking=True)
                
                # Check tensor validity
                if degraded.numel() == 0 or enhanced.numel() == 0:
                    print(f"⚠️  Empty tensors in batch {batch_idx}, skipping...")
                    continue
                
                # Normalize
                degraded = degraded * 2.0 - 1.0
                enhanced = enhanced * 2.0 - 1.0
                
                # Validation step with memory efficiency
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        val_loss, val_loss_dict, pred_enhanced = self.model.validation_step(degraded, enhanced)
                else:
                    val_loss, val_loss_dict, pred_enhanced = self.model.validation_step(degraded, enhanced)
                
                # Update metrics
                batch_size = degraded.size(0)
                for key, value in val_loss_dict.items():
                    if key in val_metrics:
                        val_metrics[key].update(value, batch_size)
                
                valid_batches += 1
                
            except Exception as e:
                print(f"⚠️  Error in validation batch {batch_idx}: {e}")
                continue
        
        if valid_batches == 0:
            print("❌ No valid validation batches found!")
            return float('inf')
        
        # Log validation metrics
        avg_val_loss = val_metrics['total_loss'].avg
        self.log_metrics('val', {key: meter.avg for key, meter in val_metrics.items()}, self.global_step)
        
        print(f"Validation Loss: {avg_val_loss:.4f} (from {valid_batches} valid batches)")
        
        # Save best model
        if avg_val_loss < self.best_val_loss:
            self.best_val_loss = avg_val_loss
            self.save_checkpoint(is_best=True)
        
        return avg_val_loss
    
    @torch.no_grad()
    def save_samples(self, degraded, enhanced, num_samples=2):
        """Save sample images with error handling"""
        try:
            self.model.eval()
            
            # Ensure we have valid data
            if degraded.numel() == 0:
                print("⚠️  Empty degraded tensor, skipping sample save")
                return
                
            if enhanced.numel() == 0:
                print("⚠️  Empty enhanced tensor, skipping sample save")
                return
            
            # Limit number of samples to avoid memory issues
            actual_samples = min(num_samples, degraded.size(0), enhanced.size(0))
            degraded = degraded[:actual_samples]
            enhanced = enhanced[:actual_samples]
            
            # Generate enhanced images
            with torch.no_grad():
                if self.use_amp:
                    with torch.cuda.amp.autocast():
                        pred_enhanced = self.model.sample(degraded, num_steps=10)  # Fast sampling for training
                else:
                    pred_enhanced = self.model.sample(degraded, num_steps=10)
            
            # Denormalize images to [0, 1]
            degraded = torch.clamp((degraded + 1.0) / 2.0, 0, 1)
            enhanced = torch.clamp((enhanced + 1.0) / 2.0, 0, 1)
            pred_enhanced = torch.clamp((pred_enhanced + 1.0) / 2.0, 0, 1)
            
            # Save images
            save_path = self.sample_dir / f'samples_step_{self.global_step}.png'
            save_images(degraded, enhanced, pred_enhanced, save_path)
            
            # Log to tensorboard
            if self.writer:
                self.writer.add_images('samples/degraded', degraded, self.global_step)
                self.writer.add_images('samples/enhanced_gt', enhanced, self.global_step)
                self.writer.add_images('samples/enhanced_pred', pred_enhanced, self.global_step)
                
        except Exception as e:
            print(f"⚠️  Error saving samples: {e}")
            print("Continuing training without sample images...")
    
    def log_metrics(self, phase, metrics, step):
        """Log metrics to tensorboard and wandb"""
        # TensorBoard
        for key, value in metrics.items():
            self.writer.add_scalar(f'{phase}/{key}', value, step)
        
        # Learning rates
        for name, optimizer in self.optimizers.items():
            self.writer.add_scalar(f'lr/{name}', optimizer.param_groups[0]['lr'], step)
        
        # WandB
        if self.config.get('use_wandb', False):
            wandb_metrics = {f'{phase}/{key}': value for key, value in metrics.items()}
            wandb_metrics.update({f'lr/{name}': opt.param_groups[0]['lr'] for name, opt in self.optimizers.items()})
            wandb.log(wandb_metrics, step=step)
    
    def save_checkpoint(self, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizers': {name: opt.state_dict() for name, opt in self.optimizers.items()},
            'schedulers': {name: sch.state_dict() for name, sch in self.schedulers.items()},
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{self.current_epoch}.pth'
        torch.save(checkpoint, checkpoint_path)
        
        # Best model checkpoint
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pth'
            torch.save(checkpoint, best_path)
            print(f"Best model saved with validation loss: {self.best_val_loss:.4f}")
        
        # Latest checkpoint
        latest_path = self.checkpoint_dir / 'latest.pth'
        torch.save(checkpoint, latest_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load model checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        for name, optimizer in self.optimizers.items():
            if name in checkpoint['optimizers']:
                optimizer.load_state_dict(checkpoint['optimizers'][name])
        
        for name, scheduler in self.schedulers.items():
            if name in checkpoint['schedulers']:
                scheduler.load_state_dict(checkpoint['schedulers'][name])
        
        self.current_epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        self.best_val_loss = checkpoint['best_val_loss']
        
        print(f"Checkpoint loaded from {checkpoint_path}")
        print(f"Resuming from epoch {self.current_epoch}, step {self.global_step}")
    
    def train(self):
        """Main training loop"""
        print("Starting training...")
        
        for epoch in range(self.current_epoch, self.config['training']['num_epochs']):
            self.current_epoch = epoch
            
            print(f"\nEpoch {epoch}/{self.config['training']['num_epochs']}")
            
            # Train epoch
            train_metrics = self.train_epoch()
            
            # Print epoch summary
            print(f"Train Loss: {train_metrics['total_loss']:.4f}")
            
            # Save checkpoint
            if epoch % self.config['training']['save_freq'] == 0:
                self.save_checkpoint()
            
            # Early stopping
            if hasattr(self, 'early_stopping'):
                if self.early_stopping.early_stop:
                    print("Early stopping triggered")
                    break
        
        print("Training completed!")
        
        # Close logging
        self.writer.close()
        if self.config.get('use_wandb', False):
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description='Train Underwater Image Enhancement Model')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--resume', type=str, default=None, help='Resume from checkpoint')
    parser.add_argument('--exp_name', type=str, default=None, help='Experiment name')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # Load config
    config = get_config(args.config)
    
    # Override config with command line arguments
    if args.exp_name:
        config['exp_name'] = args.exp_name
    if args.device:
        config['device'] = args.device
    
    # Update experiment directory
    config['exp_dir'] = f"experiments/{config['exp_name']}"
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Start training
    try:
        trainer.train()
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint()
    except Exception as e:
        print(f"Training failed with error: {e}")
        trainer.save_checkpoint()
        raise


if __name__ == "__main__":
    main()
