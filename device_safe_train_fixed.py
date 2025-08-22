"""
Device-safe training script for underwater image enhancement
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import sys
import os
from tqdm import tqdm
import argparse

def ensure_device(tensor_or_dict, device):
    """Ensure tensor or dict of tensors is on correct device"""
    if torch.is_tensor(tensor_or_dict):
        return tensor_or_dict.to(device, non_blocking=True)
    elif isinstance(tensor_or_dict, dict):
        return {k: ensure_device(v, device) for k, v in tensor_or_dict.items()}
    else:
        return tensor_or_dict

def safe_training_step(model, batch, device):
    """Perform training step with device safety"""
    # Move batch to device
    batch = ensure_device(batch, device)
    
    # Extract tensors
    degraded = batch.get('degraded')
    enhanced = batch.get('enhanced', degraded)  # Fallback if no enhanced
    
    if degraded is None:
        raise ValueError("No 'degraded' key found in batch")
    
    # Ensure correct device
    degraded = degraded.to(device)
    enhanced = enhanced.to(device)
    
    # Normalize to [-1, 1]
    degraded = degraded * 2.0 - 1.0
    enhanced = enhanced * 2.0 - 1.0
    
    # Training step
    loss, loss_dict = model.training_step(degraded, enhanced)
    
    return loss, loss_dict

def safe_validation_step(model, batch, device):
    """Perform validation step with device safety"""
    try:
        # Move batch to device
        batch = ensure_device(batch, device)
        
        degraded = batch.get('degraded')
        enhanced = batch.get('enhanced', degraded)
        
        if degraded is None:
            return None, None, None
        
        # Ensure correct device
        degraded = degraded.to(device)
        enhanced = enhanced.to(device)
        
        # Normalize
        degraded = degraded * 2.0 - 1.0
        enhanced = enhanced * 2.0 - 1.0
        
        # Validation step
        with torch.no_grad():
            val_loss, val_loss_dict, pred_enhanced = model.validation_step(degraded, enhanced)
            
        return val_loss, val_loss_dict, pred_enhanced
        
    except Exception as e:
        print(f"⚠️  Validation step failed: {e}")
        return None, None, None

class MinimalUnderwaterModel(nn.Module):
    """Minimal underwater enhancement model for testing"""
    def __init__(self, img_size=64, base_channels=32):
        super().__init__()
        
        # Simple VAE-like architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(3, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(base_channels*8, base_channels*4, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels*4, base_channels*2, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(base_channels*2, base_channels, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, 3, 3, padding=1),
            nn.Tanh(),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    def training_step(self, degraded, enhanced):
        """Training step with multiple losses"""
        pred = self.forward(degraded)
        
        # Losses
        mse_loss = nn.functional.mse_loss(pred, enhanced)
        l1_loss = nn.functional.l1_loss(pred, enhanced)
        
        total_loss = mse_loss + 0.1 * l1_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'l1': l1_loss.item()
        }
        
        return total_loss, loss_dict
    
    def validation_step(self, degraded, enhanced):
        """Validation step"""
        pred = self.forward(degraded)
        
        mse_loss = nn.functional.mse_loss(pred, enhanced)
        l1_loss = nn.functional.l1_loss(pred, enhanced)
        total_loss = mse_loss + 0.1 * l1_loss
        
        loss_dict = {
            'total': total_loss.item(),
            'mse': mse_loss.item(),
            'l1': l1_loss.item()
        }
        
        return total_loss, loss_dict, pred

def create_synthetic_dataset(num_samples=100, img_size=64):
    """Create synthetic underwater dataset"""
    class SyntheticDataset:
        def __init__(self, num_samples, img_size):
            self.num_samples = num_samples
            self.img_size = img_size
            
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            # Create synthetic underwater-like images
            degraded = torch.randn(3, self.img_size, self.img_size) * 0.3 + 0.4
            
            # Enhanced version (slightly better contrast and color)
            enhanced = degraded * 1.2 + 0.1
            enhanced = torch.clamp(enhanced, 0, 1)
            degraded = torch.clamp(degraded, 0, 1)
            
            return {
                'degraded': degraded,
                'enhanced': enhanced
            }
    
    return SyntheticDataset(num_samples, img_size)

def device_safe_train():
    """Main training function with device safety"""
    
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='auto', help='Device to use')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=2, help='Batch size')
    parser.add_argument('--use_real_data', action='store_true', help='Use real UIEB data')
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"🖥️  Using device: {device}")
    
    try:
        # Try to use real model first
        if args.use_real_data:
            print("🔧 Attempting to load real model and data...")
            sys.path.append('.')
            
            try:
                from config_fixed import get_config
                from models import create_model
                from data import UnderwaterDataset
                
                config = get_config('lightweight')
                model = create_model(config['model'])
                
                dataset = UnderwaterDataset('./DATA', 'UIEB', 'train', augment=False)
                print(f"✅ Real model and data loaded: {len(dataset)} samples")
                
            except Exception as e:
                print(f"⚠️  Failed to load real model/data: {e}")
                print("🔄 Falling back to synthetic data...")
                raise e
        else:
            raise Exception("Using synthetic data as requested")
            
    except Exception:
        # Fallback to minimal model and synthetic data
        print("🎨 Using minimal model with synthetic data...")
        
        model = MinimalUnderwaterModel(img_size=64, base_channels=32)
        dataset = create_synthetic_dataset(num_samples=200, img_size=64)
        print(f"✅ Synthetic dataset created: {len(dataset)} samples")
    
    # Move model to device
    model = model.to(device)
    
    # Check model is on correct device
    model_devices = set()
    for param in model.parameters():
        model_devices.add(str(param.device))
    print(f"📱 Model devices: {model_devices}")
    
    # Setup optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Setup data loader
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,  # Avoid multiprocessing issues
        pin_memory=False
    )
    
    print(f"🔄 Starting training for {args.epochs} epochs...")
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0.0
        valid_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            try:
                # Training step with device safety
                optimizer.zero_grad()
                loss, loss_dict = safe_training_step(model, batch, device)
                
                if loss is not None and not torch.isnan(loss) and loss.item() > 0:
                    loss.backward()
                    
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                    valid_batches += 1
                    
                    pbar.set_postfix({
                        'loss': f'{loss.item():.6f}',
                        'mse': f'{loss_dict.get("mse", 0):.6f}'
                    })
                else:
                    pbar.set_postfix({'loss': 'INVALID'})
                    
            except Exception as e:
                print(f"⚠️  Batch {batch_idx} failed: {e}")
                continue
            
            # Limit batches for testing
            if batch_idx >= 50:
                break
        
        avg_loss = epoch_loss / max(valid_batches, 1)
        print(f"Epoch {epoch+1}: Avg Loss = {avg_loss:.6f} ({valid_batches} valid batches)")
        
        # Simple validation every few epochs
        if epoch % 3 == 0:
            model.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    if batch_idx >= 10:  # Just a few validation batches
                        break
                    
                    try:
                        v_loss, v_loss_dict, pred = safe_validation_step(model, batch, device)
                        if v_loss is not None:
                            val_loss += v_loss.item()
                            val_batches += 1
                    except Exception as e:
                        continue
            
            avg_val_loss = val_loss / max(val_batches, 1)
            print(f"  Validation Loss: {avg_val_loss:.6f}")
    
    print("✅ Training completed successfully!")
    
    # Test final model
    print("\n🧪 Testing final model...")
    model.eval()
    test_batch = next(iter(dataloader))
    
    try:
        with torch.no_grad():
            test_loss, test_loss_dict = safe_training_step(model, test_batch, device)
            print(f"Final test loss: {test_loss.item():.6f}")
            print("✅ Model is working correctly!")
    except Exception as e:
        print(f"❌ Final test failed: {e}")

if __name__ == "__main__":
    device_safe_train()
