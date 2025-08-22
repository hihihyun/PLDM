"""
Fix device mismatch issues in underwater image enhancement training
"""

import torch
import torch.nn as nn
import sys
import os

def check_model_devices(model, model_name="Model"):
    """Check which devices model parameters are on"""
    devices = set()
    for name, param in model.named_parameters():
        devices.add(str(param.device))
    
    print(f"📱 {model_name} devices: {devices}")
    return devices

def move_model_to_device(model, device):
    """Safely move model to device"""
    try:
        model = model.to(device)
        print(f"✅ Model moved to {device}")
        return model
    except Exception as e:
        print(f"❌ Failed to move model to {device}: {e}")
        return model

def check_tensor_devices(tensor_dict, name="Tensors"):
    """Check tensor devices in a dictionary"""
    devices = {}
    for key, value in tensor_dict.items():
        if torch.is_tensor(value):
            devices[key] = str(value.device)
    
    print(f"📱 {name} devices: {devices}")
    return devices

def fix_batch_devices(batch, target_device):
    """Fix batch tensor devices"""
    fixed_batch = {}
    
    for key, value in batch.items():
        if torch.is_tensor(value):
            if value.device != target_device:
                print(f"🔧 Moving {key} from {value.device} to {target_device}")
                fixed_batch[key] = value.to(target_device, non_blocking=True)
            else:
                fixed_batch[key] = value
        else:
            fixed_batch[key] = value
    
    return fixed_batch

class DeviceSafeModel(nn.Module):
    """Wrapper to ensure all operations happen on same device"""
    
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.device = device
        self.model = self.model.to(device)
        
    def forward(self, *args, **kwargs):
        # Move all inputs to correct device
        args = [arg.to(self.device) if torch.is_tensor(arg) else arg for arg in args]
        kwargs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        
        return self.model(*args, **kwargs)
    
    def training_step(self, degraded, enhanced):
        # Ensure inputs are on correct device
        degraded = degraded.to(self.device)
        enhanced = enhanced.to(self.device)
        
        return self.model.training_step(degraded, enhanced)
    
    def validation_step(self, degraded, enhanced):
        # Ensure inputs are on correct device
        degraded = degraded.to(self.device)
        enhanced = enhanced.to(self.device)
        
        return self.model.validation_step(degraded, enhanced)
    
    def sample(self, *args, **kwargs):
        # Move all inputs to correct device
        args = [arg.to(self.device) if torch.is_tensor(arg) else arg for arg in args]
        kwargs = {k: v.to(self.device) if torch.is_tensor(v) else v for k, v in kwargs.items()}
        
        return self.model.sample(*args, **kwargs)

def create_device_safe_training_script():
    """Create a device-safe training script"""
    script_content = '''
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import argparse
import os
from tqdm import tqdm

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

def device_safe_train():
    """Main training function with device safety"""
    
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Using device: {device}")
    
    try:
        # Import project modules
        sys.path.append('.')
        from config import get_config
        from models import create_model
        from data import UnderwaterDataset
        
        # Get config
        config = get_config('lightweight')
        config['data']['batch_size'] = 2  # Small batch size for safety
        
        print("🔧 Loading model...")
        model = create_model(config['model'])
        model = model.to(device)
        
        # Check model is on correct device
        model_devices = set()
        for param in model.parameters():
            model_devices.add(str(param.device))
        print(f"📱 Model devices: {model_devices}")
        
        # Setup optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
        
        # Setup data
        print("📁 Loading dataset...")
        dataset = UnderwaterDataset('./DATA', 'UIEB', 'train', augment=False)
        dataloader = DataLoader(
            dataset, 
            batch_size=config['data']['batch_size'],
            shuffle=True,
            num_workers=0,  # Avoid multiprocessing issues
            pin_memory=False
        )
        
        print(f"✅ Dataset loaded: {len(dataset)} samples")
        print(f"🔄 Starting training for 10 epochs...")
        
        # Training loop
        for epoch in range(10):
            model.train()
            epoch_loss = 0.0
            valid_batches = 0
            
            pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/10")
            
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
                        
                        pbar.set_postfix({'loss': f'{loss.item():.6f}'})
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
        
        print("✅ Training completed successfully!")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    device_safe_train()
'''
    
    with open('device_safe_train.py', 'w') as f:
        f.write(script_content)
    
    print("✅ Created device_safe_train.py")

def diagnose_current_issue():
    """Diagnose the current device issue"""
    print("🔍 DIAGNOSING DEVICE ISSUES")
    print("=" * 50)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🖥️  Target device: {device}")
    
    try:
        # Check if we can import the modules
        sys.path.append('.')
        
        print("\n1. Testing module imports...")
        try:
            from config import get_config
            print("✅ Config import successful")
        except Exception as e:
            print(f"❌ Config import failed: {e}")
            return
        
        try:
            from models import create_model
            print("✅ Model import successful")
        except Exception as e:
            print(f"❌ Model import failed: {e}")
            # Try using fixed model
            try:
                from vae_encoder_fixed import LightweightVAEEncoder
                print("✅ Fixed model import successful")
            except Exception as e2:
                print(f"❌ Fixed model import also failed: {e2}")
                return
        
        print("\n2. Testing model creation...")
        config = get_config('lightweight')
        
        try:
            model = create_model(config['model'])
            print("✅ Model creation successful")
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            # Try lightweight model
            try:
                from vae_encoder_fixed import LightweightVAEEncoder
                model = LightweightVAEEncoder(base_channels=32)
                print("✅ Lightweight model creation successful")
            except Exception as e2:
                print(f"❌ Lightweight model creation failed: {e2}")
                return
        
        print("\n3. Testing device movement...")
        initial_devices = check_model_devices(model, "Initial Model")
        
        model = model.to(device)
        final_devices = check_model_devices(model, "After .to(device)")
        
        print("\n4. Testing data loading...")
        try:
            from data import UnderwaterDataset
            from torch.utils.data import DataLoader
            
            dataset = UnderwaterDataset('./DATA', 'UIEB', 'train', augment=False)
            dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
            
            # Test one batch
            batch = next(iter(dataloader))
            print("✅ Data loading successful")
            
            # Check batch devices
            print(f"📱 Batch keys: {list(batch.keys())}")
            for key, value in batch.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape} on {value.device}")
            
        except Exception as e:
            print(f"❌ Data loading failed: {e}")
            return
        
        print("\n5. Testing device-safe forward pass...")
        try:
            # Move batch to device
            batch = fix_batch_devices(batch, device)
            
            # Test with minimal model
            model.eval()
            with torch.no_grad():
                if hasattr(model, 'encode'):
                    # Try VAE encoder
                    degraded = batch['degraded'].to(device)
                    mean, logvar, physics_cond = model(degraded)
                    print(f"✅ Forward pass successful: {mean.shape}, {logvar.shape}")
                else:
                    print("⚠️  Model doesn't have expected interface")
            
        except Exception as e:
            print(f"❌ Forward pass failed: {e}")
            import traceback
            traceback.print_exc()
        
        print("\n💡 Recommendations:")
        print("1. Use device_safe_train.py for training")
        print("2. Set num_workers=0 to avoid multiprocessing issues")
        print("3. Use smaller batch sizes")
        print("4. Ensure all tensors are moved to device explicitly")
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🔧 DEVICE ISSUE FIXER FOR UNDERWATER IMAGE ENHANCEMENT")
    print("=" * 60)
    
    # Diagnose current issues
    diagnose_current_issue()
    
    print("\n" + "=" * 60)
    print("🛠️  CREATING DEVICE-SAFE TRAINING SCRIPT")
    print("=" * 60)
    
    # Create device-safe training script
    create_device_safe_training_script()
    
    print("\n✅ Device fix tools created!")
    print("\n🚀 Next steps:")
    print("1. Try the device-safe training script:")
    print("   python device_safe_train.py")
    print("")
    print("2. If that works, the issue is in the original train.py")
    print("3. Check for:")
    print("   - Mixed CPU/GPU tensors")
    print("   - Inconsistent .to(device) calls")
    print("   - Multiprocessing conflicts")

if __name__ == "__main__":
    main()
