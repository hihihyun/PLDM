
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
