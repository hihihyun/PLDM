"""
Debug training issues - diagnose why loss is 0
"""

import torch
import sys
sys.path.append('.')

from models import create_model
from data import create_dataloaders
from config import get_config

def debug_model_forward():
    """Debug model forward pass"""
    print("🔍 Debugging model forward pass...")
    
    config = get_config('uieb')
    
    # Create model
    model = create_model(config['model'])
    model.eval()
    
    # Create dummy data
    batch_size = 2
    degraded = torch.randn(batch_size, 3, 256, 256) * 0.5 + 0.5  # [0, 1] range
    enhanced = torch.randn(batch_size, 3, 256, 256) * 0.5 + 0.5  # [0, 1] range
    
    # Normalize to [-1, 1] like in training
    degraded = degraded * 2.0 - 1.0
    enhanced = enhanced * 2.0 - 1.0
    
    print(f"Input ranges - Degraded: [{degraded.min():.3f}, {degraded.max():.3f}]")
    print(f"Input ranges - Enhanced: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
    
    # Test preprocessing
    print("\n1. Testing preprocessing...")
    try:
        preprocessed = model.preprocessor.preprocess_batch(degraded)
        print(f"✅ Preprocessing successful: {preprocessed.shape}")
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        return False
    
    # Test encoding
    print("\n2. Testing VAE encoding...")
    try:
        z, mean, logvar, physics_cond = model.encode_to_latent(enhanced, preprocessed)
        print(f"✅ Encoding successful:")
        print(f"  Latent z: {z.shape}, range: [{z.min():.3f}, {z.max():.3f}]")
        print(f"  Mean: {mean.shape}, range: [{mean.min():.3f}, {mean.max():.3f}]")
        print(f"  Logvar: {logvar.shape}, range: [{logvar.min():.3f}, {logvar.max():.3f}]")
        print(f"  Physics: {physics_cond.shape}")
    except Exception as e:
        print(f"❌ Encoding failed: {e}")
        return False
    
    # Test forward diffusion
    print("\n3. Testing forward diffusion...")
    try:
        t = torch.randint(0, 1000, (batch_size,))
        z_noisy, noise = model.forward_diffusion(z, t)
        print(f"✅ Forward diffusion successful:")
        print(f"  Noisy z: {z_noisy.shape}, range: [{z_noisy.min():.3f}, {z_noisy.max():.3f}]")
        print(f"  Noise: {noise.shape}, range: [{noise.min():.3f}, {noise.max():.3f}]")
    except Exception as e:
        print(f"❌ Forward diffusion failed: {e}")
        return False
    
    # Test UNet
    print("\n4. Testing diffusion UNet...")
    try:
        predicted_noise = model.diffusion_unet(z_noisy, t, physics_cond)
        print(f"✅ UNet successful:")
        print(f"  Predicted noise: {predicted_noise.shape}, range: [{predicted_noise.min():.3f}, {predicted_noise.max():.3f}]")
    except Exception as e:
        print(f"❌ UNet failed: {e}")
        return False
    
    # Test loss computation
    print("\n5. Testing loss computation...")
    try:
        diffusion_loss = torch.nn.functional.mse_loss(predicted_noise, noise)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / batch_size
        
        print(f"✅ Loss computation successful:")
        print(f"  Diffusion loss: {diffusion_loss.item():.6f}")
        print(f"  KL loss: {kl_loss.item():.6f}")
        
        if diffusion_loss.item() == 0.0:
            print("⚠️  WARNING: Diffusion loss is exactly 0!")
            print("  This suggests predicted_noise == noise exactly")
            print(f"  Difference: {(predicted_noise - noise).abs().max().item():.10f}")
        
        if kl_loss.item() == 0.0:
            print("⚠️  WARNING: KL loss is exactly 0!")
            print("  This suggests mean=0, logvar=0 exactly")
    
    except Exception as e:
        print(f"❌ Loss computation failed: {e}")
        return False
    
    # Test full training step
    print("\n6. Testing full training step...")
    try:
        model.train()
        loss, loss_dict = model.training_step(degraded, enhanced)
        print(f"✅ Training step successful:")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.6f}")
            
        if loss_dict['total'] == 0.0:
            print("❌ PROBLEM: Total loss is 0!")
        else:
            print("✅ Total loss is non-zero - good!")
            
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        return False
    
    return True

def debug_real_data():
    """Debug with real data"""
    print("\n🔍 Debugging with real dataset...")
    
    config = get_config('uieb')
    
    try:
        train_loader, val_loader = create_dataloaders(config['data'])
        print(f"✅ DataLoaders created")
        
        # Get one batch
        for batch in train_loader:
            print(f"Batch keys: {batch.keys()}")
            
            degraded = batch['degraded']
            enhanced = batch.get('enhanced', None)
            
            print(f"Degraded shape: {degraded.shape}")
            print(f"Degraded range: [{degraded.min():.3f}, {degraded.max():.3f}]")
            
            if enhanced is not None:
                print(f"Enhanced shape: {enhanced.shape}")
                print(f"Enhanced range: [{enhanced.min():.3f}, {enhanced.max():.3f}]")
                
                # Test if they're identical (which would cause 0 loss)
                if torch.allclose(degraded, enhanced, atol=1e-6):
                    print("⚠️  WARNING: Degraded and enhanced are identical!")
                else:
                    print("✅ Degraded and enhanced are different")
                    
            else:
                print("❌ No enhanced image in batch")
                
            break
            
    except Exception as e:
        print(f"❌ Real data debug failed: {e}")
        return False
        
    return True

def main():
    print("🐛 TRAINING DEBUG TOOL")
    print("=" * 40)
    
    # Test model forward pass
    model_ok = debug_model_forward()
    
    # Test real data
    data_ok = debug_real_data()
    
    print("\n" + "=" * 40)
    print("📊 DEBUG SUMMARY")
    print("=" * 40)
    print(f"Model forward pass: {'✅ OK' if model_ok else '❌ FAILED'}")
    print(f"Real data loading: {'✅ OK' if data_ok else '❌ FAILED'}")
    
    if model_ok and data_ok:
        print("\n🎉 Basic functionality seems OK!")
        print("💡 If training loss is still 0, try:")
        print("  1. Check if degraded == enhanced (identical images)")
        print("  2. Verify mixed precision isn't causing underflow")
        print("  3. Try with --device cpu")
        print("  4. Use a larger learning rate")
    else:
        print("\n❌ Found issues that need fixing")

if __name__ == "__main__":
    main()
