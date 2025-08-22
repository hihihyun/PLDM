"""
Improved test script for debugging NaN and dimension issues
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os
sys.path.append('.')

# Import fixed modules - try multiple locations
FixedVAEEncoder = None
LightweightVAEEncoder = None
FixedResBlock = None
FixedAttentionBlock = None

# Try importing from different locations
import_success = False

# Try 1: From root directory
try:
    from vae_encoder_fixed import FixedVAEEncoder, LightweightVAEEncoder, reparameterize
    from basic_modules_fixed import FixedResBlock, FixedAttentionBlock
    print("✅ Fixed modules imported from root directory")
    import_success = True
except ImportError:
    print("⚠️  Fixed modules not found in root directory")

# Try 2: From models directory
if not import_success:
    try:
        from models.vae_encoder_fixed import FixedVAEEncoder, LightweightVAEEncoder, reparameterize
        from models.basic_modules_fixed import FixedResBlock, FixedAttentionBlock
        print("✅ Fixed modules imported from models directory")
        import_success = True
    except ImportError:
        print("⚠️  Fixed modules not found in models directory")

# Try 3: Original modules as fallback
if not import_success:
    try:
        from models.vae_encoder import EnhancedVAEEncoder
        from models.basic_modules import ResBlock, AttentionBlock
        print("⚠️  Using original modules as fallback")
        # Create aliases
        FixedVAEEncoder = EnhancedVAEEncoder
        FixedResBlock = ResBlock
        FixedAttentionBlock = AttentionBlock
    except ImportError:
        print("❌ No modules found! Will use minimal test model only")


class MinimalTestModel(nn.Module):
    """Minimal model for basic testing"""
    def __init__(self, img_size=64, in_channels=3, latent_channels=4, base_channels=32):
        super().__init__()
        
        # Simple encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(base_channels, base_channels*2, 3, stride=2, padding=1),  # /2
            nn.ReLU(),
            nn.Conv2d(base_channels*2, base_channels*4, 3, stride=2, padding=1),  # /4
            nn.ReLU(),
            nn.Conv2d(base_channels*4, base_channels*8, 3, stride=2, padding=1),  # /8
            nn.ReLU(),
        )
        
        # Output layers
        self.conv_mean = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        self.conv_logvar = nn.Conv2d(base_channels*8, latent_channels, 3, padding=1)
        
        # Simple decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, base_channels*8, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*8, base_channels*4, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*4, base_channels*2, 3, padding=1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(base_channels*2, in_channels, 3, padding=1),
            nn.Tanh(),
        )
        
    def encode(self, x):
        h = self.encoder(x)
        mean = self.conv_mean(h)
        logvar = self.conv_logvar(h)
        return mean, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def reparameterize(self, mean, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std
    
    def forward(self, x):
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar)
        recon = self.decode(z)
        return recon, mean, logvar
    
    def training_step(self, degraded, enhanced):
        """Simple training step for testing"""
        recon, mean, logvar = self.forward(degraded)
        
        # Simple losses
        recon_loss = F.mse_loss(recon, enhanced)
        kl_loss = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp()) / degraded.numel()
        
        total_loss = recon_loss + 0.001 * kl_loss
        
        return total_loss, {
            'total': total_loss.item(),
            'reconstruction': recon_loss.item(),
            'kl': kl_loss.item()
        }


def test_minimal_model():
    """Test minimal model first"""
    print("\n🧪 Testing minimal model...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {device}")
        
        # Create minimal model
        model = MinimalTestModel(img_size=64, base_channels=32).to(device)
        model.train()
        
        # Create test data
        batch_size = 2
        img_size = 64
        
        degraded = torch.randn(batch_size, 3, img_size, img_size).to(device) * 0.5
        enhanced = torch.randn(batch_size, 3, img_size, img_size).to(device) * 0.5
        
        # Normalize to [-1, 1]
        degraded = torch.tanh(degraded)
        enhanced = torch.tanh(enhanced)
        
        print(f"Input shapes: {degraded.shape}, {enhanced.shape}")
        print(f"Input ranges: [{degraded.min():.3f}, {degraded.max():.3f}]")
        
        # Test forward pass
        recon, mean, logvar = model(degraded)
        print(f"✅ Forward pass successful!")
        print(f"  Reconstruction: {recon.shape}")
        print(f"  Mean: {mean.shape}")
        print(f"  Logvar: {logvar.shape}")
        
        # Test training step
        loss, loss_dict = model.training_step(degraded, enhanced)
        print(f"✅ Training step successful!")
        print(f"  Loss: {loss.item():.6f}")
        
        # Test backward pass
        loss.backward()
        print(f"✅ Backward pass successful!")
        
        # Check gradients
        has_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                if torch.isnan(param.grad).any():
                    print(f"❌ NaN gradient in {name}")
                    return False
        
        if has_grad:
            print("✅ Valid gradients found!")
        else:
            print("❌ No gradients found!")
            return False
            
        return True
        
    except Exception as e:
        print(f"❌ Minimal model test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_fixed_vae_encoder():
    """Test fixed VAE encoder"""
    print("\n🔧 Testing fixed VAE encoder...")
    
    # Check if fixed modules are available
    if LightweightVAEEncoder is None or FixedVAEEncoder is None:
        print("⚠️  Fixed VAE encoders not available, skipping test")
        return False
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test lightweight version first
        print("Testing LightweightVAEEncoder...")
        encoder = LightweightVAEEncoder(base_channels=32).to(device)
        encoder.eval()
        
        x = torch.randn(2, 3, 64, 64).to(device) * 0.5
        x = torch.clamp(x, -1, 1)
        
        mean, logvar, physics_cond = encoder(x)
        print(f"✅ Lightweight encoder - Mean: {mean.shape}, Logvar: {logvar.shape}")
        
        # Test reparameterization
        z = reparameterize(mean, logvar)
        print(f"✅ Reparameterization - Latent: {z.shape}")
        
        # Test full encoder
        print("\nTesting FixedVAEEncoder...")
        encoder = FixedVAEEncoder(base_channels=32, physics_dim=32).to(device)
        encoder.eval()
        
        mean, logvar, physics_cond = encoder(x)
        print(f"✅ Fixed encoder - Mean: {mean.shape}, Logvar: {logvar.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Fixed VAE encoder test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_dimension_compatibility():
    """Test dimension compatibility between modules"""
    print("\n📐 Testing dimension compatibility...")
    
    # Check if fixed modules are available
    if FixedResBlock is None or FixedAttentionBlock is None:
        print("⚠️  Fixed modules not available, skipping test")
        return False
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test ResBlock with different dimensions
        print("Testing ResBlock dimension transitions...")
        
        # Test 1: Same dimensions
        block1 = FixedResBlock(64, 64).to(device)
        x1 = torch.randn(2, 64, 32, 32).to(device)
        out1 = block1(x1)
        print(f"✅ ResBlock same dim - Input: {x1.shape}, Output: {out1.shape}")
        
        # Test 2: Different dimensions
        block2 = FixedResBlock(64, 128).to(device)
        out2 = block2(x1)
        print(f"✅ ResBlock diff dim - Input: {x1.shape}, Output: {out2.shape}")
        
        # Test 3: With downsampling
        block3 = FixedResBlock(64, 128, downsample=True).to(device)
        out3 = block3(x1)
        print(f"✅ ResBlock downsample - Input: {x1.shape}, Output: {out3.shape}")
        
        # Test AttentionBlock
        print("\nTesting AttentionBlock...")
        attn = FixedAttentionBlock(128, num_heads=4).to(device)
        out_attn = attn(out2)
        print(f"✅ AttentionBlock - Input: {out2.shape}, Output: {out_attn.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Dimension compatibility test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_numerical_stability():
    """Test numerical stability with extreme values"""
    print("\n🧮 Testing numerical stability...")
    
    # Check if modules are available
    if LightweightVAEEncoder is None:
        print("⚠️  LightweightVAEEncoder not available, using MinimalTestModel")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            
            # Use minimal model instead
            model = MinimalTestModel(base_channels=32).to(device)
            model.eval()
            
            # Test with extreme values
            extreme_input = torch.randn(2, 3, 64, 64).to(device) * 10  # Large values
            
            with torch.no_grad():
                output, mean, logvar = model(extreme_input)
            
            # Check for NaN
            if torch.isnan(output).any() or torch.isnan(mean).any() or torch.isnan(logvar).any():
                print("❌ NaN detected in outputs")
                return False
            else:
                print("✅ No NaN detected with extreme inputs")
                return True
                
        except Exception as e:
            print(f"❌ Numerical stability test failed: {e}")
            return False
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Test with extreme values
        extreme_input = torch.randn(2, 3, 64, 64).to(device) * 100  # Very large values
        
        encoder = LightweightVAEEncoder(base_channels=32).to(device)
        encoder.eval()
        
        mean, logvar, physics_cond = encoder(extreme_input)
        
        # Check for NaN or extreme values
        if torch.isnan(mean).any() or torch.isnan(logvar).any():
            print("❌ NaN detected in outputs")
            return False
        
        if torch.abs(mean).max() > 100 or torch.abs(logvar).max() > 100:
            print("⚠️  Very large output values detected")
            print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
            print(f"  Logvar range: [{logvar.min():.3f}, {logvar.max():.3f}]")
        else:
            print("✅ Output values are reasonable")
            print(f"  Mean range: [{mean.min():.3f}, {mean.max():.3f}]")
            print(f"  Logvar range: [{logvar.min():.3f}, {logvar.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Numerical stability test failed: {e}")
        return False


def test_gradient_flow():
    """Test gradient flow through the model"""
    print("\n⬇️  Testing gradient flow...")
    
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        model = MinimalTestModel(base_channels=32).to(device)
        model.train()
        
        # Create input that requires gradient - make it a parameter for proper gradient tracking
        x = nn.Parameter(torch.randn(2, 3, 64, 64).to(device))
        target = torch.randn(2, 3, 64, 64).to(device)
        
        # Forward pass
        output, mean, logvar = model(x)
        loss = F.mse_loss(output, target)
        
        # Backward pass
        loss.backward()
        
        # Check input gradients
        if x.grad is not None and not torch.isnan(x.grad).any():
            print("✅ Input gradients are valid")
            grad_norm = x.grad.norm().item()
            print(f"  Input gradient norm: {grad_norm:.6f}")
        else:
            print("❌ Input gradients are invalid or NaN")
            return False
        
        # Check parameter gradients
        grad_count = 0
        nan_count = 0
        total_grad_norm = 0
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_count += 1
                if torch.isnan(param.grad).any():
                    nan_count += 1
                    print(f"❌ NaN gradient in {name}")
                else:
                    grad_norm = param.grad.norm().item()
                    total_grad_norm += grad_norm
        
        print(f"✅ {grad_count} parameters have gradients")
        print(f"  Average gradient norm: {total_grad_norm/max(grad_count, 1):.6f}")
        
        if nan_count > 0:
            print(f"❌ {nan_count} parameters have NaN gradients")
            return False
        else:
            print("✅ All gradients are valid")
        
        return True
        
    except Exception as e:
        print(f"❌ Gradient flow test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("🔧 IMPROVED TEST SUITE FOR UNDERWATER IMAGE ENHANCEMENT")
    print("=" * 70)
    
    # Run tests in order of complexity
    tests = [
        ("Minimal Model", test_minimal_model),
        ("Fixed VAE Encoder", test_fixed_vae_encoder),
        ("Dimension Compatibility", test_dimension_compatibility),
        ("Numerical Stability", test_numerical_stability),
        ("Gradient Flow", test_gradient_flow),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\n{'='*10} {test_name} {'='*10}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<25}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! The fixes are working!")
        print("\n💡 Next steps:")
        print("  1. Replace old modules with fixed versions")
        print("  2. Update your main training script")
        print("  3. Try training with: python train.py --config configs/lightweight.yaml")
    elif passed >= total // 2:
        print("\n⚠️  Most tests passed, but some issues remain")
        print("💡 Try using the MinimalTestModel or LightweightVAEEncoder for now")
    else:
        print("\n❌ Multiple issues detected. Check the error messages above.")
        print("💡 Consider starting with the MinimalTestModel")
    
    return passed == total


if __name__ == "__main__":
    main()
