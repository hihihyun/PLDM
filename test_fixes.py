"""
Test the fixes for NaN and dimension issues
"""

import torch
import sys
sys.path.append('.')

def test_basic_forward():
    """Test basic forward pass with fixed model"""
    print("🧪 Testing fixes...")
    
    from models import create_model
    from config import get_config
    
    # Use lightweight config to reduce memory
    config = get_config('lightweight')
    config['model']['base_channels'] = 32  # Very small for testing
    
    print(f"Model config: {config['model']}")
    
    # Create model
    model = create_model(config['model'])
    model.eval()
    
    # Create small test data
    batch_size = 1
    img_size = 64  # Small for quick testing
    
    degraded = torch.randn(batch_size, 3, img_size, img_size) * 0.5
    enhanced = torch.randn(batch_size, 3, img_size, img_size) * 0.5
    
    # Normalize to [-1, 1] range
    degraded = torch.clamp(degraded * 2.0 - 1.0, -1, 1)
    enhanced = torch.clamp(enhanced * 2.0 - 1.0, -1, 1)
    
    print(f"Input shapes: {degraded.shape}, {enhanced.shape}")
    print(f"Input ranges: [{degraded.min():.3f}, {degraded.max():.3f}]")
    
    try:
        # Test full training step
        print("\n1. Testing training step...")
        model.train()
        loss, loss_dict = model.training_step(degraded, enhanced)
        
        print(f"✅ Training step successful!")
        print(f"Loss: {loss.item():.6f}")
        for key, value in loss_dict.items():
            print(f"  {key}: {value:.6f}")
        
        # Check for NaN
        if torch.isnan(loss) or loss.item() == 0.0:
            print("❌ Still getting NaN or zero loss!")
            return False
        else:
            print("✅ Loss is valid and non-zero!")
        
        # Test backward pass
        print("\n2. Testing backward pass...")
        loss.backward()
        
        # Check gradients
        has_grad = False
        nan_grad = False
        for name, param in model.named_parameters():
            if param.grad is not None:
                has_grad = True
                if torch.isnan(param.grad).any():
                    print(f"❌ NaN gradient in {name}")
                    nan_grad = True
                    break
        
        if not has_grad:
            print("❌ No gradients found!")
            return False
        elif nan_grad:
            print("❌ NaN gradients detected!")
            return False
        else:
            print("✅ Gradients are valid!")
        
        # Test sampling
        print("\n3. Testing sampling...")
        model.eval()
        with torch.no_grad():
            sampled = model.sample(degraded, num_steps=5)  # Very few steps for quick test
            
        print(f"✅ Sampling successful!")
        print(f"Sample shape: {sampled.shape}")
        print(f"Sample range: [{sampled.min():.3f}, {sampled.max():.3f}]")
        
        if torch.isnan(sampled).any():
            print("❌ NaN in sampled output!")
            return False
        else:
            print("✅ Sample output is valid!")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_training_step():
    """Test with real data configuration"""
    print("\n🚀 Testing with real training configuration...")
    
    try:
        from models import create_model
        from config import get_config
        
        # Use the actual UIEB config but with smaller model
        config = get_config('uieb')
        config['model']['base_channels'] = 32  # Smaller for testing
        
        model = create_model(config['model'])
        model.train()
        
        # Create realistic test data
        batch_size = 2
        degraded = torch.randn(batch_size, 3, 256, 256) * 0.3 + 0.5  # [0.2, 0.8] range
        enhanced = torch.randn(batch_size, 3, 256, 256) * 0.3 + 0.5
        
        # Normalize like in real training
        degraded = degraded * 2.0 - 1.0
        enhanced = enhanced * 2.0 - 1.0
        
        # Test training step
        loss, loss_dict = model.training_step(degraded, enhanced)
        
        print(f"✅ Real config test successful!")
        print(f"Loss: {loss.item():.6f}")
        
        if loss.item() > 0:
            print("✅ Non-zero loss achieved!")
            return True
        else:
            print("❌ Still getting zero loss")
            return False
            
    except Exception as e:
        print(f"❌ Real config test failed: {e}")
        return False

def main():
    print("🔧 TESTING FIXES FOR NaN AND DIMENSION ISSUES")
    print("=" * 60)
    
    # Test 1: Basic forward pass
    basic_ok = test_basic_forward()
    
    # Test 2: Real training configuration
    real_ok = test_real_training_step()
    
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Basic forward pass: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    print(f"Real config test: {'✅ PASS' if real_ok else '❌ FAIL'}")
    
    if basic_ok and real_ok:
        print("\n🎉 All tests passed! Fixes are working!")
        print("You can now try training again:")
        print("python train.py --config configs/uieb.yaml --exp_name fixed_experiment")
    else:
        print("\n❌ Some tests failed. Further debugging needed.")
        
    return basic_ok and real_ok

if __name__ == "__main__":
    main()
