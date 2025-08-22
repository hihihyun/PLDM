"""
Debug tool to find exactly where device mismatch occurs
"""

import torch
import torch.nn as nn
import sys
import os
from contextlib import contextmanager

class DeviceTracker:
    """Track tensor devices during operations"""
    def __init__(self):
        self.operations = []
        
    def track_tensors(self, name, tensors):
        """Track tensor devices"""
        info = {'name': name, 'tensors': {}}
        
        if isinstance(tensors, dict):
            for key, value in tensors.items():
                if torch.is_tensor(value):
                    info['tensors'][key] = {
                        'device': str(value.device),
                        'shape': tuple(value.shape),
                        'dtype': str(value.dtype)
                    }
        elif torch.is_tensor(tensors):
            info['tensors']['main'] = {
                'device': str(tensors.device),
                'shape': tuple(tensors.shape),
                'dtype': str(tensors.dtype)
            }
        elif isinstance(tensors, (list, tuple)):
            for i, tensor in enumerate(tensors):
                if torch.is_tensor(tensor):
                    info['tensors'][f'item_{i}'] = {
                        'device': str(tensor.device),
                        'shape': tuple(tensor.shape),
                        'dtype': str(tensor.dtype)
                    }
        
        self.operations.append(info)
        return info
    
    def print_summary(self):
        """Print device tracking summary"""
        print("\n🔍 DEVICE TRACKING SUMMARY")
        print("=" * 50)
        
        for i, op in enumerate(self.operations):
            print(f"\n{i+1}. {op['name']}:")
            for name, tensor_info in op['tensors'].items():
                print(f"   {name}: {tensor_info['shape']} on {tensor_info['device']}")
            
            # Check for mixed devices
            devices = set(info['device'] for info in op['tensors'].values())
            if len(devices) > 1:
                print(f"   ⚠️  MIXED DEVICES: {devices}")

# Global tracker
tracker = DeviceTracker()

@contextmanager
def catch_device_errors():
    """Context manager to catch and analyze device errors"""
    try:
        yield
    except RuntimeError as e:
        if "Expected all tensors to be on the same device" in str(e):
            print(f"\n❌ DEVICE MISMATCH DETECTED!")
            print(f"Error: {e}")
            print("\n📊 Recent operations:")
            tracker.print_summary()
            raise e
        else:
            raise e

def debug_model_creation():
    """Debug model creation process"""
    print("🔧 DEBUGGING MODEL CREATION")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Target device: {device}")
    
    try:
        # Try importing original modules
        sys.path.append('.')
        
        # Test 1: Basic imports
        print("\n1. Testing imports...")
        try:
            from config_fixed import get_config
            config = get_config('lightweight')
            print("✅ Config loaded")
            tracker.track_tensors("config", {})
        except Exception as e:
            print(f"❌ Config failed: {e}")
            return
        
        # Test 2: Model creation  
        print("\n2. Testing model creation...")
        try:
            from models import create_model
            model = create_model(config['model'])
            print("✅ Model created")
            
            # Check initial model device
            model_devices = set()
            for name, param in model.named_parameters():
                model_devices.add(str(param.device))
            print(f"Initial model devices: {model_devices}")
            
        except Exception as e:
            print(f"❌ Model creation failed: {e}")
            # Try lightweight alternative
            try:
                from vae_encoder_fixed import LightweightVAEEncoder
                model = LightweightVAEEncoder(base_channels=32)
                print("✅ Lightweight model created")
            except Exception as e2:
                print(f"❌ Lightweight model also failed: {e2}")
                return
        
        # Test 3: Move to device
        print("\n3. Testing device movement...")
        model = model.to(device)
        
        final_devices = set()
        for name, param in model.named_parameters():
            final_devices.add(str(param.device))
        print(f"Final model devices: {final_devices}")
        
        # Test 4: Create test tensors
        print("\n4. Testing tensor creation...")
        test_input = torch.randn(2, 3, 64, 64).to(device)
        test_target = torch.randn(2, 3, 64, 64).to(device)
        
        tracker.track_tensors("test_tensors", {
            'input': test_input,
            'target': test_target
        })
        
        # Test 5: Forward pass
        print("\n5. Testing forward pass...")
        model.eval()
        
        with catch_device_errors():
            if hasattr(model, 'training_step'):
                # Test training step
                loss, loss_dict = model.training_step(test_input, test_target)
                tracker.track_tensors("training_step_output", {'loss': loss})
                print(f"✅ Training step successful: {loss.item():.6f}")
            else:
                # Test forward pass
                output = model(test_input)
                tracker.track_tensors("forward_output", {'output': output})
                print(f"✅ Forward pass successful: {output.shape}")
        
        print("\n✅ Model debugging completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Model debugging failed: {e}")
        import traceback
        traceback.print_exc()
        tracker.print_summary()

def debug_data_loading():
    """Debug data loading process"""
    print("\n📁 DEBUGGING DATA LOADING")
    print("=" * 40)
    
    try:
        from data import UnderwaterDataset
        from torch.utils.data import DataLoader
        
        # Test dataset creation
        dataset = UnderwaterDataset('./DATA', 'UIEB', 'train', augment=False)
        print(f"✅ Dataset created: {len(dataset)} samples")
        
        # Test dataloader
        dataloader = DataLoader(dataset, batch_size=1, num_workers=0)
        print("✅ DataLoader created")
        
        # Test batch loading
        batch = next(iter(dataloader))
        print("✅ Batch loaded")
        
        # Analyze batch
        tracker.track_tensors("data_batch", batch)
        
        print("Data loading successful!")
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        import traceback
        traceback.print_exc()

def debug_training_step():
    """Debug specific training step"""
    print("\n🏃 DEBUGGING TRAINING STEP")
    print("=" * 40)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    try:
        # Create simple test model
        class DebugModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv = nn.Conv2d(3, 3, 3, padding=1)
                
            def forward(self, x):
                return self.conv(x)
            
            def training_step(self, degraded, enhanced):
                pred = self.forward(degraded)
                loss = nn.functional.mse_loss(pred, enhanced)
                return loss, {'mse': loss.item()}
        
        model = DebugModel().to(device)
        print(f"✅ Debug model created on {device}")
        
        # Create test data
        degraded = torch.randn(2, 3, 64, 64).to(device)
        enhanced = torch.randn(2, 3, 64, 64).to(device)
        
        tracker.track_tensors("input_data", {
            'degraded': degraded,
            'enhanced': enhanced
        })
        
        # Test training step with monitoring
        with catch_device_errors():
            loss, loss_dict = model.training_step(degraded, enhanced)
            tracker.track_tensors("training_output", {'loss': loss})
            print(f"✅ Training step successful: {loss.item():.6f}")
        
    except Exception as e:
        print(f"❌ Training step failed: {e}")
        import traceback
        traceback.print_exc()

def main():
    print("🐛 COMPREHENSIVE DEVICE DEBUGGING TOOL")
    print("=" * 60)
    
    # Test 1: Model creation
    debug_model_creation()
    
    # Test 2: Data loading
    debug_data_loading()
    
    # Test 3: Training step
    debug_training_step()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📊 FINAL DEBUGGING SUMMARY")
    print("=" * 60)
    tracker.print_summary()
    
    print("\n💡 RECOMMENDATIONS:")
    print("1. If model creation fails → Use ultra_safe_train.py")
    print("2. If data loading fails → Check DATA folder structure")
    print("3. If training step fails → Check for mixed device tensors")
    print("4. If all fails → Run with --device cpu")

if __name__ == "__main__":
    main()
