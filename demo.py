"""
Demo script for Underwater Image Enhancement
Quick demonstration of model capabilities
"""

import torch
import argparse
from pathlib import Path
import time
from PIL import Image
import torchvision.transforms as transforms

# Import our modules
from models import UnderwaterEnhancementModel, create_model
from config import get_config
from utils import save_single_image, print_model_info, calculate_metrics


def demo_model_creation():
    """Demonstrate model creation and architecture"""
    print("🏗️  Creating Underwater Enhancement Model...")
    
    try:
        # Get default configuration - try multiple approaches
        try:
            config = get_config('default')
        except Exception as e:
            print(f"⚠️  Config loading failed: {e}")
            print("🔧 Using fallback default configuration...")
            # Import the function directly to get default config
            from config import get_default_config
            config = get_default_config()
        
        print("✅ Configuration loaded successfully")
        
        # Create model
        model = create_model(config['model'])
        print("✅ Model created successfully")
        
        # Print model information
        print_model_info(model)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 256, 256)
        print(f"\n🧪 Testing forward pass with input shape: {dummy_input.shape}")
        
        try:
            # Test training step
            dummy_target = torch.randn(1, 3, 256, 256)
            loss, loss_dict = model.training_step(dummy_input, dummy_target)
            print(f"✅ Training step successful! Loss: {loss.item():.4f}")
            print(f"Loss components: {loss_dict}")
            
            # Test sampling
            print("🎨 Testing image generation...")
            with torch.no_grad():
                start_time = time.time()
                generated = model.sample(dummy_input, num_steps=10)  # Fast sampling for demo
                generation_time = time.time() - start_time
                
            print(f"✅ Generation successful! Time: {generation_time:.2f}s")
            print(f"Generated image shape: {generated.shape}")
            
        except Exception as e:
            print(f"❌ Error during model testing: {e}")
            print(f"🔍 Error details: {type(e).__name__}")
            return False
        
    except Exception as e:
        print(f"❌ Error during model creation: {e}")
        print(f"🔍 Error details: {type(e).__name__}")
        print("💡 This might be due to missing dependencies or config issues")
        return False
    
    return True


def demo_preprocessing():
    """Demonstrate Water-Net style preprocessing"""
    print("\n🌊 Demonstrating Water-Net Preprocessing...")
    
    from models.water_physics import WaterNetPreprocessor
    
    # Create dummy underwater image
    dummy_image = torch.rand(1, 3, 256, 256)  # Random image
    
    # Apply preprocessing
    preprocessor = WaterNetPreprocessor()
    
    try:
        preprocessed = preprocessor.preprocess_batch(dummy_image)
        print(f"✅ Preprocessing successful!")
        print(f"Original shape: {dummy_image.shape}")
        print(f"Preprocessed shape: {preprocessed.shape}")
        print("Preprocessed features include: White Balance, Gamma Correction, Histogram Equalization")
        
    except Exception as e:
        print(f"❌ Error during preprocessing: {e}")
        return False
    
    return True


def demo_single_image_enhancement(image_path, output_path, model_path=None):
    """Demonstrate single image enhancement"""
    print(f"\n🖼️  Enhancing single image: {image_path}")
    
    # Check if image exists
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        return False
    
    try:
        # Load image
        image = Image.open(image_path).convert('RGB')
        print(f"✅ Image loaded: {image.size}")
        
        # Transform to tensor
        transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])
        
        image_tensor = transform(image).unsqueeze(0)
        
        if model_path and Path(model_path).exists():
            # Use pretrained model
            print(f"📦 Loading pretrained model from {model_path}")
            model = UnderwaterEnhancementModel(model_path)
        else:
            # Create new model for demonstration
            print("🆕 Creating new model for demonstration...")
            config = get_config('lightweight')  # Use lightweight config for faster demo
            base_model = create_model(config['model'])
            model = UnderwaterEnhancementModel()
            model.model = base_model
        
        # Enhance image
        print("🎨 Enhancing image...")
        start_time = time.time()
        
        with torch.no_grad():
            enhanced = model.enhance_image(image_tensor, num_steps=20)  # Fast for demo
        
        enhancement_time = time.time() - start_time
        
        # Save result
        save_single_image(enhanced, output_path, normalize=False)
        
        print(f"✅ Enhancement completed in {enhancement_time:.2f}s")
        print(f"💾 Enhanced image saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during image enhancement: {e}")
        return False


def demo_dataset_loading():
    """Demonstrate dataset loading"""
    print("\n📁 Demonstrating Dataset Loading...")
    
    from data import UnderwaterDataset
    
    # Try to create dataset (will fail if data not available, but shows the API)
    try:
        dataset_config = {
            'data_root': './DATA',
            'dataset_type': 'UIEB',
            'img_size': 256,
            'augment': False,
            'preprocessing_type': 'waternet'
        }
        
        # This will fail if DATA directory doesn't exist, but that's OK for demo
        dataset = UnderwaterDataset(
            root_dir=dataset_config['data_root'],
            dataset_type=dataset_config['dataset_type'],
            split='train',
            img_size=dataset_config['img_size'],
            augment=dataset_config['augment'],
            preprocessing_type=dataset_config['preprocessing_type']
        )
        
        print(f"✅ Dataset created successfully!")
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample keys: {sample.keys()}")
            for key, value in sample.items():
                if torch.is_tensor(value):
                    print(f"  {key}: {value.shape}")
                else:
                    print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"⚠️  Dataset loading failed (expected if DATA directory not available): {e}")
        print("💡 To test with real data, organize your dataset as described in README.md")
        return False


def demo_configurations():
    """Demonstrate different configuration options"""
    print("\n⚙️  Demonstrating Configuration Options...")
    
    config_types = ['default', 'lightweight', 'high_quality']
    
    for config_type in config_types:
        try:
            config = get_config(config_name=config_type)
            print(f"\n📋 {config_type.upper()} Configuration:")
            print(f"  Image size: {config['model']['img_size']}")
            print(f"  Base channels: {config['model']['base_channels']}")
            print(f"  Time steps: {config['model']['time_steps']}")
            print(f"  Batch size: {config['data']['batch_size']}")
            
        except Exception as e:
            print(f"❌ Error loading {config_type} config: {e}")
    
    return True


def main():
    parser = argparse.ArgumentParser(description='Underwater Image Enhancement Demo')
    parser.add_argument('--demo', type=str, choices=['all', 'model', 'preprocess', 'image', 'dataset', 'config'], 
                       default='all', help='Which demo to run')
    parser.add_argument('--image', type=str, help='Image path for single image demo')
    parser.add_argument('--output', type=str, default='demo_enhanced.jpg', help='Output path for enhanced image')
    parser.add_argument('--model_path', type=str, help='Path to pretrained model')
    
    args = parser.parse_args()
    
    print("🌊 UNDERWATER IMAGE ENHANCEMENT DEMO 🌊")
    print("=" * 50)
    
    success_count = 0
    total_demos = 0
    
    if args.demo in ['all', 'model']:
        total_demos += 1
        if demo_model_creation():
            success_count += 1
    
    if args.demo in ['all', 'preprocess']:
        total_demos += 1
        if demo_preprocessing():
            success_count += 1
    
    if args.demo in ['all', 'config']:
        total_demos += 1
        if demo_configurations():
            success_count += 1
    
    if args.demo in ['all', 'dataset']:
        total_demos += 1
        if demo_dataset_loading():
            success_count += 1
    
    if args.demo in ['all', 'image'] or args.image:
        if args.image:
            total_demos += 1
            if demo_single_image_enhancement(args.image, args.output, args.model_path):
                success_count += 1
        elif args.demo == 'image':
            print("\n🖼️  Single Image Enhancement Demo")
            print("⚠️  Please provide --image path to test image enhancement")
            print("Example: python demo.py --demo image --image test.jpg --output enhanced.jpg")
    
    print("\n" + "=" * 50)
    print(f"📊 Demo Summary: {success_count}/{total_demos} demos completed successfully")
    
    if success_count == total_demos and total_demos > 0:
        print("🎉 All demos completed successfully!")
        print("💡 You're ready to start using the underwater image enhancement model!")
    else:
        print("ℹ️  Some demos may have failed due to missing data or dependencies.")
        print("   This is normal for a first run. Check README.md for setup instructions.")
    
    print("\n🚀 Next Steps:")
    print("  1. Prepare your dataset (see README.md)")
    print("  2. Train the model: python train.py --config configs/uieb.yaml")
    print("  3. Test on images: python test.py --mode single --input_image your_image.jpg")
    
    return success_count == total_demos


if __name__ == "__main__":
    main()
