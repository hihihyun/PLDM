"""
Create test dataset for Underwater Image Enhancement
Generates synthetic underwater-style images for testing the pipeline
"""

import os
import argparse
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import random
import torch
import torchvision.transforms as transforms


def create_underwater_effect(image, effect_type='blue_green'):
    """Apply underwater-like effects to a normal image"""
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply different underwater effects
    if effect_type == 'blue_green':
        # Blue-green tint (common in underwater images)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.7)  # Reduce color saturation
        
        # Add blue-green tint
        overlay = Image.new('RGB', image.size, (0, 50, 100))
        image = Image.blend(image, overlay, 0.2)
        
        # Reduce contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.6)
        
    elif effect_type == 'murky':
        # Murky water effect
        enhancer = ImageEnhance.Brightness(image)
        image = enhancer.enhance(0.7)  # Darker
        
        # Add brownish tint
        overlay = Image.new('RGB', image.size, (50, 40, 20))
        image = Image.blend(image, overlay, 0.3)
        
        # Add noise
        noise = np.random.randint(0, 50, (image.height, image.width, 3))
        img_array = np.array(image)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(img_array)
        
    elif effect_type == 'deep_water':
        # Deep water effect (very blue)
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(0.5)
        
        # Strong blue tint
        overlay = Image.new('RGB', image.size, (0, 20, 80))
        image = Image.blend(image, overlay, 0.4)
        
        # Very low contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(0.4)
        
    # Add slight blur (water scattering effect)
    image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
    
    return image


def generate_random_image(size=(256, 256), pattern='natural'):
    """Generate random images with different patterns"""
    
    if pattern == 'natural':
        # Generate natural-looking random image
        # Create multiple frequency components
        base = np.random.rand(*size, 3) * 50
        
        # Add some structure
        x, y = np.meshgrid(np.linspace(0, 10, size[1]), np.linspace(0, 10, size[0]))
        
        # Add sinusoidal patterns
        pattern1 = 50 * np.sin(x) * np.cos(y)
        pattern2 = 30 * np.sin(2*x + 1) * np.sin(3*y + 2)
        
        for i in range(3):
            base[:, :, i] += pattern1 + pattern2
        
        # Add random texture
        base += np.random.rand(*size, 3) * 100
        
        # Normalize to 0-255
        base = np.clip(base, 0, 255).astype(np.uint8)
        
    elif pattern == 'geometric':
        # Generate geometric patterns
        base = np.zeros((*size, 3), dtype=np.uint8)
        
        # Add rectangles
        for _ in range(random.randint(5, 15)):
            x1, y1 = random.randint(0, size[1]//2), random.randint(0, size[0]//2)
            x2, y2 = random.randint(x1, size[1]), random.randint(y1, size[0])
            color = [random.randint(50, 200) for _ in range(3)]
            base[y1:y2, x1:x2] = color
        
        # Add noise
        base = base.astype(np.float32)
        base += np.random.rand(*size, 3) * 50
        base = np.clip(base, 0, 255).astype(np.uint8)
    
    elif pattern == 'coral':
        # Generate coral-like patterns
        base = np.random.rand(*size, 3) * 100 + 100  # Brighter base
        
        # Add coral-like structures
        x, y = np.meshgrid(np.linspace(0, 20, size[1]), np.linspace(0, 20, size[0]))
        
        # Multiple coral branches
        for i in range(5):
            center_x, center_y = random.uniform(5, 15), random.uniform(5, 15)
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            coral = 150 * np.exp(-dist/3) * (1 + 0.5*np.sin(10*dist))
            
            # Add to one channel more (coral colors)
            channel = random.randint(0, 2)
            base[:, :, channel] += coral
        
        base = np.clip(base, 0, 255).astype(np.uint8)
    
    return Image.fromarray(base)


def create_enhanced_version(degraded_image):
    """Create an enhanced version of degraded underwater image"""
    
    enhanced = degraded_image.copy()
    
    # Enhance color saturation
    enhancer = ImageEnhance.Color(enhanced)
    enhanced = enhancer.enhance(1.5)
    
    # Increase contrast
    enhancer = ImageEnhance.Contrast(enhanced)
    enhanced = enhancer.enhance(1.4)
    
    # Adjust brightness
    enhancer = ImageEnhance.Brightness(enhanced)
    enhanced = enhancer.enhance(1.1)
    
    # Reduce blue tint
    img_array = np.array(enhanced).astype(np.float32)
    
    # Color correction: reduce blue, enhance red
    img_array[:, :, 0] = np.clip(img_array[:, :, 0] * 1.2, 0, 255)  # Red
    img_array[:, :, 1] = np.clip(img_array[:, :, 1] * 1.1, 0, 255)  # Green  
    img_array[:, :, 2] = np.clip(img_array[:, :, 2] * 0.8, 0, 255)  # Blue
    
    enhanced = Image.fromarray(img_array.astype(np.uint8))
    
    # Slight sharpening
    enhanced = enhanced.filter(ImageFilter.UnsharpMask(radius=1, percent=50, threshold=3))
    
    return enhanced


def create_test_dataset(data_root, dataset_type='UIEB', num_samples=50):
    """Create a test dataset with synthetic underwater images"""
    
    data_root = Path(data_root)
    
    if dataset_type == 'UIEB':
        # Create UIEB structure
        degraded_dir = data_root / 'UIEB' / 'raw-890'
        enhanced_dir = data_root / 'UIEB' / 'reference-890'
        test_dir = data_root / 'UIEB' / 'challengingset-60'
        
        # Create directories
        for dir_path in [degraded_dir, enhanced_dir, test_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Creating UIEB test dataset with {num_samples} samples...")
        
        # Create training pairs
        for i in range(num_samples):
            # Generate base image
            pattern = random.choice(['natural', 'geometric', 'coral'])
            base_image = generate_random_image(size=(256, 256), pattern=pattern)
            
            # Create degraded version (underwater effect)
            effect = random.choice(['blue_green', 'murky', 'deep_water'])
            degraded = create_underwater_effect(base_image, effect)
            
            # Create enhanced version
            enhanced = create_enhanced_version(degraded)
            
            # Save images
            filename = f"test_{i:04d}.jpg"
            degraded.save(degraded_dir / filename)
            enhanced.save(enhanced_dir / filename)
            
            if i % 10 == 0:
                print(f"✅ Generated {i+1}/{num_samples} training pairs")
        
        # Create test images (smaller set)
        test_samples = min(20, num_samples // 3)
        for i in range(test_samples):
            pattern = random.choice(['natural', 'geometric', 'coral'])
            base_image = generate_random_image(size=(256, 256), pattern=pattern)
            effect = random.choice(['blue_green', 'murky', 'deep_water'])
            degraded = create_underwater_effect(base_image, effect)
            
            filename = f"challenge_{i:04d}.jpg"
            degraded.save(test_dir / filename)
        
        print(f"✅ Created {test_samples} test images")
        
    elif dataset_type == 'LSUI':
        # Create LSUI structure
        input_dir = data_root / 'LSUI' / 'input'
        gt_dir = data_root / 'LSUI' / 'GT'
        
        # Create directories
        for dir_path in [input_dir, gt_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        print(f"📁 Creating LSUI test dataset with {num_samples} samples...")
        
        # Create pairs
        for i in range(num_samples):
            pattern = random.choice(['natural', 'geometric', 'coral'])
            base_image = generate_random_image(size=(256, 256), pattern=pattern)
            
            effect = random.choice(['blue_green', 'murky', 'deep_water'])
            degraded = create_underwater_effect(base_image, effect)
            enhanced = create_enhanced_version(degraded)
            
            filename = f"lsui_{i:04d}.jpg"
            degraded.save(input_dir / filename)
            enhanced.save(gt_dir / filename)
            
            if i % 10 == 0:
                print(f"✅ Generated {i+1}/{num_samples} pairs")
    
    print(f"🎉 Test dataset created successfully!")
    print(f"📍 Location: {data_root.absolute()}")


def create_sample_images_from_web():
    """Download sample images from free sources (placeholder function)"""
    print("🌐 To use real images, you can:")
    print("1. Download from Unsplash: https://unsplash.com/s/photos/underwater")
    print("2. Download from Pexels: https://www.pexels.com/search/underwater/")
    print("3. Use your own underwater images")
    print("")
    print("💡 For now, using synthetic images for testing")


def main():
    parser = argparse.ArgumentParser(description='Create test dataset for underwater image enhancement')
    parser.add_argument('--data_root', type=str, default='./DATA', 
                       help='Root directory to create test dataset')
    parser.add_argument('--dataset_type', type=str, choices=['UIEB', 'LSUI'], 
                       default='UIEB', help='Type of dataset structure to create')
    parser.add_argument('--num_samples', type=int, default=50, 
                       help='Number of sample images to generate')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    print("🎨 UNDERWATER TEST DATASET CREATOR")
    print("=" * 40)
    print(f"Dataset type: {args.dataset_type}")
    print(f"Number of samples: {args.num_samples}")
    print(f"Output location: {Path(args.data_root).absolute()}")
    print("")
    
    # Check if data already exists
    data_root = Path(args.data_root)
    if args.dataset_type == 'UIEB':
        check_dir = data_root / 'UIEB' / 'raw-890'
    else:
        check_dir = data_root / 'LSUI' / 'input'
    
    if check_dir.exists() and len(list(check_dir.glob('*'))) > 0:
        response = input(f"⚠️  Data already exists in {check_dir}. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("❌ Cancelled by user")
            return
    
    # Create test dataset
    try:
        create_test_dataset(args.data_root, args.dataset_type, args.num_samples)
        
        # Verify creation
        print(f"\n🔍 Verifying dataset...")
        from check_dataset import check_directory_structure
        success = check_directory_structure(args.data_root, args.dataset_type)
        
        if success:
            print("✅ Verification passed!")
            print(f"\n🚀 You can now test the pipeline:")
            print(f"python demo.py --demo dataset")
            print(f"python train.py --config configs/{args.dataset_type.lower()}.yaml --exp_name test_run")
        else:
            print("❌ Verification failed")
        
    except Exception as e:
        print(f"❌ Error creating test dataset: {e}")
        return
    
    create_sample_images_from_web()


if __name__ == "__main__":
    main()
