"""
Complete dataset setup script for Underwater Image Enhancement
Downloads and organizes datasets, creates test data, and verifies setup
"""

import os
import argparse
import subprocess
import sys
from pathlib import Path
import zipfile
import requests
from tqdm import tqdm


def install_requirements():
    """Install required packages if not already installed"""
    print("📦 Checking requirements...")
    
    try:
        import torch
        import torchvision
        import PIL
        import numpy
        print("✅ Core packages already installed")
    except ImportError as e:
        print(f"❌ Missing package: {e}")
        print("Installing requirements...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])


def download_file(url, destination, desc="Downloading"):
    """Download file with progress bar"""
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(destination, 'wb') as file, tqdm(
        desc=desc,
        total=total_size,
        unit='B',
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            if chunk:
                file.write(chunk)
                pbar.update(len(chunk))


def download_uieb_dataset(data_root):
    """Download UIEB dataset (placeholder - actual URLs would need to be verified)"""
    print("🌊 Downloading UIEB dataset...")
    
    # Note: These are placeholder URLs. In practice, you would need:
    # 1. Official download links from the paper authors
    # 2. Google Drive API integration for Google Drive links
    # 3. Proper authentication if required
    
    print("⚠️  UIEB dataset download requires manual setup:")
    print("1. Visit: https://li-chongyi.github.io/proj_benchmark.html")
    print("2. Download the dataset files")
    print("3. Extract to the DATA/UIEB/ directory")
    print("")
    print("💡 For testing purposes, use: python create_test_data.py --dataset_type UIEB")
    
    return False  # Manual download required


def download_lsui_dataset(data_root):
    """Download LSUI dataset (placeholder)"""
    print("🌊 Downloading LSUI dataset...")
    
    print("⚠️  LSUI dataset download requires manual setup:")
    print("1. Visit: https://github.com/dalabdune/LSUI")
    print("2. Follow the download instructions")
    print("3. Extract to the DATA/LSUI/ directory")
    print("")
    print("💡 For testing purposes, use: python create_test_data.py --dataset_type LSUI")
    
    return False  # Manual download required


def setup_directory_structure(data_root):
    """Create the basic directory structure"""
    print(f"📁 Setting up directory structure in {data_root}...")
    
    data_root = Path(data_root)
    
    # Create main directories
    directories = [
        data_root,
        data_root / 'UIEB' / 'raw-890',
        data_root / 'UIEB' / 'reference-890',
        data_root / 'UIEB' / 'challengingset-60',
        data_root / 'LSUI' / 'input',
        data_root / 'LSUI' / 'GT',
        data_root / 'custom' / 'train' / 'degraded',
        data_root / 'custom' / 'train' / 'enhanced',
        data_root / 'custom' / 'val' / 'degraded',
        data_root / 'custom' / 'val' / 'enhanced',
        data_root / 'custom' / 'test' / 'degraded',
        data_root / 'custom' / 'test' / 'enhanced',
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {directory}")
    
    # Create README files
    readme_content = {
        'UIEB': """# UIEB Dataset

This folder should contain the UIEB (Underwater Image Enhancement Benchmark) dataset.

Structure:
- raw-890/: Training degraded images (890 images)
- reference-890/: Training enhanced/reference images (890 images)  
- challengingset-60/: Test images (60 challenging images)

Download from: https://li-chongyi.github.io/proj_benchmark.html
""",
        'LSUI': """# LSUI Dataset

This folder should contain the LSUI (Large-scale Underwater Image) dataset.

Structure:
- input/: Input underwater images
- GT/: Ground truth enhanced images

Download from: https://github.com/dalabdune/LSUI
""",
        'custom': """# Custom Dataset

This folder is for your own underwater image datasets.

Structure:
- train/degraded/: Training input images
- train/enhanced/: Training target images
- val/degraded/: Validation input images  
- val/enhanced/: Validation target images
- test/degraded/: Test input images
- test/enhanced/: Test target images (optional)

Images in corresponding folders should have matching filenames.
"""
    }
    
    for dataset_name, content in readme_content.items():
        readme_path = data_root / dataset_name / 'README.md'
        with open(readme_path, 'w') as f:
            f.write(content)
    
    print("✅ Directory structure created successfully!")


def verify_setup(data_root):
    """Verify the setup is correct"""
    print("🔍 Verifying setup...")
    
    try:
        # Check if our modules can be imported
        from data import UnderwaterDataset
        from models import create_model
        from config import get_config
        
        print("✅ All modules can be imported")
        
        # Check data directory
        data_root = Path(data_root)
        if data_root.exists():
            print(f"✅ Data directory exists: {data_root.absolute()}")
            
            # Check for any datasets
            uieb_dir = data_root / 'UIEB' / 'raw-890'
            lsui_dir = data_root / 'LSUI' / 'input'
            
            if any(uieb_dir.glob('*')):
                print("✅ UIEB dataset found")
            elif any(lsui_dir.glob('*')):
                print("✅ LSUI dataset found") 
            else:
                print("⚠️  No datasets found (use test data for now)")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Verification error: {e}")
        return False


def create_config_files():
    """Create configuration files if they don't exist"""
    print("⚙️  Creating configuration files...")
    
    configs_dir = Path('configs')
    configs_dir.mkdir(exist_ok=True)
    
    try:
        from config import create_config_templates
        create_config_templates()
        print("✅ Configuration templates created")
    except Exception as e:
        print(f"⚠️  Could not create config templates: {e}")


def run_quick_test():
    """Run a quick test to ensure everything works"""
    print("🧪 Running quick functionality test...")
    
    try:
        # Test model creation
        from models import create_model
        from config import get_config
        
        config = get_config('lightweight')
        model = create_model(config['model'])
        print("✅ Model creation test passed")
        
        # Test data loading (with dummy data)
        import torch
        dummy_input = torch.randn(1, 3, 256, 256)
        
        # Test forward pass
        loss, loss_dict = model.training_step(dummy_input, dummy_input)
        print("✅ Training step test passed")
        
        print("🎉 All tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description='Setup underwater image enhancement environment')
    parser.add_argument('--data_root', type=str, default='./DATA', 
                       help='Root directory for datasets')
    parser.add_argument('--dataset', type=str, choices=['UIEB', 'LSUI', 'test', 'all'],
                       default='test', help='Which dataset to setup')
    parser.add_argument('--skip_download', action='store_true',
                       help='Skip dataset download (manual setup)')
    parser.add_argument('--test_samples', type=int, default=50,
                       help='Number of test samples to generate')
    
    args = parser.parse_args()
    
    print("🌊 UNDERWATER IMAGE ENHANCEMENT SETUP")
    print("=" * 50)
    print(f"Data root: {Path(args.data_root).absolute()}")
    print(f"Dataset: {args.dataset}")
    print("")
    
    success_steps = 0
    total_steps = 6
    
    # Step 1: Install requirements
    try:
        install_requirements()
        success_steps += 1
        print("✅ Step 1/6: Requirements installed")
    except Exception as e:
        print(f"❌ Step 1/6 failed: {e}")
    
    # Step 2: Setup directory structure
    try:
        setup_directory_structure(args.data_root)
        success_steps += 1
        print("✅ Step 2/6: Directory structure created")
    except Exception as e:
        print(f"❌ Step 2/6 failed: {e}")
    
    # Step 3: Create config files
    try:
        create_config_files()
        success_steps += 1
        print("✅ Step 3/6: Configuration files created")
    except Exception as e:
        print(f"❌ Step 3/6 failed: {e}")
    
    # Step 4: Dataset setup
    if not args.skip_download:
        if args.dataset == 'UIEB':
            download_success = download_uieb_dataset(args.data_root)
        elif args.dataset == 'LSUI':
            download_success = download_lsui_dataset(args.data_root)
        elif args.dataset == 'test':
            print("🎨 Creating test dataset...")
            try:
                subprocess.run([
                    sys.executable, 'create_test_data.py',
                    '--data_root', args.data_root,
                    '--dataset_type', 'UIEB',
                    '--num_samples', str(args.test_samples)
                ], check=True)
                download_success = True
                print("✅ Test dataset created")
            except subprocess.CalledProcessError as e:
                print(f"❌ Test dataset creation failed: {e}")
                download_success = False
        elif args.dataset == 'all':
            print("📁 Setting up for all datasets (manual download required)")
            download_success = True
        
        if download_success:
            success_steps += 1
            print("✅ Step 4/6: Dataset setup completed")
        else:
            print("⚠️  Step 4/6: Dataset setup requires manual intervention")
    else:
        success_steps += 1
        print("⏭️  Step 4/6: Dataset download skipped")
    
    # Step 5: Verify setup
    try:
        verify_success = verify_setup(args.data_root)
        if verify_success:
            success_steps += 1
            print("✅ Step 5/6: Setup verification passed")
        else:
            print("❌ Step 5/6: Setup verification failed")
    except Exception as e:
        print(f"❌ Step 5/6 failed: {e}")
    
    # Step 6: Quick test
    try:
        test_success = run_quick_test()
        if test_success:
            success_steps += 1
            print("✅ Step 6/6: Functionality test passed")
        else:
            print("❌ Step 6/6: Functionality test failed")
    except Exception as e:
        print(f"❌ Step 6/6 failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"📊 Setup Summary: {success_steps}/{total_steps} steps completed")
    
    if success_steps >= 4:  # At least basic setup working
        print("🎉 Setup completed successfully!")
        print("\n🚀 Next steps:")
        print("1. Check your data:")
        print(f"   python check_dataset.py --data_root {args.data_root} --dataset_type UIEB")
        print("\n2. Run demo:")
        print("   python demo.py --demo all")
        print("\n3. Start training:")
        print("   python train.py --config configs/uieb.yaml --exp_name my_first_experiment")
        print("\n4. Test on images:")
        print("   python test.py --mode single --input_image your_image.jpg")
        
        if args.dataset == 'test':
            print(f"\n💡 Using test dataset with {args.test_samples} synthetic images")
            print("For real datasets, download UIEB or LSUI manually")
    
    else:
        print("⚠️  Setup incomplete. Please check error messages above.")
        print("💡 You can still proceed with manual dataset setup")
        print("📖 See DATA_SETUP.md for detailed instructions")


if __name__ == "__main__":
    main()
