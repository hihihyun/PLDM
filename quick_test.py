"""
Quick test script to verify installation and basic functionality
Run this first to check if everything is working
"""

import sys
import os
from pathlib import Path

def test_python_version():
    """Check Python version"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} (Good)")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} (Need 3.8+)")
        return False

def test_basic_imports():
    """Test basic package imports"""
    print("\n📦 Testing basic imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('torchvision', 'TorchVision'),
        ('PIL', 'Pillow'),
        ('numpy', 'NumPy'),
        ('yaml', 'PyYAML'),
    ]
    
    success_count = 0
    for package, name in packages:
        try:
            __import__(package)
            print(f"✅ {name}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {name}: {e}")
    
    print(f"Import success: {success_count}/{len(packages)}")
    return success_count == len(packages)

def test_config():
    """Test configuration loading"""
    print("\n⚙️  Testing configuration...")
    
    try:
        # Add current directory to path for imports
        sys.path.insert(0, os.getcwd())
        
        from config import get_default_config, get_config
        
        # Test direct function call
        config = get_default_config()
        print("✅ Default config created directly")
        
        # Test get_config function
        config2 = get_config('default')
        print("✅ Config loaded via get_config('default')")
        
        # Check config structure
        required_keys = ['model', 'data', 'training']
        for key in required_keys:
            if key in config:
                print(f"✅ Config has '{key}' section")
            else:
                print(f"❌ Config missing '{key}' section")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_model_import():
    """Test model imports"""
    print("\n🧠 Testing model imports...")
    
    try:
        sys.path.insert(0, os.getcwd())
        
        from models import create_model
        print("✅ Model creation function imported")
        
        from config import get_default_config
        config = get_default_config()
        
        # Try to create model (this might fail due to dependencies)
        try:
            model = create_model(config['model'])
            print("✅ Model created successfully")
            return True
        except Exception as e:
            print(f"⚠️  Model creation failed: {e}")
            print("   This might be due to missing PyTorch or other dependencies")
            return False
        
    except Exception as e:
        print(f"❌ Model import failed: {e}")
        return False

def test_data_import():
    """Test data imports"""
    print("\n💿 Testing data imports...")
    
    try:
        sys.path.insert(0, os.getcwd())
        
        from data import UnderwaterDataset
        print("✅ Dataset class imported")
        
        # Test dataset creation (will fail if no data, but import should work)
        try:
            dataset = UnderwaterDataset('./DATA', 'UIEB', 'train', augment=False)
            print(f"✅ Dataset created (size: {len(dataset)})")
            return True
        except Exception as e:
            print(f"⚠️  Dataset creation failed: {e}")
            print("   This is expected if DATA folder doesn't exist")
            return True  # Import success is what matters
        
    except Exception as e:
        print(f"❌ Data import failed: {e}")
        return False

def test_directory_structure():
    """Check basic directory structure"""
    print("\n📁 Checking directory structure...")
    
    required_files = [
        'config.py',
        'utils.py', 
        'models/__init__.py',
        'models/main_model.py',
        'data/__init__.py',
        'data/dataset.py'
    ]
    
    missing_files = []
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path}")
        else:
            print(f"❌ {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"⚠️  Missing files: {missing_files}")
        return False
    
    return True

def create_configs_if_needed():
    """Create config directory and files if needed"""
    print("\n📝 Setting up configuration files...")
    
    try:
        configs_dir = Path('configs')
        configs_dir.mkdir(exist_ok=True)
        
        # Create basic config files
        sys.path.insert(0, os.getcwd())
        from config import get_default_config, get_uieb_config, get_lightweight_config, save_config
        
        configs = {
            'default.yaml': get_default_config(),
            'uieb.yaml': get_uieb_config(), 
            'lightweight.yaml': get_lightweight_config()
        }
        
        for filename, config in configs.items():
            config_path = configs_dir / filename
            if not config_path.exists():
                save_config(config, config_path)
                print(f"✅ Created {filename}")
            else:
                print(f"✅ {filename} already exists")
        
        return True
        
    except Exception as e:
        print(f"❌ Config creation failed: {e}")
        return False

def main():
    print("🚀 UNDERWATER IMAGE ENHANCEMENT - QUICK TEST")
    print("=" * 50)
    
    tests = [
        ("Python Version", test_python_version),
        ("Basic Imports", test_basic_imports),
        ("Directory Structure", test_directory_structure),
        ("Configuration", test_config),
        ("Model Imports", test_model_import),
        ("Data Imports", test_data_import),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Setup configs
    print(f"\n{'='*20} Setup {'='*20}")
    create_configs_if_needed()
    
    # Summary
    print(f"\n{'='*20} SUMMARY {'='*20}")
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} {test_name}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! You're ready to use the system.")
        print("\nNext steps:")
        print("1. python demo.py --demo all")
        print("2. python setup_dataset.py --dataset test")
        print("3. python train.py --config configs/lightweight.yaml")
        
    elif passed >= total - 2:
        print("\n⚠️  Most tests passed. Minor issues may exist but system should work.")
        print("\nTry running:")
        print("python demo.py --demo all")
        
    else:
        print("\n❌ Multiple tests failed. Please check:")
        print("1. Install requirements: pip install -r requirements.txt")
        print("2. Check Python version (need 3.8+)")
        print("3. Verify all files are present")
        
    return passed == total

if __name__ == "__main__":
    main()
