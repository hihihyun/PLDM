"""
Dataset checker script for Underwater Image Enhancement
Validates dataset structure and provides detailed information
"""

import os
import argparse
from pathlib import Path
from PIL import Image
import json


def check_directory_structure(data_root, dataset_type):
    """Check if the dataset directory structure is correct"""
    data_root = Path(data_root)
    
    print(f"🔍 Checking {dataset_type} dataset structure...")
    print(f"📁 Data root: {data_root.absolute()}")
    
    if not data_root.exists():
        print(f"❌ Data root directory not found: {data_root}")
        return False
    
    if dataset_type == 'UIEB':
        required_dirs = [
            'UIEB/raw-890',
            'UIEB/reference-890', 
            'UIEB/challengingset-60'
        ]
        
        print("\n📂 Expected UIEB structure:")
        print("DATA/")
        print("└── UIEB/")
        print("    ├── raw-890/          # Training degraded images")
        print("    ├── reference-890/    # Training enhanced images")
        print("    └── challengingset-60/ # Test images")
        
    elif dataset_type == 'LSUI':
        required_dirs = [
            'LSUI/input',
            'LSUI/GT'
        ]
        
        print("\n📂 Expected LSUI structure:")
        print("DATA/")
        print("└── LSUI/")
        print("    ├── input/    # Input degraded images")
        print("    └── GT/       # Ground truth images")
    
    else:
        print(f"❌ Unknown dataset type: {dataset_type}")
        return False
    
    print(f"\n🔍 Checking required directories...")
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = data_root / dir_path
        if full_path.exists():
            print(f"✅ {dir_path}")
        else:
            print(f"❌ {dir_path} - NOT FOUND")
            all_exist = False
    
    return all_exist


def count_files_and_check_pairs(data_root, dataset_type):
    """Count files and check if pairs match"""
    data_root = Path(data_root)
    
    print(f"\n📊 Analyzing {dataset_type} dataset files...")
    
    if dataset_type == 'UIEB':
        degraded_dir = data_root / 'UIEB' / 'raw-890'
        enhanced_dir = data_root / 'UIEB' / 'reference-890'
        test_dir = data_root / 'UIEB' / 'challengingset-60'
        
        # Check training pairs
        if degraded_dir.exists() and enhanced_dir.exists():
            degraded_files = list(degraded_dir.glob('*'))
            enhanced_files = list(enhanced_dir.glob('*'))
            
            degraded_files = [f for f in degraded_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            enhanced_files = [f for f in enhanced_files if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            print(f"📈 Training degraded images: {len(degraded_files)}")
            print(f"📈 Training enhanced images: {len(enhanced_files)}")
            
            # Check pairing
            degraded_names = {f.name for f in degraded_files}
            enhanced_names = {f.name for f in enhanced_files}
            
            matched_pairs = degraded_names & enhanced_names
            unmatched_degraded = degraded_names - enhanced_names
            unmatched_enhanced = enhanced_names - degraded_names
            
            print(f"✅ Matched pairs: {len(matched_pairs)}")
            if unmatched_degraded:
                print(f"⚠️  Unmatched degraded: {len(unmatched_degraded)}")
                if len(unmatched_degraded) <= 5:
                    print(f"   Examples: {list(unmatched_degraded)}")
            if unmatched_enhanced:
                print(f"⚠️  Unmatched enhanced: {len(unmatched_enhanced)}")
                if len(unmatched_enhanced) <= 5:
                    print(f"   Examples: {list(unmatched_enhanced)}")
        
        # Check test set
        if test_dir.exists():
            test_files = [f for f in test_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            print(f"📈 Test images: {len(test_files)}")
    
    elif dataset_type == 'LSUI':
        input_dir = data_root / 'LSUI' / 'input'
        gt_dir = data_root / 'LSUI' / 'GT'
        
        if input_dir.exists() and gt_dir.exists():
            input_files = [f for f in input_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            gt_files = [f for f in gt_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
            
            print(f"📈 Input images: {len(input_files)}")
            print(f"📈 GT images: {len(gt_files)}")
            
            # Check pairing
            input_names = {f.name for f in input_files}
            gt_names = {f.name for f in gt_files}
            
            matched_pairs = input_names & gt_names
            print(f"✅ Matched pairs: {len(matched_pairs)}")


def check_image_properties(data_root, dataset_type, num_samples=5):
    """Check properties of sample images"""
    print(f"\n🖼️  Checking sample image properties...")
    
    data_root = Path(data_root)
    
    if dataset_type == 'UIEB':
        sample_dir = data_root / 'UIEB' / 'raw-890'
    elif dataset_type == 'LSUI':
        sample_dir = data_root / 'LSUI' / 'input'
    else:
        return
    
    if not sample_dir.exists():
        print(f"❌ Sample directory not found: {sample_dir}")
        return
    
    image_files = [f for f in sample_dir.glob('*') if f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']]
    
    if not image_files:
        print(f"❌ No image files found in {sample_dir}")
        return
    
    sample_files = image_files[:num_samples]
    
    sizes = []
    formats = []
    
    for img_path in sample_files:
        try:
            with Image.open(img_path) as img:
                sizes.append(img.size)
                formats.append(img.format)
                print(f"📷 {img_path.name}: {img.size} - {img.format}")
        except Exception as e:
            print(f"❌ Error reading {img_path.name}: {e}")
    
    if sizes:
        widths = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        
        print(f"\n📊 Image statistics:")
        print(f"   Width range: {min(widths)} - {max(widths)}")
        print(f"   Height range: {min(heights)} - {max(heights)}")
        print(f"   Formats: {set(formats)}")


def test_data_loading(data_root, dataset_type):
    """Test if the dataset can be loaded properly"""
    print(f"\n🧪 Testing data loading...")
    
    try:
        from data import UnderwaterDataset
        
        dataset = UnderwaterDataset(
            root_dir=data_root,
            dataset_type=dataset_type,
            split='train',
            img_size=256,
            augment=False,
            preprocessing_type='none'
        )
        
        print(f"✅ Dataset created successfully")
        print(f"📊 Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"📝 Sample keys: {list(sample.keys())}")
            
            for key, value in sample.items():
                if hasattr(value, 'shape'):
                    print(f"   {key}: {value.shape} ({value.dtype})")
                else:
                    print(f"   {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return False


def generate_report(data_root, dataset_type):
    """Generate a comprehensive report"""
    report = {
        'dataset_type': dataset_type,
        'data_root': str(Path(data_root).absolute()),
        'checks': {}
    }
    
    # Structure check
    structure_ok = check_directory_structure(data_root, dataset_type)
    report['checks']['structure'] = structure_ok
    
    if structure_ok:
        # File count and pairing
        count_files_and_check_pairs(data_root, dataset_type)
        
        # Image properties
        check_image_properties(data_root, dataset_type)
        
        # Data loading test
        loading_ok = test_data_loading(data_root, dataset_type)
        report['checks']['data_loading'] = loading_ok
    
    # Save report
    report_path = Path(data_root) / f'{dataset_type.lower()}_check_report.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n📋 Report saved to: {report_path}")
    
    return report


def provide_setup_instructions(dataset_type):
    """Provide setup instructions for missing datasets"""
    print(f"\n📖 Setup Instructions for {dataset_type}:")
    print("=" * 50)
    
    if dataset_type == 'UIEB':
        print("1. Download UIEB dataset:")
        print("   - Official page: https://li-chongyi.github.io/proj_benchmark.html")
        print("   - Google Drive: https://drive.google.com/drive/folders/1BVozhoEp4l_E7k4SAmtCKpTTsZLaK9xO")
        print("")
        print("2. Extract to the following structure:")
        print("   DATA/")
        print("   └── UIEB/")
        print("       ├── raw-890/")
        print("       ├── reference-890/")
        print("       └── challengingset-60/")
        print("")
        print("3. Or use the download script:")
        print("   python scripts/download_uieb.py")
    
    elif dataset_type == 'LSUI':
        print("1. Download LSUI dataset:")
        print("   - GitHub: https://github.com/dalabdune/LSUI")
        print("   - Paper: https://ieeexplore.ieee.org/document/9001231")
        print("")
        print("2. Extract to the following structure:")
        print("   DATA/")
        print("   └── LSUI/")
        print("       ├── input/")
        print("       └── GT/")


def main():
    parser = argparse.ArgumentParser(description='Check underwater image dataset')
    parser.add_argument('--dataset_type', type=str, choices=['UIEB', 'LSUI'], 
                       default='UIEB', help='Dataset type to check')
    parser.add_argument('--data_root', type=str, default='./DATA', 
                       help='Root directory containing datasets')
    parser.add_argument('--report', action='store_true', 
                       help='Generate detailed report')
    
    args = parser.parse_args()
    
    print("🔍 UNDERWATER DATASET CHECKER")
    print("=" * 40)
    
    # Check if data root exists
    if not Path(args.data_root).exists():
        print(f"❌ Data root directory not found: {args.data_root}")
        print("💡 Creating data root directory...")
        Path(args.data_root).mkdir(parents=True, exist_ok=True)
        print(f"✅ Created: {Path(args.data_root).absolute()}")
    
    # Run checks
    if args.report:
        report = generate_report(args.data_root, args.dataset_type)
        
        # Summary
        print(f"\n📋 SUMMARY")
        print("=" * 20)
        all_passed = all(report['checks'].values())
        
        if all_passed:
            print("🎉 All checks passed! Dataset is ready for use.")
        else:
            print("⚠️  Some checks failed. See details above.")
            provide_setup_instructions(args.dataset_type)
    
    else:
        # Quick check
        structure_ok = check_directory_structure(args.data_root, args.dataset_type)
        
        if structure_ok:
            print("🎉 Dataset structure looks good!")
            count_files_and_check_pairs(args.data_root, args.dataset_type)
        else:
            print("❌ Dataset setup incomplete.")
            provide_setup_instructions(args.dataset_type)
    
    print(f"\n💡 To get detailed report: python check_dataset.py --dataset_type {args.dataset_type} --report")


if __name__ == "__main__":
    main()
