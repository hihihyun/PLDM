"""
Dataset loaders for UIEB and LSUI underwater image enhancement datasets
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path
import json


class UnderwaterDataset(Dataset):
    """General underwater image enhancement dataset"""
    def __init__(self, root_dir, dataset_type='UIEB', split='train', 
                 img_size=256, augment=True, preprocessing_type='all'):
        """
        Args:
            root_dir: Root directory containing the dataset
            dataset_type: 'UIEB' or 'LSUI'
            split: 'train', 'val', or 'test'
            img_size: Target image size for resizing
            augment: Whether to apply data augmentation
            preprocessing_type: 'all', 'waternet', or 'none'
        """
        self.root_dir = Path(root_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        self.preprocessing_type = preprocessing_type
        
        # Dataset paths
        if dataset_type == 'UIEB':
            self.setup_uieb_paths()
        elif dataset_type == 'LSUI':
            self.setup_lsui_paths()
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")
        
        # Get file pairs
        self.data_pairs = self.load_data_pairs()
        
        # Transforms
        self.setup_transforms()
        
        print(f"Loaded {len(self.data_pairs)} {split} pairs from {dataset_type}")
    
    def setup_uieb_paths(self):
        """Setup paths for UIEB dataset"""
        if self.split == 'train':
            self.degraded_dir = self.root_dir / 'UIEB' / 'raw-890'
            self.enhanced_dir = self.root_dir / 'UIEB' / 'reference-890'
        else:
            self.degraded_dir = self.root_dir / 'UIEB' / 'challengingset-60'
            # For test set, we might not have ground truth
            self.enhanced_dir = None
    
    def setup_lsui_paths(self):
        """Setup paths for LSUI dataset"""
        self.degraded_dir = self.root_dir / 'LSUI' / 'input'
        self.enhanced_dir = self.root_dir / 'LSUI' / 'GT'
    
    def load_data_pairs(self):
        """Load image file pairs with better validation split handling"""
        pairs = []
        
        if not self.degraded_dir.exists():
            raise FileNotFoundError(f"Degraded directory not found: {self.degraded_dir}")
        
        # Get degraded image files
        degraded_files = sorted([f for f in self.degraded_dir.glob('*') 
                               if f.suffix.lower() in ['.jpg', '.jpeg', '.png']])
        
        # For validation split, use a subset of training data if no separate validation set
        if self.split == 'val' and self.dataset_type == 'UIEB':
            # Use last 20% of training data for validation
            total_files = len(degraded_files)
            val_start = int(total_files * 0.8)
            degraded_files = degraded_files[val_start:]
            print(f"Using last {len(degraded_files)} files for validation")
        elif self.split == 'train' and self.dataset_type == 'UIEB':
            # Use first 80% for training
            total_files = len(degraded_files)
            train_end = int(total_files * 0.8)
            degraded_files = degraded_files[:train_end]
            print(f"Using first {len(degraded_files)} files for training")
        
        successful_pairs = 0
        for degraded_file in degraded_files:
            if self.enhanced_dir and self.enhanced_dir.exists():
                # Look for corresponding enhanced image
                enhanced_file = self.enhanced_dir / degraded_file.name
                if enhanced_file.exists():
                    pairs.append((degraded_file, enhanced_file))
                    successful_pairs += 1
                else:
                    # Try different naming conventions
                    found_match = False
                    for suffix in ['_enhanced', '_gt', '_reference']:
                        alt_name = degraded_file.stem + suffix + degraded_file.suffix
                        enhanced_file = self.enhanced_dir / alt_name
                        if enhanced_file.exists():
                            pairs.append((degraded_file, enhanced_file))
                            successful_pairs += 1
                            found_match = True
                            break
                    
                    if not found_match:
                        print(f"Warning: No enhanced version found for {degraded_file.name}")
                        # For validation, skip unpaired images
                        if self.split == 'val':
                            continue
                        # For training, we might still use it (depends on use case)
                        else:
                            pairs.append((degraded_file, None))
            else:
                # Test set without ground truth
                pairs.append((degraded_file, None))
        
        print(f"Successfully paired {successful_pairs}/{len(degraded_files)} images for {self.split}")
        return pairs
    
    def setup_transforms(self):
        """Setup image transforms"""
        # Basic transforms
        self.basic_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
        ])
        
        # Augmentation transforms
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=15),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
                transforms.RandomResizedCrop(self.img_size, scale=(0.8, 1.0)),
                transforms.ToTensor(),
            ])
        else:
            self.augment_transform = self.basic_transform
    
    def apply_waternet_preprocessing(self, image):
        """Apply Water-Net style preprocessing"""
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        
        # Convert to numpy for OpenCV operations
        img_np = np.array(image)
        
        # White balance
        wb_img = self.simple_white_balance(img_np)
        
        # Gamma correction
        gamma_img = np.power(img_np / 255.0, 0.7) * 255
        gamma_img = gamma_img.astype(np.uint8)
        
        # Histogram equalization (CLAHE)
        he_img = self.clahe_enhancement(img_np)
        
        # Convert back to tensors
        wb_tensor = self.basic_transform(Image.fromarray(wb_img))
        gamma_tensor = self.basic_transform(Image.fromarray(gamma_img))
        he_tensor = self.basic_transform(Image.fromarray(he_img))
        
        # Concatenate all preprocessing results
        preprocessed = torch.cat([wb_tensor, gamma_tensor, he_tensor], dim=0)  # [9, H, W]
        
        return preprocessed
    
    def simple_white_balance(self, image):
        """Simple white balance using gray world assumption"""
        # Compute channel means
        mean_r = np.mean(image[:, :, 0])
        mean_g = np.mean(image[:, :, 1])
        mean_b = np.mean(image[:, :, 2])
        
        # Gray world assumption
        gray_mean = (mean_r + mean_g + mean_b) / 3
        
        # Compute scaling factors
        scale_r = gray_mean / mean_r if mean_r > 0 else 1.0
        scale_g = gray_mean / mean_g if mean_g > 0 else 1.0
        scale_b = gray_mean / mean_b if mean_b > 0 else 1.0
        
        # Apply scaling
        balanced = image.copy().astype(np.float32)
        balanced[:, :, 0] = np.clip(balanced[:, :, 0] * scale_r, 0, 255)
        balanced[:, :, 1] = np.clip(balanced[:, :, 1] * scale_g, 0, 255)
        balanced[:, :, 2] = np.clip(balanced[:, :, 2] * scale_b, 0, 255)
        
        return balanced.astype(np.uint8)
    
    def clahe_enhancement(self, image):
        """CLAHE histogram equalization"""
        # Convert RGB to LAB
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        
        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab[:, :, 0] = clahe.apply(lab[:, :, 0])
        
        # Convert back to RGB
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        
        return enhanced
    
    def __len__(self):
        return len(self.data_pairs)
    
    def __getitem__(self, idx):
        degraded_path, enhanced_path = self.data_pairs[idx]
        
        # Load degraded image
        degraded_img = Image.open(degraded_path).convert('RGB')
        
        # Apply transforms
        if self.augment:
            # Use same random seed for both images
            seed = np.random.randint(0, 2**32)
            
            # Transform degraded image
            np.random.seed(seed)
            torch.manual_seed(seed)
            degraded_tensor = self.augment_transform(degraded_img)
        else:
            degraded_tensor = self.basic_transform(degraded_img)
        
        result = {
            'degraded': degraded_tensor,
            'filename': degraded_path.name
        }
        
        # Load enhanced image if available
        if enhanced_path is not None:
            enhanced_img = Image.open(enhanced_path).convert('RGB')
            
            if self.augment:
                # Use same random seed for consistency
                np.random.seed(seed)
                torch.manual_seed(seed)
                enhanced_tensor = self.augment_transform(enhanced_img)
            else:
                enhanced_tensor = self.basic_transform(enhanced_img)
            
            result['enhanced'] = enhanced_tensor
        
        # Apply Water-Net preprocessing if requested
        if self.preprocessing_type in ['waternet', 'all']:
            preprocessed = self.apply_waternet_preprocessing(degraded_tensor)
            result['preprocessed'] = preprocessed
        
        return result


class PairedUnderwaterDataset(Dataset):
    """Dataset specifically for paired underwater images with precise alignment"""
    def __init__(self, degraded_dir, enhanced_dir, img_size=256, augment=True):
        self.degraded_dir = Path(degraded_dir)
        self.enhanced_dir = Path(enhanced_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get paired files
        self.pairs = self.get_paired_files()
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
        
        if augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.3),
                transforms.RandomRotation(degrees=10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
            ])
    
    def get_paired_files(self):
        """Get paired degraded and enhanced files"""
        pairs = []
        
        degraded_files = sorted(list(self.degraded_dir.glob('*')))
        
        for deg_file in degraded_files:
            if deg_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                # Look for corresponding enhanced file
                enh_file = self.enhanced_dir / deg_file.name
                if enh_file.exists():
                    pairs.append((deg_file, enh_file))
        
        return pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        deg_path, enh_path = self.pairs[idx]
        
        # Load images
        deg_img = Image.open(deg_path).convert('RGB')
        enh_img = Image.open(enh_path).convert('RGB')
        
        # Apply same transforms to both images
        if self.augment:
            seed = np.random.randint(0, 2**32)
            
            np.random.seed(seed)
            torch.manual_seed(seed)
            deg_tensor = self.augment_transform(deg_img)
            
            np.random.seed(seed)
            torch.manual_seed(seed)
            enh_tensor = self.augment_transform(enh_img)
        else:
            deg_tensor = self.transform(deg_img)
            enh_tensor = self.transform(enh_img)
        
        return {
            'degraded': deg_tensor,
            'enhanced': enh_tensor,
            'filename': deg_path.name
        }


def create_dataloaders(config):
    """Create train and validation dataloaders with robust error handling"""
    
    try:
        # Training dataset
        train_dataset = UnderwaterDataset(
            root_dir=config['data_root'],
            dataset_type=config['dataset_type'],
            split='train',
            img_size=config['img_size'],
            augment=config['augment'],
            preprocessing_type=config['preprocessing_type']
        )
        
        print(f"✅ Training dataset loaded: {len(train_dataset)} samples")
        
    except Exception as e:
        print(f"❌ Error loading training dataset: {e}")
        raise
    
    try:
        # Validation dataset - use same data as training but different split
        val_dataset = UnderwaterDataset(
            root_dir=config['data_root'],
            dataset_type=config['dataset_type'],
            split='val',
            img_size=config['img_size'],
            augment=False,  # No augmentation for validation
            preprocessing_type=config['preprocessing_type']
        )
        
        print(f"✅ Validation dataset loaded: {len(val_dataset)} samples")
        
        # If validation dataset is empty, use a subset of training data
        if len(val_dataset) == 0:
            print("⚠️  Validation dataset is empty, using subset of training data")
            # Create a subset of training data for validation
            train_size = len(train_dataset)
            val_size = max(1, train_size // 10)  # Use 10% for validation
            val_indices = list(range(train_size - val_size, train_size))
            
            from torch.utils.data import Subset
            val_dataset = Subset(train_dataset, val_indices)
            print(f"✅ Created validation subset: {len(val_dataset)} samples")
            
    except Exception as e:
        print(f"⚠️  Error loading validation dataset: {e}")
        print("Using subset of training data for validation")
        
        # Fallback: use subset of training data
        train_size = len(train_dataset)
        val_size = max(1, train_size // 10)
        val_indices = list(range(train_size - val_size, train_size))
        
        from torch.utils.data import Subset
        val_dataset = Subset(train_dataset, val_indices)
        print(f"✅ Fallback validation subset: {len(val_dataset)} samples")
    
    # DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    
    print(f"✅ DataLoaders created - Train: {len(train_loader)} batches, Val: {len(val_loader)} batches")
    
    return train_loader, val_loader


def save_dataset_info(dataset, save_path):
    """Save dataset information for reproducibility"""
    info = {
        'dataset_type': dataset.dataset_type,
        'split': dataset.split,
        'num_samples': len(dataset),
        'img_size': dataset.img_size,
        'augment': dataset.augment,
        'preprocessing_type': dataset.preprocessing_type,
        'sample_files': [str(pair[0]) for pair in dataset.data_pairs[:10]]  # First 10 files
    }
    
    with open(save_path, 'w') as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    # Test dataset loading
    config = {
        'data_root': './DATA',
        'dataset_type': 'UIEB',
        'img_size': 256,
        'batch_size': 4,
        'augment': True,
        'preprocessing_type': 'waternet',
        'num_workers': 4
    }
    
    try:
        # Create datasets
        train_loader, val_loader = create_dataloaders(config)
        
        print(f"Train batches: {len(train_loader)}")
        print(f"Val batches: {len(val_loader)}")
        
        # Test a batch
        for batch in train_loader:
            print("Batch keys:", batch.keys())
            print("Degraded shape:", batch['degraded'].shape)
            if 'enhanced' in batch:
                print("Enhanced shape:", batch['enhanced'].shape)
            if 'preprocessed' in batch:
                print("Preprocessed shape:", batch['preprocessed'].shape)
            break
            
    except Exception as e:
        print(f"Dataset test failed: {e}")
        print("Please ensure the dataset is properly organized in the DATA directory")
        print("Expected structure:")
        print("DATA/")
        print("  UIEB/")
        print("    raw-890/")
        print("    reference-890/")
        print("    challengingset-60/")
        print("  LSUI/")
        print("    input/")
        print("    GT/")
