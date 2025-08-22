"""
Dataset loaders for UIEB and LSUI underwater image enhancement datasets
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
# models 폴더의 water_physics에서 WaterNetPreprocessor를 import합니다.
from ..models.water_physics import WaterNetPreprocessor

class UnderwaterDataset(Dataset):
    """General underwater image enhancement dataset for UIEB and LSUI"""
    def __init__(self, root_dir, dataset_type, split, img_size, augment, preprocessing_type):
        self.root_dir = Path(root_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        self.preprocessing_type = preprocessing_type

        if dataset_type == 'UIEB':
            self.degraded_dir = self.root_dir / 'UIEB' / 'raw-890'
            self.enhanced_dir = self.root_dir / 'UIEB' / 'reference-890'
        elif dataset_type == 'LSUI':
            self.degraded_dir = self.root_dir / 'LSUI' / 'input'
            self.enhanced_dir = self.root_dir / 'LSUI' / 'GT'
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        self.data_pairs = self._load_data_pairs()
        self.setup_transforms()
        print(f"Loaded {len(self.data_pairs)} images for {dataset_type} {split} set.")

    def _load_data_pairs(self):
        pairs = []
        degraded_files = sorted(self.degraded_dir.glob('*.*'))
        for degraded_file in degraded_files:
            enhanced_file = self.enhanced_dir / degraded_file.name
            if enhanced_file.exists():
                pairs.append((degraded_file, enhanced_file))
        return pairs

    def setup_transforms(self):
        transform_list = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) # -1 to 1 range
        ]
        if self.augment:
            self.transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
            ] + transform_list)
        else:
            self.transform = transforms.Compose(transform_list)

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        degraded_path, enhanced_path = self.data_pairs[idx]
        degraded_img = Image.open(degraded_path).convert('RGB')
        enhanced_img = Image.open(enhanced_path).convert('RGB')
        
        degraded = self.transform(degraded_img)
        enhanced = self.transform(enhanced_img)

        result = {'degraded': degraded, 'enhanced': enhanced}

        if self.preprocessing_type == 'waternet':
            # unsqueeze/squeeze to create a batch of 1 for the preprocessor
            preprocessed_tensor = WaterNetPreprocessor.preprocess_batch(degraded.unsqueeze(0)).squeeze(0)
            result['preprocessed'] = preprocessed_tensor
            
        return result


def create_dataloaders(config_data):
    train_dataset = UnderwaterDataset(
        root_dir=config_data['data_root'],
        dataset_type=config_data['dataset_type'],
        split='train',
        img_size=config_data['img_size'],
        augment=config_data['augment'],
        preprocessing_type=config_data['preprocessing_type']
    )
    val_dataset = UnderwaterDataset(
        root_dir=config_data['data_root'],
        dataset_type=config_data['dataset_type'],
        split='val',
        img_size=config_data['img_size'],
        augment=False,
        preprocessing_type=config_data['preprocessing_type']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_data['batch_size'],
        shuffle=True,
        num_workers=config_data.get('num_workers', 2), # Add default value
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config_data['batch_size'],
        shuffle=False,
        num_workers=config_data.get('num_workers', 2),
        pin_memory=True
    )
    return train_loader, val_loader