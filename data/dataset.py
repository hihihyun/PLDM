"""
Dataset loaders for UIEB and LSUI underwater image enhancement datasets
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from pathlib import Path
import numpy as np

class UnderwaterDataset(Dataset):
    def __init__(self, root_dir, dataset_type, split, img_size, augment):
        self.root_dir = Path(root_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'

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
        image_extensions = ['*.png', '*.jpg', '*.jpeg']
        degraded_files = []
        for ext in image_extensions:
            degraded_files.extend(list(self.degraded_dir.glob(ext)))
        
        degraded_files = sorted(degraded_files)

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

        # 👇 [수정] 전처리 로직을 모델 내부로 옮겼으므로 여기서 제거
        result = {'degraded': degraded, 'enhanced': enhanced}
            
        return result


def create_dataloaders(config_data):
    train_dataset = UnderwaterDataset(
        root_dir=config_data['data_root'],
        dataset_type=config_data['dataset_type'],
        split='train',
        img_size=config_data['img_size'],
        augment=config_data['augment']
    )
    # A simple validation split (e.g., last 10% of training data)
    # For a real scenario, you'd have a dedicated validation set.
    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(0.1 * num_train))
    
    # Use a fixed random seed for reproducibility of splits
    np.random.seed(42)
    np.random.shuffle(indices)

    train_indices, val_indices = indices[split:], indices[:split]

    train_sampler = torch.utils.data.SubsetRandomSampler(train_indices)
    val_sampler = torch.utils.data.SubsetRandomSampler(val_indices)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config_data['batch_size'],
        sampler=train_sampler,
        num_workers=config_data.get('num_workers', 2),
        pin_memory=True
    )
    val_loader = DataLoader(
        train_dataset, # Use the same dataset object
        batch_size=config_data['batch_size'],
        sampler=val_sampler,
        num_workers=config_data.get('num_workers', 2),
        pin_memory=True
    )
    return train_loader, val_loader