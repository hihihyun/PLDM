"""
Dataset loaders for UIEB and LSUI underwater image enhancement datasets
"""
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
import cv2
from pathlib import Path

class UnderwaterDataset(Dataset):
    """General underwater image enhancement dataset for UIEB and LSUI"""
    def __init__(self, root_dir, dataset_type, split, img_size, augment, preprocessing_type):
        self.root_dir = Path(root_dir)
        self.dataset_type = dataset_type
        self.split = split
        self.img_size = img_size
        self.augment = augment and split == 'train'
        self.preprocessing_type = preprocessing_type

        # 1. 데이터셋 경로 설정
        if dataset_type == 'UIEB':
            self.degraded_dir = self.root_dir / 'UIEB' / 'raw-890'
            self.enhanced_dir = self.root_dir / 'UIEB' / 'reference-890'
        elif dataset_type == 'LSUI':
            # LSUI는 별도의 훈련/검증 구분이 없어 전체를 사용합니다.
            self.degraded_dir = self.root_dir / 'LSUI' / 'input'
            self.enhanced_dir = self.root_dir / 'LSUI' / 'GT'
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        # 2. 이미지 파일 목록 불러오기
        self.data_pairs = self._load_data_pairs()

        # 3. 이미지 변환 설정
        self.setup_transforms()
        print(f"Loaded {len(self.data_pairs)} images for {dataset_type} {split} set.")

    def _load_data_pairs(self):
        """ degraded와 enhanced 이미지 파일 쌍을 찾습니다. """
        pairs = []
        degraded_files = sorted(self.degraded_dir.glob('*.*'))

        for degraded_file in degraded_files:
            enhanced_file = self.enhanced_dir / degraded_file.name
            if enhanced_file.exists():
                pairs.append((degraded_file, enhanced_file))
        return pairs

    def setup_transforms(self):
        """이미지 변환 (리사이즈, 텐서 변환, 증강)을 설정합니다."""
        self.base_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)) # [-1, 1] 범위로 정규화
        ])
        if self.augment:
            self.augment_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
                transforms.Resize((self.img_size, self.img_size)),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
            ])
        else:
            self.augment_transform = self.base_transform

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        degraded_path, enhanced_path = self.data_pairs[idx]

        degraded_img = Image.open(degraded_path).convert('RGB')
        enhanced_img = Image.open(enhanced_path).convert('RGB')

        if self.augment:
            degraded = self.augment_transform(degraded_img)
            enhanced = self.augment_transform(enhanced_img)
        else:
            degraded = self.base_transform(degraded_img)
            enhanced = self.base_transform(enhanced_img)
            
        result = {'degraded': degraded, 'enhanced': enhanced}

        if self.preprocessing_type == 'waternet':
             # WaterNetPreprocessor를 사용하여 전처리된 이미지 생성
            preprocessed = WaterNetPreprocessor.preprocess_batch(degraded.unsqueeze(0)).squeeze(0)
            result['preprocessed'] = preprocessed

        return result


def create_dataloaders(config):
    """훈련 및 검증 데이터로더를 생성합니다."""
    train_dataset = UnderwaterDataset(
        root_dir=config['data_root'],
        dataset_type=config['dataset_type'],
        split='train',
        img_size=config['img_size'],
        augment=config['augment'],
        preprocessing_type=config['preprocessing_type']
    )
    # LSUI와 같이 별도의 val set이 없는 경우, train set의 일부를 사용하도록 할 수 있습니다.
    # 여기서는 간단하게 train_dataset을 그대로 사용합니다.
    val_dataset = UnderwaterDataset(
        root_dir=config['data_root'],
        dataset_type=config['dataset_type'],
        split='val', # 실제로는 train 데이터셋을 사용하지만, augment는 적용하지 않습니다.
        img_size=config['img_size'],
        augment=False,
        preprocessing_type=config['preprocessing_type']
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        pin_memory=True
    )
    return train_loader, val_loader