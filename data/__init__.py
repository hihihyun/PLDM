"""
Data package for Underwater Image Enhancement Diffusion Model
"""

from .dataset import (
    UnderwaterDataset,
    PairedUnderwaterDataset,
    create_dataloaders,
    save_dataset_info
)

__all__ = [
    'UnderwaterDataset',
    'PairedUnderwaterDataset', 
    'create_dataloaders',
    'save_dataset_info'
]

# Dataset configuration shortcuts
UIEB_CONFIG = {
    'dataset_type': 'UIEB',
    'img_size': 256,
    'augment': True,
    'preprocessing_type': 'waternet'
}

LSUI_CONFIG = {
    'dataset_type': 'LSUI',
    'img_size': 256,
    'augment': True,
    'preprocessing_type': 'waternet'
}

def get_dataset_config(dataset_name):
    """Get predefined dataset configuration"""
    configs = {
        'UIEB': UIEB_CONFIG,
        'LSUI': LSUI_CONFIG
    }
    
    if dataset_name in configs:
        return configs[dataset_name].copy()
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(configs.keys())}")
