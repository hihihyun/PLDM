"""
Underwater Image Enhancement with Diffusion Models

A comprehensive toolkit for underwater image enhancement using
VAE-Diffusion models with Water-Net physics-based preprocessing.
"""

# Main model imports
from models import (
    UnderwaterEnhancementDiffusion,
    UnderwaterEnhancementModel,
    create_model
)

# Data imports
from data import (
    UnderwaterDataset,
    create_dataloaders,
    get_dataset_config
)

# Configuration imports
from config import get_config, save_config

# Utility imports
from utils import (
    set_seed,
    calculate_metrics,
    save_single_image,
    load_image,
    print_model_info
)

__version__ = '1.0.0'
__author__ = 'Underwater Enhancement Research Team'
__email__ = 'contact@underwater-enhancement.com'
__description__ = 'Underwater Image Enhancement using Diffusion Models'
__url__ = 'https://github.com/underwater-enhancement/diffusion-model'

__all__ = [
    # Main classes
    'UnderwaterEnhancementDiffusion',
    'UnderwaterEnhancementModel',
    'UnderwaterDataset',
    
    # Factory functions
    'create_model',
    'create_dataloaders',
    'get_config',
    'get_dataset_config',
    
    # Utilities
    'set_seed',
    'calculate_metrics',
    'save_single_image',
    'load_image',
    'print_model_info',
    'save_config'
]

# Quick setup function
def quick_setup(dataset_type='UIEB', model_type='default', device='cuda'):
    """
    Quick setup for underwater image enhancement
    
    Args:
        dataset_type: 'UIEB' or 'LSUI'
        model_type: 'default', 'lightweight', or 'high_quality' 
        device: 'cuda' or 'cpu'
    
    Returns:
        model, config, dataset_config
    """
    # Get configurations
    config = get_config(config_name=model_type)
    config['device'] = device
    
    dataset_config = get_dataset_config(dataset_type)
    
    # Create model
    model = create_model(config['model'])
    
    return model, config, dataset_config

# Package info
def print_package_info():
    """Print package information"""
    print("=" * 60)
    print("UNDERWATER IMAGE ENHANCEMENT WITH DIFFUSION MODELS")
    print("=" * 60)
    print(f"Version: {__version__}")
    print(f"Author: {__author__}")
    print(f"Description: {__description__}")
    print("=" * 60)
    print("\nQuick Start:")
    print("  from underwater_enhancement import quick_setup")
    print("  model, config, dataset_config = quick_setup('UIEB')")
    print("\nFor detailed documentation, see README.md")
    print("=" * 60)

# Show info when imported
if __name__ != "__main__":
    print_package_info()
