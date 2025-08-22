"""
Fixed configuration file with stable settings for underwater image enhancement
"""

import os
import yaml
from typing import Dict, Any


def get_safe_config() -> Dict[str, Any]:
    """Get safe configuration that works with fixed modules"""
    return {
        'model': {
            'img_size': 256,
            'in_channels': 3,
            'latent_channels': 4,
            'base_channels': 64,  # Reduced from 128 for stability
            'time_steps': 500,    # Reduced from 1000 for faster training
            'physics_dim': 32,    # Reduced from 64 for compatibility
            'num_heads': 4,       # Reduced for stability
            'use_attention': True,
            'use_physics': True,
        },
        'training': {
            'num_epochs': 100,
            'batch_size': 4,      # Reduced for stability
            'vae_lr': 1e-4,
            'diffusion_lr': 1e-4,
            'weight_decay': 1e-5,
            'grad_clip': 1.0,
            'warmup_steps': 1000,
            'save_every': 10,
            'eval_every': 5,
            'mixed_precision': False,  # Disabled for stability
        },
        'data': {
            'data_root': './DATA',
            'dataset_type': 'UIEB',
            'augment': True,
            'num_workers': 2,     # Reduced to avoid conflicts
            'pin_memory': True,
        },
        'loss': {
            'reconstruction_weight': 1.0,
            'kl_weight': 0.001,
            'perceptual_weight': 0.1,
            'ssim_weight': 0.1,
            'physics_weight': 0.05,
        },
        'diffusion': {
            'noise_schedule': 'linear',
            'beta_start': 0.0001,
            'beta_end': 0.02,
            'num_train_timesteps': 500,  # Reduced from 1000
            'prediction_type': 'epsilon',
            'ddim_steps': 20,     # Reduced for faster inference
        }
    }


def get_lightweight_config() -> Dict[str, Any]:
    """Get very lightweight configuration for debugging"""
    return {
        'model': {
            'img_size': 64,       # Very small for debugging
            'in_channels': 3,
            'latent_channels': 4,
            'base_channels': 32,  # Very small
            'time_steps': 100,    # Very few steps
            'physics_dim': 16,    # Minimal physics features
            'num_heads': 2,       # Minimal attention heads
            'use_attention': False,  # Disable for speed
            'use_physics': False,    # Disable for debugging
        },
        'training': {
            'num_epochs': 10,
            'batch_size': 2,
            'vae_lr': 1e-3,      # Higher LR for faster convergence
            'diffusion_lr': 1e-3,
            'weight_decay': 0,
            'grad_clip': 0.5,
            'warmup_steps': 100,
            'save_every': 5,
            'eval_every': 2,
            'mixed_precision': False,
        },
        'data': {
            'data_root': './DATA',
            'dataset_type': 'synthetic',  # Use synthetic data
            'augment': False,
            'num_workers': 0,
            'pin_memory': False,
        },
        'loss': {
            'reconstruction_weight': 1.0,
            'kl_weight': 0.01,
            'perceptual_weight': 0.0,     # Disable for speed
            'ssim_weight': 0.0,           # Disable for speed
            'physics_weight': 0.0,        # Disable for speed
        },
        'diffusion': {
            'noise_schedule': 'linear',
            'beta_start': 0.001,
            'beta_end': 0.01,
            'num_train_timesteps': 100,
            'prediction_type': 'epsilon',
            'ddim_steps': 5,              # Very few steps
        }
    }


def get_minimal_config() -> Dict[str, Any]:
    """Get minimal configuration for basic testing"""
    return {
        'model': {
            'type': 'minimal',    # Use MinimalTestModel
            'img_size': 64,
            'in_channels': 3,
            'latent_channels': 4,
            'base_channels': 32,
        },
        'training': {
            'num_epochs': 5,
            'batch_size': 2,
            'lr': 1e-3,
            'weight_decay': 0,
            'grad_clip': 1.0,
        },
        'data': {
            'data_root': './DATA',
            'dataset_type': 'synthetic',
            'num_workers': 0,
        },
        'loss': {
            'reconstruction_weight': 1.0,
            'kl_weight': 0.001,
        }
    }


def get_config(config_name: str = 'safe') -> Dict[str, Any]:
    """
    Get configuration by name
    
    Args:
        config_name: One of 'safe', 'lightweight', 'minimal'
    """
    configs = {
        'safe': get_safe_config,
        'lightweight': get_lightweight_config,
        'minimal': get_minimal_config,
    }
    
    if config_name not in configs:
        print(f"⚠️  Unknown config '{config_name}', using 'safe'")
        config_name = 'safe'
    
    return configs[config_name]()


def save_config(config: Dict[str, Any], filepath: str):
    """Save configuration to YAML file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    with open(filepath, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)
    
    print(f"✅ Configuration saved to {filepath}")


def load_config(filepath: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    if not os.path.exists(filepath):
        print(f"⚠️  Config file {filepath} not found, using safe config")
        return get_safe_config()
    
    with open(filepath, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"✅ Configuration loaded from {filepath}")
    return config


def create_all_configs():
    """Create all configuration files"""
    os.makedirs('configs', exist_ok=True)
    
    configs = {
        'safe': get_safe_config(),
        'lightweight': get_lightweight_config(),
        'minimal': get_minimal_config(),
    }
    
    for name, config in configs.items():
        filepath = f'configs/{name}.yaml'
        save_config(config, filepath)
    
    print(f"✅ Created {len(configs)} configuration files in configs/")


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration for common issues"""
    issues = []
    
    # Check model dimensions
    model = config.get('model', {})
    base_channels = model.get('base_channels', 64)
    physics_dim = model.get('physics_dim', 32)
    
    if base_channels < 32:
        issues.append(f"base_channels ({base_channels}) might be too small")
    
    if physics_dim > base_channels:
        issues.append(f"physics_dim ({physics_dim}) should not exceed base_channels ({base_channels})")
    
    # Check training settings
    training = config.get('training', {})
    batch_size = training.get('batch_size', 4)
    
    if batch_size > 8:
        issues.append(f"batch_size ({batch_size}) might cause memory issues")
    
    # Check diffusion settings
    diffusion = config.get('diffusion', {})
    time_steps = diffusion.get('num_train_timesteps', 500)
    
    if time_steps > 1000:
        issues.append(f"num_train_timesteps ({time_steps}) might be too high for debugging")
    
    if issues:
        print("⚠️  Configuration validation issues:")
        for issue in issues:
            print(f"  - {issue}")
        return False
    else:
        print("✅ Configuration validation passed")
        return True


def get_device_config():
    """Get device-specific configuration recommendations"""
    import torch
    
    device_config = {
        'device': 'cpu',
        'recommended_batch_size': 2,
        'recommended_base_channels': 32,
    }
    
    if torch.cuda.is_available():
        device_config['device'] = 'cuda'
        
        # Get GPU memory
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3  # GB
        
        if gpu_memory >= 16:
            device_config['recommended_batch_size'] = 8
            device_config['recommended_base_channels'] = 128
        elif gpu_memory >= 8:
            device_config['recommended_batch_size'] = 4
            device_config['recommended_base_channels'] = 64
        else:
            device_config['recommended_batch_size'] = 2
            device_config['recommended_base_channels'] = 32
        
        print(f"🖥️  GPU detected: {gpu_memory:.1f}GB VRAM")
    else:
        print("🖥️  No GPU detected, using CPU")
    
    print(f"💡 Recommended settings:")
    print(f"  - batch_size: {device_config['recommended_batch_size']}")
    print(f"  - base_channels: {device_config['recommended_base_channels']}")
    
    return device_config


if __name__ == "__main__":
    print("🔧 CONFIGURATION SETUP TOOL")
    print("=" * 40)
    
    # Get device recommendations
    device_config = get_device_config()
    
    # Create all config files
    create_all_configs()
    
    # Test configuration loading
    print("\n📋 Testing configurations...")
    
    for config_name in ['safe', 'lightweight', 'minimal']:
        print(f"\nTesting {config_name} config:")
        config = get_config(config_name)
        validate_config(config)
    
    print("\n💡 Usage examples:")
    print("  from config_fixed import get_config")
    print("  config = get_config('safe')  # or 'lightweight', 'minimal'")
    print("  ")
    print("  # Load from file")
    print("  config = load_config('configs/safe.yaml')")
