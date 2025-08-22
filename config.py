"""
Configuration file for Underwater Image Enhancement Diffusion Model
"""

import yaml
import json
from pathlib import Path


def get_default_config():
    """Get default configuration"""
    config = {
        # Experiment settings
        'exp_name': 'underwater_diffusion_v1',
        'exp_dir': 'experiments/underwater_diffusion_v1',
        'device': 'cuda',
        'seed': 42,
        'use_amp': True,  # Mixed precision training
        
        # Model configuration
        'model': {
            'img_size': 256,
            'in_channels': 3,
            'latent_channels': 4,
            'base_channels': 128,
            'time_steps': 1000,
            'physics_dim': 128
        },
        
        # Data configuration
        'data': {
            'data_root': './DATA',
            'dataset_type': 'UIEB',  # 'UIEB' or 'LSUI'
            'img_size': 256,
            'batch_size': 4,  # Reduced from 8 to 4 for memory
            'num_workers': 4,
            'augment': True,
            'preprocessing_type': 'waternet'  # 'all', 'waternet', or 'none'
        },
        
        # Training configuration
        'training': {
            'num_epochs': 1000,
            'vae_lr': 1e-4,
            'diffusion_lr': 1e-4,
            'min_lr': 1e-6,
            'weight_decay': 1e-5,
            'grad_clip': 0.5,  # Reduced from 1.0 to 0.5
            'log_freq': 100,
            'val_freq': 1000,
            'save_freq': 10,
            'max_val_batches': 20  # Reduced from 50 to 20
        },
        
        # Loss configuration
        'loss': {
            'use_adversarial': False,
            'structure_weight': 1.0,
            'underwater_weight': 0.5,
            'adversarial_weight': 0.1
        },
        
        # Logging configuration
        'logging': {
            'use_wandb': False,
            'wandb_project': 'underwater-enhancement',
            'log_dir': 'logs'
        },
        
        # Testing configuration
        'testing': {
            'num_steps': 50,
            'use_ddim': True,
            'eta': 0.0
        }
    }
    
    return config


def get_uieb_config():
    """Configuration for UIEB dataset"""
    config = get_default_config()
    
    config['exp_name'] = 'underwater_diffusion_uieb'
    config['data']['dataset_type'] = 'UIEB'
    config['data']['batch_size'] = 2  # Reduced from 6 to 2 for memory
    config['training']['num_epochs'] = 500
    
    # Memory optimization settings
    config['model']['base_channels'] = 64  # Reduced from 128 to 64
    config['training']['grad_clip'] = 0.5  # Smaller gradient clipping
    
    return config


def get_lsui_config():
    """Configuration for LSUI dataset"""
    config = get_default_config()
    
    config['exp_name'] = 'underwater_diffusion_lsui'
    config['data']['dataset_type'] = 'LSUI'
    config['data']['batch_size'] = 8
    config['training']['num_epochs'] = 300
    
    return config


def get_lightweight_config():
    """Lightweight configuration for faster training/inference"""
    config = get_default_config()
    
    config['exp_name'] = 'underwater_diffusion_light'
    config['model']['base_channels'] = 64
    config['model']['physics_dim'] = 64
    config['model']['time_steps'] = 500
    config['data']['batch_size'] = 12
    config['testing']['num_steps'] = 20
    
    return config


def get_high_quality_config():
    """High quality configuration for best results"""
    config = get_default_config()
    
    config['exp_name'] = 'underwater_diffusion_hq'
    config['model']['img_size'] = 512
    config['model']['base_channels'] = 192
    config['model']['physics_dim'] = 192
    config['data']['img_size'] = 512
    config['data']['batch_size'] = 4
    config['training']['num_epochs'] = 2000
    config['testing']['num_steps'] = 100
    
    return config


def get_config(config_path_or_name=None, config_name=None):
    """
    Load configuration from file or get predefined config
    
    Args:
        config_path_or_name: Path to config file (YAML or JSON) OR predefined config name
        config_name: Name of predefined config (deprecated, use first argument)
    """
    
    # Handle backward compatibility and different calling patterns
    if config_path_or_name is not None:
        # Check if it's a file path or config name
        config_path = Path(config_path_or_name)
        
        # If it looks like a file path and exists
        if config_path.suffix.lower() in ['.yaml', '.yml', '.json'] and config_path.exists():
            # Load from file
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                with open(config_path, 'r') as f:
                    config = json.load(f)
            else:
                raise ValueError(f"Unsupported config file format: {config_path.suffix}")
            
            # Merge with default config to ensure all keys are present
            default_config = get_default_config()
            config = merge_configs(default_config, config)
            
        else:
            # Treat as predefined config name
            config_name = str(config_path_or_name)
            
            if config_name == 'default':
                config = get_default_config()
            elif config_name == 'uieb':
                config = get_uieb_config()
            elif config_name == 'lsui':
                config = get_lsui_config()
            elif config_name == 'lightweight':
                config = get_lightweight_config()
            elif config_name == 'high_quality':
                config = get_high_quality_config()
            else:
                # Try as file path that doesn't exist
                if config_path.suffix.lower() in ['.yaml', '.yml', '.json']:
                    raise FileNotFoundError(f"Config file not found: {config_path}")
                else:
                    raise ValueError(f"Unknown config name: {config_name}")
    
    elif config_name is not None:
        # Legacy: second parameter as config name
        if config_name == 'default':
            config = get_default_config()
        elif config_name == 'uieb':
            config = get_uieb_config()
        elif config_name == 'lsui':
            config = get_lsui_config()
        elif config_name == 'lightweight':
            config = get_lightweight_config()
        elif config_name == 'high_quality':
            config = get_high_quality_config()
        else:
            raise ValueError(f"Unknown config name: {config_name}")
    
    else:
        # Return default config
        config = get_default_config()
    
    # Validate config
    validate_config(config)
    
    return config


def merge_configs(base_config, override_config):
    """Recursively merge two configs"""
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config):
    """Validate configuration"""
    required_keys = [
        'exp_name', 'device', 'model', 'data', 'training'
    ]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    # Validate model config
    model_config = config['model']
    required_model_keys = ['img_size', 'in_channels', 'latent_channels', 'base_channels']
    for key in required_model_keys:
        if key not in model_config:
            raise ValueError(f"Missing required model config key: {key}")
    
    # Validate data config
    data_config = config['data']
    required_data_keys = ['data_root', 'dataset_type', 'batch_size']
    for key in required_data_keys:
        if key not in data_config:
            raise ValueError(f"Missing required data config key: {key}")
    
    # Validate dataset type
    if data_config['dataset_type'] not in ['UIEB', 'LSUI']:
        raise ValueError(f"Invalid dataset_type: {data_config['dataset_type']}")
    
    # Validate image sizes match
    if model_config['img_size'] != data_config['img_size']:
        print(f"Warning: Model img_size ({model_config['img_size']}) != Data img_size ({data_config['img_size']})")
        # Auto-correct
        data_config['img_size'] = model_config['img_size']
    
    # Validate training config
    training_config = config['training']
    required_training_keys = ['num_epochs', 'vae_lr', 'diffusion_lr']
    for key in required_training_keys:
        if key not in training_config:
            raise ValueError(f"Missing required training config key: {key}")


def save_config(config, save_path):
    """Save configuration to file"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    if save_path.suffix.lower() in ['.yaml', '.yml']:
        with open(save_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
    elif save_path.suffix.lower() == '.json':
        with open(save_path, 'w') as f:
            json.dump(config, f, indent=2)
    else:
        raise ValueError(f"Unsupported save format: {save_path.suffix}")
    
    print(f"Config saved to {save_path}")


def create_config_templates():
    """Create configuration template files"""
    templates_dir = Path('configs')
    templates_dir.mkdir(exist_ok=True)
    
    configs = {
        'default.yaml': get_default_config(),
        'uieb.yaml': get_uieb_config(),
        'lsui.yaml': get_lsui_config(),
        'lightweight.yaml': get_lightweight_config(),
        'high_quality.yaml': get_high_quality_config()
    }
    
    for filename, config in configs.items():
        save_config(config, templates_dir / filename)
    
    print(f"Config templates created in {templates_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Configuration utilities')
    parser.add_argument('--create_templates', action='store_true', 
                       help='Create configuration template files')
    parser.add_argument('--validate', type=str, help='Validate a config file')
    parser.add_argument('--show', type=str, help='Show a predefined config')
    
    args = parser.parse_args()
    
    if args.create_templates:
        create_config_templates()
    
    elif args.validate:
        try:
            config = get_config(args.validate)
            print(f"Config {args.validate} is valid")
        except Exception as e:
            print(f"Config validation failed: {e}")
    
    elif args.show:
        try:
            config = get_config(config_name=args.show)
            import pprint
            pprint.pprint(config)
        except Exception as e:
            print(f"Error showing config: {e}")
    
    else:
        print("Use --help to see available options")
        print("Available predefined configs: default, uieb, lsui, lightweight, high_quality")
