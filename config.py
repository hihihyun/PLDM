"""
Configuration file for Underwater Image Enhancement Diffusion Model
"""
def get_config(config_name='uieb'):
    if config_name == 'uieb':
        return get_uieb_config()
    elif config_name == 'lsui':
        return get_lsui_config()
    else:
        raise ValueError(f"Unknown config name: {config_name}")

def get_base_config():
    """Base configuration"""
    return {
        'exp_name': 'underwater_diffusion_v1',
        'exp_dir': 'experiments/underwater_diffusion_v1',
        'device': 'cuda',
        'seed': 42,
        'use_amp': True,

        'model': {
            'img_size': 256,
            'in_channels': 3,
            'latent_channels': 4,
            'base_channels': 128,
            'time_steps': 1000,
            'physics_dim': 128
        },

        'data': {
            'data_root': './DATA',
            'img_size': 256,
            'batch_size': 4,
            'num_workers': 4,
            'augment': True,
            'preprocessing_type': 'waternet'
        },

        'training': {
            'num_epochs': 500,
            'lr': 1e-4,
            'val_freq': 10,
            'save_freq': 10,
        },
    }

def get_uieb_config():
    """Configuration for UIEB dataset"""
    config = get_base_config()
    config['exp_name'] = 'underwater_diffusion_uieb'
    config['exp_dir'] = f"experiments/{config['exp_name']}"
    config['data']['dataset_type'] = 'UIEB'
    config['data']['batch_size'] = 4
    return config

def get_lsui_config():
    """Configuration for LSUI dataset"""
    config = get_base_config()
    config['exp_name'] = 'underwater_diffusion_lsui'
    config['exp_dir'] = f"experiments/{config['exp_name']}"
    config['data']['dataset_type'] = 'LSUI'
    config['data']['batch_size'] = 2 # LSUI is larger, so smaller batch size
    return config