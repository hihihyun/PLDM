"""
Memory-efficient training script for limited GPU memory
"""

import torch
import argparse
from config import get_config
from train import Trainer

def get_memory_efficient_config():
    """Get memory-efficient configuration"""
    config = get_config('lightweight')
    
    # Further memory optimizations
    config['model']['base_channels'] = 32  # Very small model
    config['model']['physics_dim'] = 32
    config['data']['batch_size'] = 1  # Minimum batch size
    config['training']['max_val_batches'] = 5
    config['use_amp'] = True
    
    # Faster training for testing
    config['training']['num_epochs'] = 100
    config['training']['val_freq'] = 200
    config['training']['log_freq'] = 50
    
    return config

def main():
    parser = argparse.ArgumentParser(description='Memory-efficient training')
    parser.add_argument('--exp_name', type=str, default='memory_efficient_test')
    
    args = parser.parse_args()
    
    print("🧠 MEMORY-EFFICIENT UNDERWATER ENHANCEMENT TRAINING")
    print("=" * 60)
    
    # Get memory-efficient config
    config = get_memory_efficient_config()
    config['exp_name'] = args.exp_name
    config['exp_dir'] = f"experiments/{args.exp_name}"
    
    print("💾 Memory optimization settings:")
    print(f"  - Base channels: {config['model']['base_channels']}")
    print(f"  - Batch size: {config['data']['batch_size']}")
    print(f"  - Mixed precision: {config['use_amp']}")
    print(f"  - Max val batches: {config['training']['max_val_batches']}")
    
    # Clear cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print(f"🚀 GPU memory before training: {torch.cuda.memory_allocated()/1e9:.1f}GB")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Start training
    try:
        trainer.train()
    except torch.cuda.OutOfMemoryError as e:
        print(f"❌ Still out of memory: {e}")
        print("💡 Try:")
        print("  1. Reduce batch size to 1")
        print("  2. Use CPU: --device cpu")
        print("  3. Reduce image size to 128")
    except KeyboardInterrupt:
        print("Training interrupted. Saving checkpoint...")
        trainer.save_checkpoint()

if __name__ == "__main__":
    main()
