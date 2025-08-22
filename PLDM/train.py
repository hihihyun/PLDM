"""
Training script for Underwater Image Enhancement Diffusion Model
"""
import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

from models.main_model import create_model
from data.dataset import create_dataloaders
from config import get_config
from utils import AverageMeter, save_checkpoint, set_seed, save_images

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'] if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")

        set_seed(config['seed'])

        # Directories
        self.exp_dir = Path(config['exp_dir'])
        self.checkpoint_dir = self.exp_dir / 'checkpoints'
        self.sample_dir = self.exp_dir / 'samples'
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.sample_dir.mkdir(parents=True, exist_ok=True)

        # Model, Data, Optimizer
        self.model = create_model(config['model']).to(self.device)
        self.train_loader, self.val_loader = create_dataloaders(config['data'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['training']['lr'])
        self.scaler = torch.cuda.amp.GradScaler(enabled=config['use_amp'])

        self.epoch = 0
        self.global_step = 0

    def train_epoch(self):
        self.model.train()
        meter = AverageMeter()
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch}")

        for batch in pbar:
            degraded = batch['degraded'].to(self.device)
            enhanced = batch['enhanced'].to(self.device)
            preprocessed = batch.get('preprocessed', None)
            if preprocessed is not None:
                preprocessed = preprocessed.to(self.device)

            self.optimizer.zero_grad()

            with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                loss, loss_dict = self.model.training_step(degraded, enhanced, preprocessed)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            meter.update(loss.item())
            pbar.set_postfix(loss=f'{meter.avg:.4f}')
            self.global_step += 1

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        # For simplicity, this is a placeholder. A full validation loop would go here.
        # It would calculate metrics like PSNR, SSIM on the validation set.
        print("Validation step...")
        batch = next(iter(self.val_loader))
        degraded = batch['degraded'][:4].to(self.device)
        enhanced = batch['enhanced'][:4].to(self.device)

        pred_enhanced = self.model.sample(degraded)

        # Denormalize to [0, 1] for saving
        degraded = torch.clamp((degraded + 1.0) / 2.0, 0, 1)
        enhanced = torch.clamp((enhanced + 1.0) / 2.0, 0, 1)
        pred_enhanced = torch.clamp((pred_enhanced + 1.0) / 2.0, 0, 1)

        save_path = self.sample_dir / f'epoch_{self.epoch}.png'
        save_images(degraded, enhanced, pred_enhanced, save_path, normalize=False)
        print(f"Saved sample images to {save_path}")

    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            self.train_epoch()

            if epoch % self.config['training']['val_freq'] == 0:
                self.validate()

            if epoch % self.config['training']['save_freq'] == 0:
                save_checkpoint(self.model.state_dict(), self.checkpoint_dir, f'epoch_{epoch}.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='uieb', help='Name of the config to use (e.g., uieb, lsui)')
    args = parser.parse_args()

    config = get_config(config_name=args.config)
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()