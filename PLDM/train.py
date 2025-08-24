"""
Training script for Underwater Image Enhancement Diffusion Model
"""
# --- ▼ 경로 문제 해결 코드 (기존과 동일) ▼ ---
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
# --- ▲ 경로 문제 해결 코드 끝 ▲ ---

import torch
from tqdm import tqdm
import argparse

from PLDM.models.main_model import create_model 
from PLDM.data.dataset import create_dataloaders
from PLDM.config import get_config
from PLDM.utils import AverageMeter, save_checkpoint, set_seed, save_images

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

        # 👇 [수정 완료] 모델을 생성할 때 device 정보를 명시적으로 전달합니다.
        self.model = create_model(config['model'], device=self.device).to(self.device)
        
        self.train_loader, self.val_loader = create_dataloaders(config['data'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config['training']['lr'])
        self.scaler = torch.cuda.amp.GradScaler(enabled=config.get('use_amp', False))

        self.epoch = 0
        self.best_val_loss = float('inf')

    # ... (이하 코드는 이전과 동일합니다) ...
    def train_epoch(self):
        self.model.train()
        meter = AverageMeter()
        pbar = tqdm(self.train_loader, desc=f"Epoch {self.epoch + 1}/{self.config['training']['num_epochs']}")

        for batch in pbar:
            degraded = batch['degraded'].to(self.device)
            enhanced = batch['enhanced'].to(self.device)
            preprocessed = batch.get('preprocessed', None)
            if preprocessed is not None:
                preprocessed = preprocessed.to(self.device)

            self.optimizer.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=self.config.get('use_amp', False)):
                loss, loss_dict = self.model.training_step(degraded, enhanced, preprocessed)

            if torch.isinf(loss) or torch.isnan(loss):
                print(f"\n🔴 Skipping batch due to invalid loss: {loss.item()}. Check model stability.")
                continue
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            meter.update(loss.item(), degraded.size(0))
            pbar.set_postfix(loss=f'{meter.avg:.4f}')

    @torch.no_grad()
    def validate(self):
        self.model.eval()
        val_meter = AverageMeter()
        print("\nConducting validation...")
        
        for batch in tqdm(self.val_loader, desc="Validation"):
            degraded = batch['degraded'].to(self.device)
            enhanced = batch['enhanced'].to(self.device)
            preprocessed = batch.get('preprocessed', None)
            if preprocessed is not None:
                preprocessed = preprocessed.to(self.device)
            
            loss, _ = self.model.training_step(degraded, enhanced, preprocessed)
            if not (torch.isinf(loss) or torch.isnan(loss)):
                val_meter.update(loss.item(), degraded.size(0))

        print(f"Validation Loss: {val_meter.avg:.4f}")

        if val_meter.avg < self.best_val_loss:
            self.best_val_loss = val_meter.avg
            print(f"🎉 New best model found with validation loss: {self.best_val_loss:.4f}")
            save_checkpoint(self.model.state_dict(), self.checkpoint_dir, 'best_model.pth')

        # Save sample images
        batch = next(iter(self.val_loader))
        degraded_sample = batch['degraded'][:4].to(self.device)
        enhanced_gt_sample = batch['enhanced'][:4].to(self.device)
        pred_enhanced = self.model.sample(degraded_sample)
        
        save_path = self.sample_dir / f'epoch_{self.epoch + 1}.png'
        save_images(degraded_sample, enhanced_gt_sample, pred_enhanced, save_path)
        print(f"Saved sample images to {save_path}")

    def train(self):
        for epoch in range(self.config['training']['num_epochs']):
            self.epoch = epoch
            self.train_epoch()

            if (epoch + 1) % self.config['training']['val_freq'] == 0:
                self.validate()

            if (epoch + 1) % self.config['training']['save_freq'] == 0:
                save_checkpoint(self.model.state_dict(), self.checkpoint_dir, f'epoch_{epoch+1}.pth')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='uieb', help='Name of the config to use (e.g., uieb, lsui)')
    args = parser.parse_args()

    config = get_config(config_name=args.config)
    trainer = Trainer(config)
    trainer.train()

if __name__ == "__main__":
    main()