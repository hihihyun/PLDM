import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

from denoising_diffusion_pytorch import Unet, GaussianDiffusion

class UIEBPairedDataset(Dataset):
    """
    UIEB 페어드 데이터셋: (노이즈 이미지, 깨끗한 이미지) 쌍
    """
    def __init__(self, raw_folder, reference_folder, image_size, split='train', train_ratio=0.8, random_seed=42):
        self.raw_folder = Path(raw_folder)
        self.ref_folder = Path(reference_folder)
        self.image_size = image_size
        self.split = split
        
        # 이미지 파일들 찾기
        self.raw_paths = sorted(list(self.raw_folder.glob("*.jpg")) + 
                               list(self.raw_folder.glob("*.png")))
        self.ref_paths = sorted(list(self.ref_folder.glob("*.jpg")) + 
                               list(self.ref_folder.glob("*.png")))
        
        assert len(self.raw_paths) == len(self.ref_paths), \
            f"Raw images ({len(self.raw_paths)}) and reference images ({len(self.ref_paths)}) count mismatch"
        
        # Train/Validation 분할
        import random
        random.seed(random_seed)
        indices = list(range(len(self.raw_paths)))
        random.shuffle(indices)
        
        split_idx = int(len(indices) * train_ratio)
        if split == 'train':
            self.indices = indices[:split_idx]
        elif split == 'val':
            self.indices = indices[split_idx:]
        else:
            raise ValueError("Split must be 'train' or 'val'")
        
        print(f"{split.upper()} dataset: {len(self.indices)} images")
        
        # Transform 설정 (train은 augmentation 추가)
        if split == 'train':
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),          # 좌우 플립
                T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),  # 색상 변화
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] 범위로 정규화
            ])
        else:  # validation
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        
        # 노이즈 이미지 (조건)
        raw_img = Image.open(self.raw_paths[real_idx]).convert('RGB')
        raw_tensor = self.transform(raw_img)
        
        # 깨끗한 이미지 (타겟)
        ref_img = Image.open(self.ref_paths[real_idx]).convert('RGB')
        ref_tensor = self.transform(ref_img)
        
        return raw_tensor, ref_tensor

class ConditionalUnet(nn.Module):
    """
    조건부 U-Net: 노이즈 이미지를 조건으로 받는 U-Net
    """
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        
        # 조건 이미지 (raw image)를 처리하는 인코더
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(3, dim, 7, padding=3),
            nn.GroupNorm(8, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim),
            nn.SiLU()
        )
        
        # 메인 U-Net (입력 채널 = 3 + dim, 노이즈+조건 결합)
        self.unet = Unet(
            dim=dim,
            dim_mults=dim_mults,
            channels=3 + dim,  # RGB + condition features
            out_dim=3,         # RGB 출력
            flash_attn=False   # Flash Attention 비활성화
        )
        
        # 필요한 속성들을 ConditionalUnet에도 추가
        self.random_or_learned_sinusoidal_cond = False
        self.self_condition = False
        self.channels = 3
    
    def forward(self, x, time, condition):
        """
        x: [B, 3, H, W] 노이즈가 있는 타겟 이미지
        time: [B] 타임스텝
        condition: [B, 3, H, W] 조건 이미지 (raw 이미지)
        """
        # 조건 이미지를 특성으로 변환
        cond_features = self.condition_encoder(condition)
        
        # 노이즈 이미지와 조건 특성을 결합
        x_cond = torch.cat([x, cond_features], dim=1)
        
        # U-Net으로 처리
        return self.unet(x_cond, time)

class ConditionalGaussianDiffusion(GaussianDiffusion):
    """
    조건부 Gaussian Diffusion
    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
    
    def forward(self, x_start, condition):
        """
        훈련용 forward pass
        """
        b = x_start.shape[0]
        device = x_start.device
        
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        
        # 노이즈 추가
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # 모델 예측: 조건을 사용해서 노이즈 예측
        pred_noise = self.model(x_noisy, t, condition)
        
        # Loss 계산
        if self.objective == 'pred_noise':
            target = noise
        elif self.objective == 'pred_x0':
            target = x_start
        else:
            raise ValueError(f'unknown objective {self.objective}')
            
        loss = F.mse_loss(pred_noise, target, reduction='mean')
        return loss
    
    @torch.no_grad()
    def sample_conditional(self, condition, batch_size=None):
        """
        조건부 샘플링: raw 이미지가 주어졌을 때 깨끗한 이미지 생성
        """
        if batch_size is None:
            batch_size = condition.shape[0]
            
        device = condition.device
        image_size = self.image_size
        
        # image_size가 튜플인 경우 정수로 변환
        if isinstance(image_size, (tuple, list)):
            h, w = image_size
        else:
            h = w = image_size
        
        # 랜덤 노이즈에서 시작
        img = torch.randn(batch_size, 3, h, w, device=device)
        
        # 디노이징 과정
        for t in reversed(range(0, self.num_timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # 모델 예측
            pred_noise = self.model(img, t_tensor, condition)
            
            # 이전 스텝으로 디노이징 (간단한 DDPM 방식)
            if self.objective == 'pred_noise':
                alpha_t = self.alphas_cumprod[t]
                alpha_t_prev = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0)
                
                # x_0 예측
                x_0_pred = (img - torch.sqrt(1 - alpha_t) * pred_noise) / torch.sqrt(alpha_t)
                x_0_pred = torch.clamp(x_0_pred, -1, 1)
                
                # x_{t-1} 계산
                if t > 0:
                    noise = torch.randn_like(img)
                    img = torch.sqrt(alpha_t_prev) * x_0_pred + torch.sqrt(1 - alpha_t_prev) * noise
                else:
                    img = x_0_pred
        
        return img

def main():
    # 1. 데이터셋 설정 (Train/Val 분할)
    train_dataset = UIEBPairedDataset(
        raw_folder='./data/UIEB/raw',           # 노이즈 있는 수중 이미지
        reference_folder='./data/UIEB/reference', # 깨끗한 수중 이미지
        image_size=256,      # 원래대로 256 유지
        split='train'
    )
    
    val_dataset = UIEBPairedDataset(
        raw_folder='./data/UIEB/raw',
        reference_folder='./data/UIEB/reference',
        image_size=256,      # 원래대로 256 유지
        split='val'
    )
    
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, pin_memory=True)
    
    # 2. 조건부 모델 생성
    model = ConditionalUnet(dim=64, dim_mults=(1, 2, 4, 8))
    
    # 3. 조건부 Diffusion 모델
    diffusion = ConditionalGaussianDiffusion(
        model,
        image_size=256,      # 원래대로 256 유지
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_noise',
        beta_schedule='cosine'
    )
    
    # 4. 훈련 설정 (Gradient Accumulation 추가)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    diffusion = diffusion.to(device)
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=1e-4)
    
    # Gradient Accumulation 설정
    gradient_accumulate_every = 8  # 2 * 8 = effective batch size 16
    
    print(f"Training on {len(train_dataset)} images, Validation on {len(val_dataset)} images")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Effective batch size: {2 * gradient_accumulate_every}")
    
    # 5. 훈련 루프 (Gradient Accumulation 포함)
    num_epochs = 100
    best_val_loss = float('inf')
    
    for epoch in range(num_epochs):
        # ============ 훈련 단계 ============
        diffusion.train()
        total_train_loss = 0
        
        for batch_idx, (raw_imgs, clean_imgs) in enumerate(train_dataloader):
            raw_imgs = raw_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            
            # Forward pass
            loss = diffusion(clean_imgs, raw_imgs)  # (타겟, 조건)
            loss = loss / gradient_accumulate_every  # 스케일링
            
            # Backward pass
            loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % gradient_accumulate_every == 0 or (batch_idx + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            total_train_loss += loss.item() * gradient_accumulate_every  # 원래 크기로 복원
            
            if batch_idx % 50 == 0:
                print(f'Epoch {epoch}, Batch {batch_idx}/{len(train_dataloader)}, Loss: {loss.item() * gradient_accumulate_every:.4f}')
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        
        # ============ 검증 단계 ============
        diffusion.eval()
        total_val_loss = 0
        with torch.no_grad():
            for raw_imgs, clean_imgs in val_dataloader:
                raw_imgs = raw_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                
                val_loss = diffusion(clean_imgs, raw_imgs)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        
        print(f'Epoch {epoch} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        # Best model 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': diffusion.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, './best_model.pth')
            print(f'✓ Best model saved at epoch {epoch} with val loss: {best_val_loss:.4f}')
        
        # 주기적으로 샘플 생성
        if epoch % 10 == 0:
            with torch.no_grad():
                # Validation set에서 첫 번째 배치로 테스트
                raw_test, clean_test = next(iter(val_dataloader))
                raw_test = raw_test[:4].to(device)  # 4개만 테스트
                clean_test = clean_test[:4].to(device)
                
                # 깨끗한 이미지 생성
                generated = diffusion.sample_conditional(raw_test)
                
                print(f"Generated samples shape: {generated.shape}")
                # TODO: 이미지 저장 로직 추가 (raw vs generated vs ground_truth 비교)

if __name__ == "__main__":
    main()