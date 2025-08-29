import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import make_grid
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random

# denoising-diffusion-pytorch 라이브러리의 Unet, GaussianDiffusion 클래스를 가져옵니다.
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# ==============================================================================
# 0. 헬퍼 함수
# ==============================================================================
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# ==============================================================================
# 1. 데이터셋 클래스
# ==============================================================================
class UIEBPairedDataset(Dataset):
    def __init__(self, raw_folder, reference_folder, image_size, split='train', train_ratio=0.8, random_seed=42):
        self.raw_folder = Path(raw_folder)
        self.ref_folder = Path(reference_folder)
        self.image_size = image_size
        
        all_raw_paths = sorted(list(self.raw_folder.glob("*.jpg")) + list(self.raw_folder.glob("*.png")))
        all_ref_paths = sorted(list(self.ref_folder.glob("*.jpg")) + list(self.ref_folder.glob("*.png")))
        
        assert len(all_raw_paths) == len(all_ref_paths), "Raw and reference image counts mismatch"
        
        random.seed(random_seed)
        indices = list(range(len(all_raw_paths)))
        random.shuffle(indices)
        split_idx = int(len(indices) * train_ratio)
        
        self.indices = indices[:split_idx] if split == 'train' else indices[split_idx:]
        self.raw_paths = [all_raw_paths[i] for i in self.indices]
        self.ref_paths = [all_ref_paths[i] for i in self.indices]

        print(f"Loaded {split.upper()} dataset: {len(self.indices)} images")
        
        if split == 'train':
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
        else:
            self.transform = T.Compose([
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        raw_img = Image.open(self.raw_paths[idx]).convert('RGB')
        ref_img = Image.open(self.ref_paths[idx]).convert('RGB')
        return self.transform(raw_img), self.transform(ref_img)

# ==============================================================================
# 2. 모델 및 Diffusion 클래스
# ==============================================================================
class ConditionalUnet(nn.Module):
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(3, dim, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(dim, dim, kernel_size=3, padding=1)
        )
        
        self.unet = Unet(dim=dim, dim_mults=dim_mults, channels=3 + dim, out_dim=3)
        self.channels = 3
        self.self_condition = False
        
    def forward(self, x, time, condition):
        cond_features = self.condition_encoder(condition)
        x_cond = torch.cat([x, cond_features], dim=1)
        return self.unet(x_cond, time)

class ConditionalGaussianDiffusion(GaussianDiffusion):
    def p_losses(self, x_start, t, *, condition, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = self.model(x_noisy, t, condition)
        
        if self.objective != 'pred_noise':
             raise ValueError('Objective must be pred_noise for this implementation')
        return F.mse_loss(pred_noise, noise)

    def forward(self, clean_imgs, raw_imgs):
        b = clean_imgs.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=clean_imgs.device).long()
        return self.p_losses(clean_imgs, t, condition=raw_imgs)

    @torch.no_grad()
    def sample_conditional(self, condition, start_timestep=None):
        batch_size = condition.shape[0]
        device = self.betas.device
        start_timestep = default(start_timestep, self.num_timesteps - 1)
        
        # 입력 이미지에 노이즈를 추가하여 샘플링 시작점으로 사용
        t = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
        img = self.q_sample(x_start=condition, t=t)

        for i in tqdm(reversed(range(0, start_timestep)), desc='Sampling', leave=False):
            times = torch.full((batch_size,), i, device=device, dtype=torch.long)
            pred_noise = self.model(img, times, condition)
            
            x_start = self.predict_start_from_noise(img, times, pred_noise)
            if hasattr(self, 'clip_denoised') and self.clip_denoised:
                 x_start.clamp_(-1., 1.)

            model_mean, _, _ = self.q_posterior(x_start=x_start, x_t=img, t=times)
            
            noise = torch.randn_like(img) if i > 0 else 0.
            
            posterior_variance = self._get_variance(times)
            img = model_mean + torch.sqrt(posterior_variance) * noise
            
        return self.unnormalize(img)
    
    def _get_variance(self, t):
        # DDPM의 posterior variance를 사용
        alpha_t = self.alphas_cumprod[t]
        alpha_t_prev = self.alphas_cumprod[t-1] if t[0] > 0 else torch.tensor(1.0)
        
        # DDPM 논문의 Equation 7
        posterior_variance = self.betas[t] * (1. - alpha_t_prev) / (1. - alpha_t)
        return posterior_variance.reshape(-1, 1, 1, 1)

# ==============================================================================
# 3. 평가 및 시각화 헬퍼 함수
# ==============================================================================
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse == 0: return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def unnormalize_image(tensor):
    return (tensor.clamp(-1., 1.) + 1.0) / 2.0

def visualize_and_evaluate(diffusion_model, dataloader, device, epoch, save_folder="results", num_samples=2, start_timestep=500):
    diffusion_model.eval()
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True)
    
    raw_imgs, ref_imgs = next(iter(dataloader))
    raw_imgs = raw_imgs[:num_samples].to(device)
    ref_imgs = ref_imgs[:num_samples].to(device)

    with torch.no_grad():
        generated_imgs = diffusion_model.sample_conditional(raw_imgs, start_timestep=start_timestep)

    raw_unnorm = unnormalize_image(raw_imgs)
    ref_unnorm = unnormalize_image(ref_imgs)
    
    psnr_scores = [calculate_psnr(gen, ref) for gen, ref in zip(generated_imgs, ref_unnorm)]
    avg_psnr = np.mean(psnr_scores)
    
    print(f"\nEpoch {epoch} Evaluation - Average PSNR: {avg_psnr:.2f} dB")

    all_images = [img.cpu() for pair in zip(raw_unnorm, generated_imgs, ref_unnorm) for img in pair]
    grid = make_grid(all_images, nrow=3, padding=5)
    
    save_path = save_folder / f"epoch_{epoch}_psnr_{avg_psnr:.2f}.png"
    T.ToPILImage()(grid).save(save_path)
    print(f"Evaluation image saved to: {save_path}")

def plot_loss_history(train_losses, val_losses, save_folder="results"):
    save_folder = Path(save_folder)
    save_folder.mkdir(exist_ok=True)
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    save_path = save_folder / "loss_history.png"
    plt.savefig(save_path)
    print(f"Loss history plot saved to: {save_path}")
    plt.close()

# ==============================================================================
# 4. 메인 훈련 스크립트
# ==============================================================================
def main():
    # --- 하이퍼파라미터 설정 ---
    IMAGE_SIZE = 512
    BATCH_SIZE = 2
    GRAD_ACCUM_EVERY = 8
    NUM_EPOCHS = 50
    LEARNING_RATE = 1e-4
    EVAL_START_TIMESTEP = 500 # 샘플링 시작 타임스텝

    # --- 경로 설정 ---
    RAW_DATA_FOLDER = './data/UIEB/raw'
    REF_DATA_FOLDER = './data/UIEB/reference'

    # --- 장치 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 데이터셋 및 데이터로더 ---
    train_dataset = UIEBPairedDataset(RAW_DATA_FOLDER, REF_DATA_FOLDER, IMAGE_SIZE, split='train')
    val_dataset = UIEBPairedDataset(RAW_DATA_FOLDER, REF_DATA_FOLDER, IMAGE_SIZE, split='val')
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    # --- 모델 및 Diffusion 설정 ---
    model = ConditionalUnet(dim=64, dim_mults=(1, 2, 4, 8))
    diffusion = ConditionalGaussianDiffusion(
        model, image_size=(IMAGE_SIZE, IMAGE_SIZE), timesteps=1000,
        objective='pred_noise', beta_schedule='cosine'
    ).to(device)
    
    diffusion.clip_denoised = True

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=LEARNING_RATE)

    # --- 훈련 루프 ---
    best_val_loss = float('inf')
    train_loss_history, val_loss_history = [], []

    for epoch in range(NUM_EPOCHS):
        diffusion.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}", leave=True)
        for i, (raw_imgs, clean_imgs) in enumerate(progress_bar):
            raw_imgs, clean_imgs = raw_imgs.to(device), clean_imgs.to(device)
            
            with torch.cuda.amp.autocast():
                loss = diffusion(clean_imgs, raw_imgs)
                loss = loss / GRAD_ACCUM_EVERY
            
            loss.backward()
            total_train_loss += loss.item()

            if (i + 1) % GRAD_ACCUM_EVERY == 0 or (i + 1) == len(train_dataloader):
                optimizer.step()
                optimizer.zero_grad()
            
            progress_bar.set_postfix(loss=loss.item() * GRAD_ACCUM_EVERY)

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_loss_history.append(avg_train_loss)

        # --- 검증 단계 ---
        diffusion.eval()
        total_val_loss = 0
        with torch.no_grad():
            for raw_imgs, clean_imgs in val_dataloader:
                raw_imgs, clean_imgs = raw_imgs.to(device), clean_imgs.to(device)
                with torch.cuda.amp.autocast():
                    val_loss = diffusion(clean_imgs, raw_imgs)
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_loss_history.append(avg_val_loss)

        print(f'Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(diffusion.state_dict(), './best_model_512.pth')
            print(f'✓ Best model saved with val loss: {best_val_loss:.4f}')

        # --- 10 에포크마다 평가 및 시각화 ---
        if (epoch + 1) % 10 == 0:
            visualize_and_evaluate(diffusion, val_dataloader, device, epoch + 1, start_timestep=EVAL_START_TIMESTEP)

    print("\nTraining finished!")
    plot_loss_history(train_loss_history, val_loss_history)

if __name__ == "__main__":
    main()