# 필요한 라이브러리들을 임포트합니다.
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
# VGGPerceptualLoss를 위해 torchvision에서 vgg16과 새로운 가중치 API를 가져옵니다.
from torchvision.models import vgg16, VGG16_Weights
from torchvision.utils import make_grid
from pathlib import Path
import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import random
import argparse

# 우리가 사용하는 Diffusion 모델 라이브러리를 임포트합니다.
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# ==============================================================================
# 0. 헬퍼 함수 (Helper Functions)
# ==============================================================================
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

# ==============================================================================
# 1. 데이터셋 클래스 (Dataset Class)
# ==============================================================================
class UIEBPairedDataset(Dataset):
    def __init__(self, raw_folder, reference_folder, image_size, split='train', train_ratio=0.9, random_seed=42):
        self.raw_folder = Path(raw_folder)
        self.ref_folder = Path(reference_folder)
        self.image_size = image_size
        all_raw_paths = sorted(list(self.raw_folder.glob("*.jpg")) + list(self.raw_folder.glob("*.png")))
        all_ref_paths = sorted(list(self.ref_folder.glob("*.png")) + list(self.ref_folder.glob("*.jpg")))
        assert len(all_raw_paths) == len(all_ref_paths), "Raw 이미지와 Reference 이미지의 개수가 일치하지 않습니다."
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
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
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
# 2. 손실 함수 정의 (Loss Function)
# ==============================================================================
class VGGPerceptualLoss(nn.Module):
    def __init__(self, resize=True):
        super().__init__()
        vgg_model = vgg16(weights=VGG16_Weights.DEFAULT)
        blocks = [vgg_model.features[:4].eval(),
                  vgg_model.features[4:9].eval(),
                  vgg_model.features[9:16].eval(),
                  vgg_model.features[16:23].eval()]
        for bl in blocks:
            for p in bl.parameters():
                p.requires_grad = False
        self.blocks = nn.ModuleList(blocks)
        self.transform = nn.functional.interpolate
        self.resize = resize
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    def forward(self, input, target):
        input = (input * 0.5 + 0.5 - self.mean) / self.std
        target = (target * 0.5 + 0.5 - self.mean) / self.std
        if self.resize:
            input = self.transform(input, mode='bilinear', size=(224, 224), align_corners=False)
            target = self.transform(target, mode='bilinear', size=(224, 224), align_corners=False)
        loss = 0.0
        x, y = input, target
        for block in self.blocks:
            x, y = block(x), block(y)
            loss += nn.functional.l1_loss(x, y)
        return loss

# ==============================================================================
# 3. 모델 및 Diffusion 클래스
# ==============================================================================
class ConditionalUnet(Unet):
    def __init__(self, *args, **kwargs):
        kwargs.pop('channels', None)
        super().__init__(*args, channels=6, out_dim=3, **kwargs)
    def forward(self, x, time, condition):
        x_cond = torch.cat([x, condition], dim=1)
        return super().forward(x_cond, time, x_self_cond=None)

class ConditionalGaussianDiffusion(GaussianDiffusion):
    def p_losses(self, x_start, t, *, condition, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = self.model(x_noisy, t, condition=condition)
        denoised_img = self.predict_start_from_noise(x_noisy, t, pred_noise)
        pixel_loss = F.l1_loss(denoised_img, x_start)
        return pixel_loss, denoised_img

    def forward(self, clean_imgs, raw_imgs):
        b = clean_imgs.shape[0]
        t = torch.randint(0, self.num_timesteps, (b,), device=clean_imgs.device).long()
        return self.p_losses(clean_imgs, t, condition=raw_imgs)

    @torch.no_grad()
    def sample_conditional(self, condition, start_timestep=None):
        batch_size = condition.shape[0]
        device = self.betas.device
        start_timestep = default(start_timestep, self.num_timesteps - 1)
        t = torch.full((batch_size,), start_timestep, device=device, dtype=torch.long)
        img = self.q_sample(x_start=condition, t=t)
        for i in tqdm(reversed(range(0, start_timestep)), desc='Sampling', total=start_timestep, leave=False):
            times = torch.full((batch_size,), i, device=device, dtype=torch.long)
            img, _ = self.p_sample(img, times, x_self_cond=condition)
        return self.unnormalize(img)

    def p_sample(self, x, t, x_self_cond=None):
        b, *_, device = *x.shape, x.device
        batched_times = t if isinstance(t, torch.Tensor) else torch.full((b,), t, device=x.device, dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(x=x, t=batched_times, x_self_cond=x_self_cond)
        noise = torch.randn_like(x) if (isinstance(t, int) and t > 0) or (isinstance(t, torch.Tensor) and t[0] > 0) else 0.
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    def p_mean_variance(self, x, t, x_self_cond):
        pred_noise = self.model(x, t, condition=x_self_cond)
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        if hasattr(self, 'clip_denoised') and self.clip_denoised:
             x_start.clamp_(-1., 1.)
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

# ==============================================================================
# 4. 평가 및 시각화 헬퍼 함수
# ==============================================================================
def calculate_psnr(img1, img2):
    mse = F.mse_loss(img1, img2)
    if mse.item() == 0: return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse.item()))

def unnormalize_image(tensor):
    return (tensor.clamp(-1., 1.) + 1.0) / 2.0

def visualize_and_evaluate(diffusion_model, dataloader, device, epoch, save_folder="results", num_samples=4, start_timestep=500):
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
    print(f"\nEpoch {epoch} Evaluation (T={start_timestep}) - Average PSNR: {avg_psnr:.2f} dB")
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
    plt.savefig(save_folder / "loss_history.png")
    plt.close()

# ==============================================================================
# 5. 메인 훈련 스크립트
# ==============================================================================
def main(args):
    # --- 하이퍼파라미터 ---
    IMAGE_SIZE = args.image_size
    BATCH_SIZE = args.batch_size
    GRAD_ACCUM_EVERY = 16 // BATCH_SIZE
    NUM_EPOCHS = 500
    # ❗️수정 1: 학습률을 낮춰 훈련 안정성을 높입니다.
    LEARNING_RATE = 2e-5
    EVAL_START_TIMESTEP = 500
    MODEL_SAVE_PATH = f'./best_model_perceptual_{IMAGE_SIZE}.pth'
    
    lambda_pixel = 1.0
    lambda_perceptual = 0.5

    # --- 경로 및 장치 설정 ---
    RAW_DATA_FOLDER = './data/UIEB/raw'
    REF_DATA_FOLDER = './data/UIEB/reference'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 데이터셋 및 데이터로더 ---
    train_dataset = UIEBPairedDataset(RAW_DATA_FOLDER, REF_DATA_FOLDER, IMAGE_SIZE, split='train', train_ratio=0.9)
    val_dataset = UIEBPairedDataset(RAW_DATA_FOLDER, REF_DATA_FOLDER, IMAGE_SIZE, split='val', train_ratio=0.9)
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=4)
    val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=4)

    # --- 모델, 손실 함수, 옵티마이저 ---
    model = ConditionalUnet(dim=128, dim_mults=(1, 2, 4, 8), dropout=0.1)
    diffusion = ConditionalGaussianDiffusion(
        model, image_size=(IMAGE_SIZE, IMAGE_SIZE), timesteps=1000,
        objective='pred_noise', beta_schedule='cosine'
    ).to(device)
    
    perceptual_loss_fn = VGGPerceptualLoss().to(device)

    if args.load_checkpoint:
        print(f"Loading checkpoint from: {args.load_checkpoint}")
        diffusion.load_state_dict(torch.load(args.load_checkpoint, map_location=device))

    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)

    # --- 훈련 루프 ---
    best_val_loss = float('inf')
    train_loss_history, val_loss_history = [], []

    for epoch in range(1, NUM_EPOCHS + 1):
        diffusion.train()
        total_train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}/{NUM_EPOCHS}")
        for i, (raw_imgs, clean_imgs) in enumerate(progress_bar):
            raw_imgs, clean_imgs = raw_imgs.to(device), clean_imgs.to(device)
            
            optimizer.zero_grad()
            
            # ❗️`torch.amp.autocast`를 사용하여 최신 방식으로 AMP를 적용합니다.
            with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                pixel_loss, generated_imgs = diffusion(clean_imgs, raw_imgs)
                loss_perceptual = perceptual_loss_fn(generated_imgs, clean_imgs)
                total_loss = lambda_pixel * pixel_loss + lambda_perceptual * loss_perceptual
            
            total_loss.backward()
            
            # ❗️수정 2: 그래디언트 클리핑을 추가하여 그래디언트 폭주를 방지합니다.
            torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
            
            optimizer.step()

            total_train_loss += total_loss.item()
            progress_bar.set_postfix(
                Total_Loss=total_loss.item(), Pixel_L1=pixel_loss.item()
            )

        avg_train_loss = total_train_loss / len(train_dataloader)
        train_loss_history.append(avg_train_loss)
        
        scheduler.step()
        
        # --- 검증 단계 ---
        diffusion.eval()
        total_val_loss = 0
        with torch.no_grad():
            for raw_imgs, clean_imgs in val_dataloader:
                raw_imgs, clean_imgs = raw_imgs.to(device), clean_imgs.to(device)
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    pixel_loss, generated_imgs = diffusion(clean_imgs, raw_imgs)
                    loss_perceptual = perceptual_loss_fn(generated_imgs, clean_imgs)
                val_loss = lambda_pixel * pixel_loss + lambda_perceptual * loss_perceptual
                total_val_loss += val_loss.item()
        
        avg_val_loss = total_val_loss / len(val_dataloader)
        val_loss_history.append(avg_val_loss)

        print(f'Epoch {epoch}/{NUM_EPOCHS} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(diffusion.state_dict(), MODEL_SAVE_PATH)
            print(f'✓ Best model saved to {MODEL_SAVE_PATH} with val loss: {best_val_loss:.4f}')

        if epoch == 1 or epoch % 10 == 0:
            visualize_and_evaluate(diffusion, val_dataloader, device, epoch, start_timestep=EVAL_START_TIMESTEP)

    print("\nTraining finished!")
    plot_loss_history(train_loss_history, val_loss_history)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_size', type=int, default=256, help='image size of the model')
    parser.add_argument('--batch_size', type=int, default=4, help='batch size for training')
    parser.add_argument('--load_checkpoint', type=str, default=None, help='path to checkpoint to load')
    args = parser.parse_args()
    main(args)