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

# denoising-diffusion-pytorch 라이브러리의 Unet, GaussianDiffusion 클래스를 가져옵니다.
# 훈련 코드에 사용된 클래스 정의가 필요합니다.
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# ==============================================================================
# 1. 훈련 시 사용했던 클래스 정의 (수정 없이 그대로 사용)
# ==============================================================================

class UIEBPairedDataset(Dataset):
    """
    UIEB 페어드 데이터셋: (노이즈 이미지, 깨끗한 이미지) 쌍
    """
    def __init__(self, raw_folder, reference_folder, image_size, split='val', train_ratio=0.8, random_seed=42):
        self.raw_folder = Path(raw_folder)
        self.ref_folder = Path(reference_folder)
        self.image_size = image_size
        self.split = split
        
        self.raw_paths = sorted(list(self.raw_folder.glob("*.jpg")) + 
                               list(self.raw_folder.glob("*.png")))
        self.ref_paths = sorted(list(self.ref_folder.glob("*.jpg")) + 
                               list(self.ref_folder.glob("*.png")))
        
        assert len(self.raw_paths) == len(self.ref_paths), \
            f"Raw images ({len(self.raw_paths)}) and reference images ({len(self.ref_paths)}) count mismatch"
        
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
        
        print(f"Loaded {split.upper()} dataset: {len(self.indices)} images")
        
        # Validation용 Transform
        self.transform = T.Compose([
            T.Resize((image_size, image_size)),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]) # [-1, 1] 범위로 정규화
        ])
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        real_idx = self.indices[idx]
        raw_img = Image.open(self.raw_paths[real_idx]).convert('RGB')
        raw_tensor = self.transform(raw_img)
        ref_img = Image.open(self.ref_paths[real_idx]).convert('RGB')
        ref_tensor = self.transform(ref_img)
        return raw_tensor, ref_tensor

class ConditionalUnet(nn.Module):
    """
    조건부 U-Net: 노이즈 이미지를 조건으로 받는 U-Net
    """
    def __init__(self, dim=64, dim_mults=(1, 2, 4, 8)):
        super().__init__()
        self.condition_encoder = nn.Sequential(
            nn.Conv2d(3, dim, 7, padding=3),
            nn.GroupNorm(8, dim), nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(8, dim), nn.SiLU()
        )
        self.unet = Unet(dim=dim, dim_mults=dim_mults, channels=3 + dim, out_dim=3, flash_attn=False)
        self.random_or_learned_sinusoidal_cond = False
        self.self_condition = False
        self.channels = 3
    
    def forward(self, x, time, condition):
        cond_features = self.condition_encoder(condition)
        x_cond = torch.cat([x, cond_features], dim=1)
        return self.unet(x_cond, time)

class ConditionalGaussianDiffusion(GaussianDiffusion):
    """
    조건부 Gaussian Diffusion
    """
    def __init__(self, model, **kwargs):
        super().__init__(model, **kwargs)
    
    def forward(self, x_start, condition):
        b = x_start.shape[0]
        device = x_start.device
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        pred_noise = self.model(x_noisy, t, condition)
        
        if self.objective == 'pred_noise': target = noise
        elif self.objective == 'pred_x0': target = x_start
        else: raise ValueError(f'unknown objective {self.objective}')
            
        loss = F.mse_loss(pred_noise, target, reduction='mean')
        return loss
    
    @torch.no_grad()
    def sample_conditional(self, condition, batch_size=None):
        if batch_size is None:
            batch_size = condition.shape[0]
            
        device = self.betas.device
        shape = (batch_size, self.channels, *self.image_size)
        img = torch.randn(shape, device=device)

        # DDIM 샘플링을 사용하거나 p_sample_loop를 사용할 수 있습니다.
        # 여기서는 p_sample_loop를 기반으로 간단히 구현합니다.
        for t in reversed(range(0, self.num_timesteps)):
            times = torch.full((batch_size,), t, device=device, dtype=torch.long)
            
            # self.p_sample 로직을 일부 인용하여 단순화
            model_mean, _, model_log_variance, _ = self.p_mean_variance(x=img, t=times, x_self_cond=condition)
            noise = torch.randn_like(img) if t > 0 else 0.
            img = model_mean + (0.5 * model_log_variance).exp() * noise

        # [-1, 1] 범위를 [0, 1]로 변환
        img = self.unnormalize(img)
        return img

    def p_mean_variance(self, x, t, x_self_cond):
        # sample_conditional에서 사용하기 위해 재정의
        # 원래 GaussianDiffusion의 p_mean_variance는 x_self_cond를 받지만,
        # 우리 모델은 condition을 받으므로, 모델 예측 부분을 수정합니다.
        pred_noise = self.model(x, t, x_self_cond) # 여기서 x_self_cond가 우리의 condition입니다.
        
        # 원본 p_mean_variance의 나머지 로직
        x_start = self.predict_start_from_noise(x, t, pred_noise)
        x_start.clamp_(-1., 1.)
        
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance, x_start

# ==============================================================================
# 2. 평가를 위한 헬퍼 함수
# ==============================================================================

def calculate_psnr(img1, img2):
    """
    두 이미지(torch tensor) 간의 PSNR을 계산합니다.
    이미지는 [0, 1] 범위로 가정합니다.
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))

def unnormalize_image(tensor):
    """
    [-1, 1] 범위의 텐서를 [0, 1] 범위로 변환합니다.
    """
    return (tensor + 1.0) / 2.0

# ==============================================================================
# 3. 메인 평가 및 시각화 함수
# ==============================================================================

def evaluate_and_visualize(diffusion_model, val_dataloader, device, num_samples=5):
    """
    모델을 평가하고 결과를 시각화하는 메인 함수
    """
    diffusion_model.eval()
    
    # Validation 데이터셋에서 한 배치 가져오기
    raw_imgs, ref_imgs = next(iter(val_dataloader))
    
    # 지정된 수만큼의 샘플만 선택
    raw_imgs = raw_imgs[:num_samples].to(device)
    ref_imgs = ref_imgs[:num_samples].to(device)
    
    print(f"Running inference on {num_samples} validation images...")
    
    # 조건부 샘플링을 통해 이미지 생성
    with torch.no_grad():
        # sample_conditional은 [0,1] 범위의 텐서를 반환하도록 수정되었습니다.
        generated_imgs = diffusion_model.sample_conditional(raw_imgs)

    # [-1, 1] -> [0, 1]로 정규화 해제
    raw_imgs_unnorm = unnormalize_image(raw_imgs)
    ref_imgs_unnorm = unnormalize_image(ref_imgs)
    # generated_imgs는 이미 [0, 1] 범위
    
    psnr_scores = []
    for i in range(num_samples):
        psnr = calculate_psnr(generated_imgs[i], ref_imgs_unnorm[i])
        psnr_scores.append(psnr)
        print(f"Sample {i+1} PSNR: {psnr:.2f} dB")
        
    avg_psnr = np.mean(psnr_scores)
    print(f"\nAverage PSNR over {num_samples} samples: {avg_psnr:.2f} dB")

    # 시각화를 위해 이미지들을 리스트로 묶기
    all_images = []
    for i in range(num_samples):
        all_images.extend([raw_imgs_unnorm[i].cpu(), generated_imgs[i].cpu(), ref_imgs_unnorm[i].cpu()])
        
    # 이미지 그리드 생성
    grid = make_grid(all_images, nrow=3, padding=5) # nrow=3 -> Raw, Generated, Ref
    
    # 이미지 표시
    plt.figure(figsize=(12, num_samples * 4))
    plt.imshow(grid.permute(1, 2, 0))
    plt.axis('off')
    
    # 제목 추가
    ax = plt.gca()
    ax.text(0.5, 0.98, f"Average PSNR: {avg_psnr:.2f} dB",
            ha='center', va='top', transform=ax.transAxes, fontsize=14, color='white',
            bbox=dict(facecolor='black', alpha=0.5, pad=3))
    
    # 열 제목 추가
    col_titles = ["Raw Image (Input)", "Generated Image (Output)", "Reference Image (GT)"]
    for i, title in enumerate(col_titles):
        x_pos = (i + 0.5) / 3.0
        ax.text(x_pos, 1.01, title, ha='center', va='bottom', transform=ax.transAxes, fontsize=12)
        
    plt.show()
    
    # 결과 이미지 파일로 저장
    save_path = "evaluation_results.png"
    T.ToPILImage()(grid).save(save_path)
    print(f"\nResult grid saved to {save_path}")

# ==============================================================================
# 4. 실행 스크립트
# ==============================================================================

if __name__ == "__main__":
    # --- 설정 ---
    IMAGE_SIZE = 256
    NUM_SAMPLES_TO_EVAL = 5
    MODEL_PATH = './best_model.pth' # 저장된 모델 가중치 경로
    RAW_DATA_FOLDER = './data/UIEB/raw'
    REF_DATA_FOLDER = './data/UIEB/reference'
    
    # --- 장치 설정 ---
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # --- 모델 초기화 ---
    print("Initializing model...")
    model = ConditionalUnet(dim=64, dim_mults=(1, 2, 4, 8))
    
    diffusion = ConditionalGaussianDiffusion(
        model,
        image_size=(IMAGE_SIZE, IMAGE_SIZE), # 튜플로 전달
        timesteps=1000,
        sampling_timesteps=250,
        objective='pred_noise',
        beta_schedule='cosine'
    )
    
    # --- 가중치 로드 ---
    print(f"Loading model weights from {MODEL_PATH}...")
    checkpoint = torch.load(MODEL_PATH, map_location=device)
    diffusion.load_state_dict(checkpoint['model_state_dict'])
    diffusion = diffusion.to(device)
    print(f"Model loaded. Trained for {checkpoint.get('epoch', 'N/A')} epochs with validation loss {checkpoint.get('val_loss', 'N/A'):.4f}")
    
    # --- 데이터셋 및 데이터로더 준비 ---
    print("Preparing validation dataloader...")
    val_dataset = UIEBPairedDataset(
        raw_folder=RAW_DATA_FOLDER,
        reference_folder=REF_DATA_FOLDER,
        image_size=IMAGE_SIZE,
        split='val'
    )
    # 셔플을 False로 하여 항상 동일한 이미지로 평가
    val_dataloader = DataLoader(val_dataset, batch_size=NUM_SAMPLES_TO_EVAL, shuffle=False)
    
    # --- 평가 실행 ---
    evaluate_and_visualize(diffusion, val_dataloader, device, num_samples=NUM_SAMPLES_TO_EVAL)