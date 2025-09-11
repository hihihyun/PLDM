import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

device = "cuda" if torch.cuda.is_available() else "cpu"

print(f"사용 장치: {device}")

# 1. U-Net 모델 설정
# 주석: Unet에 conditional=True 인자를 전달합니다.
model = Unet(
    dim = 128,
    dim_mults = (1, 2, 4, 8, 16),
    flash_attn = False,
    conditional = True
)

# 2. Gaussian Diffusion 설정
diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,
    objective = 'pred_v'
).to(device)

# 3. Trainer 설정 및 훈련 시작
# 주석: Trainer에 raw 이미지와 reference 이미지 폴더 경로를 각각 전달합니다.
trainer = Trainer(
    diffusion,
    raw_folder = 'data/UIEB/raw-890', # 원본 수중 영상(raw) 데이터셋 경로
    ref_folder = 'data/UIEB/reference-890', # 선명한 영상(reference) 데이터셋 경로
    train_batch_size = 2,
    val_batch_size= 8,
    train_lr = 8e-5,
    train_num_steps = 700000,
    gradient_accumulate_every = 8,
    ema_decay = 0.995,
    amp = True,
    # 주석: calculate_fid 대신 calculate_psnr를 사용합니다.
    calculate_psnr = True,
    save_and_sample_every = 560,
    early_stopping_patience = 10,
    resume_from_checkpoint = True # 체크포인트에서 재개할지 여부
)

# 훈련 시작
trainer.train()