import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
from torchvision.utils import save_image

# 1. 모델 및 Diffusion 설정 (훈련 스크립트와 동일하게)
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = True
)

diffusion = GaussianDiffusion(
    model,
    image_size = 256,
    timesteps = 1000,
    objective = 'pred_v'
)

# 2. Trainer를 통해 모델 가중치 로드
# Trainer는 학습 상태(optimizer, ema 등)를 포함하므로,
# 여기서는 가중치 로드 목적으로만 사용합니다.
trainer = Trainer(
    diffusion,
    'denoising-diffusion-pytorch-main/data/UIEB/reference-890', # 경로는 필요하지만, 샘플링 시에는 사용되지 않음
    train_batch_size = 4,
    calculate_fid = False # FID 계산 비활성화
)

# 로드할 모델 체크포인트의 마일스톤 번호 지정
# 예: 10 에포크 후 저장된 모델은 1, 20 에포크 후는 2
MILESTONE_NUMBER = 10 # 예시로 10번째 저장된 모델(100 에포크)을 로드
trainer.load(MILESTONE_NUMBER)

# 3. 이미지 샘플링
# batch_size: 생성할 이미지 개수
SAMPLED_IMAGES_COUNT = 4
sampled_images = diffusion.sample(batch_size = SAMPLED_IMAGES_COUNT)

# 4. 생성된 이미지 저장
# save_image 함수는 텐서를 이미지 파일로 저장해줍니다.
save_image(sampled_images, f'./sampled-images-milestone-{MILESTONE_NUMBER}.png', nrow = int(SAMPLED_IMAGES_COUNT ** 0.5))

print(f"{SAMPLED_IMAGES_COUNT}개의 이미지가 'sampled-images-milestone-{MILESTONE_NUMBER}.png' 파일로 저장되었습니다.")