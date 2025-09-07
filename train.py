import torch
from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer

# 1. U-Net 모델 설정
# dim: 모델의 기본 채널 수. 높을수록 모델 표현력이 좋아지지만 무거워집니다.
# dim_mults: U-Net의 각 다운샘플링 단계에서 채널 수를 몇 배로 늘릴지 결정합니다.
model = Unet(
    dim = 64,
    dim_mults = (1, 2, 4, 8),
    flash_attn = False # Flash Attention을 사용하여 메모리 사용량 감소 및 속도 향상
)

# 2. Gaussian Diffusion 설정
# Unet 모델을 감싸 확산 및 노이즈 제거 프로세스를 관리합니다.
diffusion = GaussianDiffusion(
    model,
    image_size = 256,         # 훈련할 이미지 크기
    timesteps = 1000,           # 노이즈를 추가하는 총 단계 수
    objective = 'pred_v'      # 예측 목표. 'pred_v'는 고품질 이미지 생성에 효과적입니다.
)

# 3. Trainer 설정 및 훈련 시작
# 데이터셋 로딩, 모델 최적화, 샘플링, 저장을 자동화합니다.
trainer = Trainer(
    diffusion,
    'data/UIEB/reference-890', # UIEB 데이터셋 경로
    train_batch_size = 8,
    train_lr = 8e-5,
    train_num_steps = 700000,         # 총 훈련 스텝 수
    gradient_accumulate_every = 4,    # 그래디언트 축적. 실질적 배치 크기를 늘려 안정적인 훈련을 돕습니다.
    ema_decay = 0.995,                # Exponential Moving Average. 모델 성능을 부드럽게 업데이트합니다.
    amp = True,                       # Automatic Mixed Precision. 혼합 정밀도 사용으로 훈련 속도 향상
    calculate_fid = False,            # FID 점수 계산 비활성화 (단순화를 위해)
    save_and_sample_every = 1120      # 10 에포크(890 이미지 / 배치 8 * 10)마다 모델 저장 및 샘플 생성
)

# 훈련 시작
trainer.train()