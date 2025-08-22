"""
Ultra-safe training script - CPU only, no complex models
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
import argparse

class SimplestModel(nn.Module):
    """가장 단순한 모델 - device 문제 없음"""
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 16, 3, padding=1)
        self.conv4 = nn.Conv2d(16, 3, 3, padding=1)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        h = self.relu(self.conv1(x))
        h = self.relu(self.conv2(h))
        h = self.relu(self.conv3(h))
        h = self.tanh(self.conv4(h))
        return h

class SimpleDataset(Dataset):
    """완전히 단순한 데이터셋"""
    def __init__(self, size=100, img_size=64):
        self.size = size
        self.img_size = img_size
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # CPU에서 직접 생성
        degraded = torch.randn(3, self.img_size, self.img_size) * 0.3 + 0.5
        enhanced = degraded * 1.1 + 0.05  # 약간 개선된 버전
        
        # [0, 1] 범위로 클램핑
        degraded = torch.clamp(degraded, 0, 1)
        enhanced = torch.clamp(enhanced, 0, 1)
        
        return degraded, enhanced

def ultra_safe_train():
    """완전히 안전한 훈련"""
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    parser.add_argument('--epochs', type=int, default=5, help='Epochs')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    args = parser.parse_args()
    
    print("🛡️  ULTRA SAFE UNDERWATER TRAINING")
    print("=" * 50)
    
    device = torch.device(args.device)
    print(f"🖥️  Device: {device}")
    
    # 1. 가장 단순한 모델
    model = SimplestModel().to(device)
    print(f"✅ Model created with {sum(p.numel() for p in model.parameters())} parameters")
    
    # 2. 가장 단순한 데이터
    dataset = SimpleDataset(size=200, img_size=64)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    print(f"✅ Dataset created: {len(dataset)} samples")
    
    # 3. 옵티마이저
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()
    
    print(f"🔄 Starting {args.epochs} epochs...")
    
    # 4. 훈련 루프
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        valid_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}")
        
        for batch_idx, (degraded, enhanced) in enumerate(pbar):
            try:
                # Device로 이동
                degraded = degraded.to(device)
                enhanced = enhanced.to(device)
                
                # 정규화 [-1, 1]
                degraded = degraded * 2.0 - 1.0
                enhanced = enhanced * 2.0 - 1.0
                
                # 순전파
                optimizer.zero_grad()
                pred = model(degraded)
                loss = criterion(pred, enhanced)
                
                # 역전파
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                valid_batches += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.6f}'})
                
                # 안전을 위해 배치 수 제한
                if batch_idx >= 100:
                    break
                    
            except Exception as e:
                print(f"❌ Batch {batch_idx} failed: {e}")
                continue
        
        avg_loss = total_loss / max(valid_batches, 1)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.6f}")
        
        # 간단한 검증
        if epoch % 2 == 0:
            model.eval()
            with torch.no_grad():
                test_degraded, test_enhanced = next(iter(dataloader))
                test_degraded = test_degraded.to(device) * 2.0 - 1.0
                test_enhanced = test_enhanced.to(device) * 2.0 - 1.0
                
                test_pred = model(test_degraded)
                test_loss = criterion(test_pred, test_enhanced)
                print(f"  Validation Loss: {test_loss.item():.6f}")
    
    print("✅ Ultra safe training completed!")
    
    # 최종 테스트
    print("\n🧪 Final test...")
    model.eval()
    test_input = torch.randn(1, 3, 64, 64).to(device)
    with torch.no_grad():
        test_output = model(test_input)
        print(f"Input shape: {test_input.shape}")
        print(f"Output shape: {test_output.shape}")
        print(f"Output range: [{test_output.min():.3f}, {test_output.max():.3f}]")
    
    # 모델 저장
    torch.save(model.state_dict(), 'ultra_safe_model.pth')
    print("💾 Model saved as ultra_safe_model.pth")

if __name__ == "__main__":
    ultra_safe_train()
