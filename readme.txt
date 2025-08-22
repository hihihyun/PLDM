# Underwater Image Enhancement with Diffusion Models

A state-of-the-art underwater image enhancement model that combines VAE (Variational Autoencoder), Diffusion Models, and Water-Net physics-based preprocessing to restore underwater images.

## 🌊 Features

- **Physics-Based Preprocessing**: Incorporates Water-Net style preprocessing including white balance, gamma correction, and histogram equalization
- **VAE-Diffusion Architecture**: Combines the efficiency of VAE with the quality of diffusion models
- **Conditional Enhancement**: Uses physics-based features as conditioning for targeted underwater image restoration
- **Multiple Loss Functions**: Reconstruction, perceptual, SSIM, frequency domain, and underwater-specific losses
- **Flexible Training**: Supports both UIEB and LSUI datasets with configurable parameters
- **Fast Inference**: DDIM sampling for faster generation

## 📁 Project Structure

```
├── models/
│   ├── __init__.py           # Models package initialization
│   ├── basic_modules.py      # Basic building blocks (ResBlock, Attention, etc.)
│   ├── water_physics.py      # Water-Net preprocessing and physics modules
│   ├── vae_encoder.py        # VAE encoder with physics integration
│   ├── vae_decoder.py        # VAE decoder with multiple variants
│   ├── diffusion_unet.py     # Conditional diffusion UNet
│   ├── loss_functions.py     # Comprehensive loss functions
│   └── main_model.py         # Complete integrated model
├── data/
│   ├── __init__.py           # Data package initialization
│   └── dataset.py            # Dataset loaders for UIEB and LSUI
├── configs/                  # Configuration templates (auto-generated)
├── experiments/              # Training results and checkpoints
├── DATA/                     # 📁 Put your datasets here!
│   ├── UIEB/                 # UIEB dataset folder
│   │   ├── raw-890/          # Training degraded images
│   │   ├── reference-890/    # Training enhanced images
│   │   └── challengingset-60/ # Test images
│   └── LSUI/                 # LSUI dataset folder
│       ├── input/            # Input images
│       └── GT/               # Ground truth images
├── train.py                  # Training script
├── test.py                   # Testing and inference script
├── demo.py                   # 🎭 Demo script for quick testing
├── config.py                 # Configuration management
├── utils.py                  # Utility functions
├── setup_dataset.py          # 🚀 One-click setup script
├── check_dataset.py          # 🔍 Dataset validation script
├── create_test_data.py       # 🎨 Synthetic data generator
├── setup.py                  # Package installation script
├── requirements.txt          # Python dependencies
├── QUICKSTART.md             # ⚡ Quick start guide
├── DATA_SETUP.md             # 📁 Detailed data setup guide
└── README.md                 # This file
```

## 🚀 Quick Start

### 1. Installation

```bash
# Clone the repository
git clone <repository-url>
cd underwater-image-enhancement

# Install dependencies
pip install -r requirements.txt

# OR install as a package
pip install -e .
```

### 2. One-Click Setup (Recommended)

```bash
# Complete setup with test data
python setup_dataset.py --dataset test --test_samples 50

# Verify everything works
python demo.py --demo all
```

**That's it!** 🎉 Your model is ready to use!

### 3. Manual Dataset Setup

If you want to use real datasets:

```bash
# Check current data status
python check_dataset.py --dataset_type UIEB --report

# Create test data for development
python create_test_data.py --dataset_type UIEB --num_samples 50
```

Organize your datasets in the `DATA/` folder:

```
DATA/
├── UIEB/
│   ├── raw-890/              # Training degraded images
│   ├── reference-890/        # Training ground truth images
│   └── challengingset-60/    # Test images
└── LSUI/
    ├── input/                # Degraded images
    └── GT/                   # Ground truth images
```

**Download Links:**
- **UIEB**: [Official Page](https://li-chongyi.github.io/proj_benchmark.html)
- **LSUI**: [GitHub Repository](https://github.com/dalabdune/LSUI)

### 4. Training

```bash
# Train with default UIEB configuration
python train.py --config configs/uieb.yaml --exp_name uieb_experiment

# Quick test with lightweight model
python train.py --config configs/lightweight.yaml --exp_name quick_test

# High quality training
python train.py --config configs/high_quality.yaml --exp_name hq_experiment
```

### 5. Testing

```bash
# Test single image
python test.py --mode single \
    --model_path experiments/uieb_experiment/checkpoints/best_model.pth \
    --input_image test_image.jpg \
    --output_image enhanced_result.jpg

# Test entire dataset
python test.py --mode dataset \
    --model_path experiments/uieb_experiment/checkpoints/best_model.pth \
    --test_dataset UIEB \
    --data_root ./DATA \
    --output_dir ./results

# Batch processing
python test.py --mode batch \
    --model_path experiments/uieb_experiment/checkpoints/best_model.pth \
    --input_dir ./test_images \
    --output_dir ./enhanced_images
```

### 6. Python API Usage

```python
# Quick setup
from underwater_enhancement import quick_setup
model, config, dataset_config = quick_setup('UIEB', 'default')

# Or manual setup
from models import UnderwaterEnhancementModel
from PIL import Image
import torchvision.transforms as transforms

# Load model
model = UnderwaterEnhancementModel('path/to/checkpoint.pth')

# Process image
image = Image.open('underwater_image.jpg')
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])
image_tensor = transform(image).unsqueeze(0)

# Enhance image
enhanced = model.enhance_image(image_tensor, num_steps=50)

# Save result
enhanced_pil = transforms.ToPILImage()(enhanced.squeeze(0))
enhanced_pil.save('enhanced_result.jpg')
```

📖 **For detailed setup instructions, see [QUICKSTART.md](QUICKSTART.md) and [DATA_SETUP.md](DATA_SETUP.md)**

## ⚙️ Configuration

The project uses YAML configuration files for easy experimentation. Key configurations include:

### Model Configuration
```yaml
model:
  img_size: 256
  in_channels: 3
  latent_channels: 4
  base_channels: 128
  time_steps: 1000
  physics_dim: 128
```

### Training Configuration
```yaml
training:
  num_epochs: 1000
  vae_lr: 1e-4
  diffusion_lr: 1e-4
  batch_size: 8
  grad_clip: 1.0
```

### Available Presets
- `default.yaml`: Balanced configuration for general use
- `uieb.yaml`: Optimized for UIEB dataset
- `lsui.yaml`: Optimized for LSUI dataset
- `lightweight.yaml`: Faster training/inference
- `high_quality.yaml`: Best quality results

## 📊 Model Architecture

### VAE-Diffusion Pipeline
1. **Preprocessing**: Water-Net style physics-based preprocessing
2. **Encoding**: VAE encoder with physics feature integration
3. **Diffusion**: Conditional diffusion in latent space
4. **Decoding**: VAE decoder to image space

### Key Components
- **Water Physics Module**: Estimates transmission, backscatter, and attenuation
- **Conditional Fusion**: Combines original and preprocessed features
- **Attention Mechanisms**: Self and cross-attention for feature refinement
- **Multi-Loss Training**: Comprehensive loss functions for quality enhancement

## 🔬 Technical Details

### Loss Functions
- **Reconstruction Loss**: L1 + L2 reconstruction
- **Perceptual Loss**: VGG-based feature matching
- **SSIM Loss**: Structural similarity preservation
- **Gradient Loss**: Edge and structure preservation
- **Frequency Loss**: Texture detail enhancement
- **Underwater-Specific Loss**: Color correction and contrast enhancement

### Diffusion Sampling
- **DDPM**: Standard denoising diffusion
- **DDIM**: Faster deterministic sampling
- **Configurable Steps**: 10-100 steps for speed/quality trade-off

## 📈 Performance

### Metrics
The model is evaluated using:
- PSNR (Peak Signal-to-Noise Ratio)
- SSIM (Structural Similarity Index)
- LPIPS (Learned Perceptual Image Patch Similarity)
- Underwater-specific color metrics

### Speed
- Training: ~2-3 seconds per batch (8 images, 256x256) on RTX 3090
- Inference: ~1-5 seconds per image depending on sampling steps

## 🛠️ Advanced Usage

### Custom Dataset
```python
from data.dataset import UnderwaterDataset

dataset = UnderwaterDataset(
    root_dir='path/to/your/data',
    dataset_type='custom',
    img_size=256,
    preprocessing_type='waternet'
)
```

### Model Inference
```python
from models.main_model import UnderwaterEnhancementModel

model = UnderwaterEnhancementModel('path/to/checkpoint.pth')
enhanced = model.enhance_image(degraded_image, num_steps=50)
```

### Configuration Creation
```python
from config import get_config, save_config

config = get_config('default')
config['model']['img_size'] = 512
save_config(config, 'custom_config.yaml')
```

## 📚 Research Background

This implementation is based on combining:

1. **Water-Net**: Physics-based underwater image enhancement
2. **Diffusion Models**: State-of-the-art generative modeling
3. **VAE**: Efficient latent space representation
4. **Conditional Generation**: Physics-informed enhancement

### Datasets
- **UIEB**: Underwater Image Enhancement Benchmark
- **LSUI**: Large-scale Underwater Image dataset

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Water-Net authors for physics-based preprocessing insights
- Diffusion model research community
- UIEB and LSUI dataset creators
- PyTorch team for the deep learning framework

## 📞 Contact

For questions or issues, please:
1. Check the [QUICKSTART.md](QUICKSTART.md) for common solutions
2. Validate your setup: `python check_dataset.py --report`
3. Run diagnostics: `python demo.py --demo all`
4. Create a detailed issue with logs and configuration

## 🔧 Additional Tools

- **🔍 Dataset Checker**: `python check_dataset.py --dataset_type UIEB --report`
- **🎨 Test Data Generator**: `python create_test_data.py --num_samples 50`
- **🚀 One-Click Setup**: `python setup_dataset.py --dataset test`
- **🎭 Feature Demo**: `python demo.py --demo all`
- **⚙️ Config Generator**: `python config.py --create_templates`

## 🔮 Future Work

- [ ] Transformer-based attention mechanisms
- [ ] Multi-scale training and inference
- [ ] Real-time video enhancement
- [ ] Mobile deployment optimization
- [ ] Additional underwater datasets support

---

**Note**: This is a research implementation. For production use, consider model optimization and thorough testing on your specific use cases.