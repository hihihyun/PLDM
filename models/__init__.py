"""
Models package for Underwater Image Enhancement Diffusion Model
"""

from .main_model import (
    UnderwaterEnhancementDiffusion,
    UnderwaterEnhancementModel,
    create_model
)

from .vae_encoder import (
    EnhancedVAEEncoder,
    ResidualVAEEncoder,
    reparameterize
)

from .vae_decoder import (
    VAEDecoder,
    SkipConnectionVAEDecoder,
    ProgressiveVAEDecoder
)

from .diffusion_unet import (
    ConditionalDiffusionUNet,
    LightweightDiffusionUNet
)

from .water_physics import (
    WaterPhysicsModule,
    WaterNetPreprocessor,
    ConditionalFusionModule
)

from .loss_functions import (
    StructureAwareLoss,
    VGGPerceptualLoss,
    UnderwaterSpecificLoss,
    CombinedLoss
)

from .basic_modules import (
    ResBlock,
    AttentionBlock,
    ConditionalResBlock,
    CrossAttentionBlock,
    SinusoidalPositionEmbedding,
    SEBlock,
    ConvBlock,
    DepthwiseSeparableConv
)

__all__ = [
    # Main models
    'UnderwaterEnhancementDiffusion',
    'UnderwaterEnhancementModel',
    'create_model',
    
    # VAE components
    'EnhancedVAEEncoder',
    'ResidualVAEEncoder',
    'VAEDecoder',
    'SkipConnectionVAEDecoder',
    'ProgressiveVAEDecoder',
    'reparameterize',
    
    # Diffusion components
    'ConditionalDiffusionUNet',
    'LightweightDiffusionUNet',
    
    # Water physics components
    'WaterPhysicsModule',
    'WaterNetPreprocessor',
    'ConditionalFusionModule',
    
    # Loss functions
    'StructureAwareLoss',
    'VGGPerceptualLoss',
    'UnderwaterSpecificLoss',
    'CombinedLoss',
    
    # Basic modules
    'ResBlock',
    'AttentionBlock',
    'ConditionalResBlock',
    'CrossAttentionBlock',
    'SinusoidalPositionEmbedding',
    'SEBlock',
    'ConvBlock',
    'DepthwiseSeparableConv'
]

# Version info
__version__ = '1.0.0'
__author__ = 'Underwater Enhancement Team'
__description__ = 'Underwater Image Enhancement using Diffusion Models with Water-Net Physics'
