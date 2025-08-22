"""
Setup script for Underwater Image Enhancement with Diffusion Models
"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements
def read_requirements():
    with open("requirements.txt", "r", encoding="utf-8") as fh:
        return [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="underwater-image-enhancement",
    version="1.0.0",
    author="Underwater Enhancement Research Team",
    author_email="contact@underwater-enhancement.com",
    description="Underwater Image Enhancement using Diffusion Models with Water-Net Physics",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/underwater-enhancement/diffusion-model",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "pre-commit>=2.17.0",
        ],
        "advanced": [
            "albumentations>=1.2.0",
            "kornia>=0.6.0",
            "hydra-core>=1.2.0",
            "lightning>=1.6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "underwater-enhance=train:main",
            "underwater-test=test:main",
            "underwater-config=config:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.json", "*.md", "*.txt"],
    },
    keywords=[
        "underwater", "image enhancement", "diffusion models", 
        "computer vision", "deep learning", "pytorch", "water-net"
    ],
    project_urls={
        "Bug Reports": "https://github.com/underwater-enhancement/diffusion-model/issues",
        "Source": "https://github.com/underwater-enhancement/diffusion-model",
        "Documentation": "https://underwater-enhancement.readthedocs.io/",
    },
)
