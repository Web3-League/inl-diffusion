"""
INL-Diffusion: Image Generation with Integrator Neurons

Setup script for PyPI distribution.
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="inl-diffusion",
    version="0.3.0",
    author="Pacific Prime",
    author_email="contact@pacific-prime.ai",
    description="Image generation with Integrator Neurons - Text-to-image diffusion model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Web3-League/inl-diffusion",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "tqdm>=4.66.0",
        "pillow>=9.0.0",
        "numpy>=1.21.0",
    ],
    extras_require={
        "train": [
            "datasets>=2.16.0",
            "tensorboard>=2.15.0",
            "wandb>=0.16.0",
            "accelerate>=0.25.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "flake8>=6.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "inl-diffusion-train-vae=train_vae:main",
            "inl-diffusion-train-dit=train_dit:main",
            "inl-diffusion-generate=generate:main",
        ],
    },
)
