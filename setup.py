"""
Setup script for Lightweight ViT Detection System.

This package provides a lightweight Vision Transformer-based
object detection system optimized for mobile and edge devices.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = ""
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().split("\n")
        if line.strip() and not line.startswith("#")
    ]

setup(
    name="lightweight-vit-detection",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Lightweight Vision Transformer for Real-time Object Detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/lightweight-vit-detection",
    
    # Package configuration
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Dependencies
    install_requires=requirements,
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "flake8>=3.9.0",
            "mypy>=0.900",
            "pre-commit>=2.0.0",
            "black>=21.0.0",
            "isort>=5.0.0",
        ],
        "deploy": [
            "onnx>=1.10.0",
            "onnxruntime>=1.10.0",
            "onnx-simplifier>=0.3.0",
        ],
    },
    
    # Entry points
    entry_points={
        "console_scripts": [
            "lvit-train=scripts.train:main",
            "lvit-eval=scripts.evaluate:main",
            "lvit-infer=scripts.inference:main",
            "lvit-distill=scripts.distill:main",
        ],
    },
    
    # Classifiers
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    
    # Keywords
    keywords=[
        "deep learning",
        "computer vision",
        "object detection",
        "vision transformer",
        "mobilevit",
        "pytorch",
        "edge computing",
    ],
    
    # Include package data
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml"],
    },
)
