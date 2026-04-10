"""
Lightweight Vision Transformer for Real-time Object Detection
Setup script for package installation
"""

from setuptools import setup, find_packages

with open("requirements.txt", "r", encoding="utf-8") as f:
    requirements = [line.strip() for line in f.readlines() 
                    if line.strip() and not line.startswith("#")]

setup(
    name="lightweight_vit_detection",
    version="1.0.0",
    author="CDS540 Project Team",
    author_email="team@example.com",
    description="Lightweight Vision Transformer for Real-time Object Detection",
    long_description=open("README.md", encoding="utf-8").read() if os.path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/example/lightweight-vit-detection",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "tensorrt": ["nvidia-tensorrt>=8.0.0"],
        "dev": [
            "black>=22.0.0",
            "isort>=5.10.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
    },
    entry_points={
        "console_scripts": [
            "vit-train=tools.train:main",
            "vit-evaluate=tools.evaluate:main",
            "vit-export=tools.export_model:main",
            "vit-demo=tools.demo:main",
        ],
    },
)

import os
