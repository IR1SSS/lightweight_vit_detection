# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller Specification File for Lightweight ViT Detection System.

This file configures how the application is packaged into an executable.

Usage:
    pyinstaller build.spec
    
Or use the build script:
    python scripts/build_exe.py
"""

import sys
from pathlib import Path

# Project root directory
project_root = Path(SPECPATH)

# Collect all source files
src_path = project_root / 'src'

block_cipher = None

# Analysis configuration
a = Analysis(
    ['run_app.py'],
    pathex=[str(project_root)],
    binaries=[],
    datas=[
        # Include configuration files
        ('configs', 'configs'),
        # Include any additional resources
    ],
    hiddenimports=[
        # PyTorch imports
        'torch',
        'torch.nn',
        'torch.nn.functional',
        'torchvision',
        'torchvision.transforms',
        
        # PyQt6 imports
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        
        # OpenCV imports
        'cv2',
        
        # NumPy imports
        'numpy',
        
        # Other imports
        'einops',
        'PIL',
        'yaml',
        'tqdm',
        
        # Project modules
        'src',
        'src.models',
        'src.models.mobilevit',
        'src.models.base_model',
        'src.models.backbone',
        'src.models.detection_head',
        'src.data',
        'src.data.datasets',
        'src.data.transforms',
        'src.training',
        'src.distillation',
        'src.quantization',
        'src.deployment',
        'src.utils',
        'src.utils.config_loader',
        'src.utils.logger',
        'src.gui',
        'src.gui.main_window',
        'src.gui.detector',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary packages to reduce size
        'tkinter',
        'matplotlib',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'sphinx',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# PYZ archive
pyz = PYZ(
    a.pure,
    a.zipped_data,
    cipher=block_cipher
)

# Executable configuration
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='LViT-Detection',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Set to True for debugging
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add icon path here: 'resources/icon.ico'
)

# Collect all files
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='LViT-Detection',
)
