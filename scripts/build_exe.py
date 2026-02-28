#!/usr/bin/env python
"""
Build Script for Lightweight ViT Detection System.

This script builds the application into an executable using PyInstaller.

Usage:
    python scripts/build_exe.py
    python scripts/build_exe.py --onefile  # Single file executable
    python scripts/build_exe.py --debug    # Debug mode with console
"""

import os
import sys
import shutil
import argparse
import subprocess
from pathlib import Path


def check_dependencies():
    """Check if required dependencies are installed."""
    required = ['pyinstaller', 'PyQt6', 'torch', 'cv2']
    missing = []
    
    for dep in required:
        try:
            __import__(dep.replace('-', '_'))
        except ImportError:
            missing.append(dep)
            
    if missing:
        print(f"Missing dependencies: {', '.join(missing)}")
        print("Please install them first:")
        print(f"  pip install {' '.join(missing)}")
        return False
        
    return True


def clean_build_dirs(project_root: Path):
    """Clean previous build directories."""
    dirs_to_clean = ['build', 'dist', '__pycache__']
    
    for dir_name in dirs_to_clean:
        dir_path = project_root / dir_name
        if dir_path.exists():
            print(f"Cleaning {dir_path}...")
            shutil.rmtree(dir_path)
            
    # Clean .pyc files
    for pyc_file in project_root.rglob('*.pyc'):
        pyc_file.unlink()
        
    for pycache in project_root.rglob('__pycache__'):
        shutil.rmtree(pycache)


def build_executable(project_root: Path, onefile: bool = False, debug: bool = False):
    """Build the executable."""
    # Prepare PyInstaller command
    cmd = ['pyinstaller']
    
    if onefile:
        # Single file mode
        cmd.extend([
            '--onefile',
            '--name=LViT-Detection',
            '--windowed' if not debug else '--console',
            '--add-data=configs;configs',
            '--hidden-import=torch',
            '--hidden-import=torchvision',
            '--hidden-import=PyQt6.QtCore',
            '--hidden-import=PyQt6.QtGui',
            '--hidden-import=PyQt6.QtWidgets',
            '--hidden-import=cv2',
            '--hidden-import=einops',
            '--hidden-import=numpy',
            '--hidden-import=PIL',
            '--hidden-import=yaml',
            '--collect-all=torch',
            '--collect-all=torchvision',
            '--collect-all=PyQt6',
            'run_app.py'
        ])
    else:
        # Use spec file
        spec_file = project_root / 'build.spec'
        if not spec_file.exists():
            print(f"Error: Spec file not found: {spec_file}")
            return False
        cmd.append(str(spec_file))
        
    # Run PyInstaller
    print("Building executable...")
    print(f"Command: {' '.join(cmd)}")
    
    result = subprocess.run(cmd, cwd=str(project_root))
    
    if result.returncode != 0:
        print("Build failed!")
        return False
        
    print("Build completed successfully!")
    
    # Print output location
    if onefile:
        exe_path = project_root / 'dist' / 'LViT-Detection.exe'
    else:
        exe_path = project_root / 'dist' / 'LViT-Detection' / 'LViT-Detection.exe'
        
    if exe_path.exists():
        print(f"\nExecutable location: {exe_path}")
        print(f"File size: {exe_path.stat().st_size / (1024*1024):.1f} MB")
        
    return True


def main():
    """Main build function."""
    parser = argparse.ArgumentParser(description='Build Lightweight ViT Detection executable')
    parser.add_argument('--onefile', action='store_true',
                        help='Build as single file executable')
    parser.add_argument('--debug', action='store_true',
                        help='Build with console for debugging')
    parser.add_argument('--clean', action='store_true',
                        help='Clean build directories before building')
    parser.add_argument('--clean-only', action='store_true',
                        help='Only clean build directories')
    
    args = parser.parse_args()
    
    # Get project root
    project_root = Path(__file__).parent.parent
    
    print("=" * 60)
    print("Lightweight ViT Detection System - Build Tool")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print()
    
    # Clean if requested
    if args.clean or args.clean_only:
        clean_build_dirs(project_root)
        if args.clean_only:
            print("Clean completed.")
            return
            
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
        
    # Build
    success = build_executable(project_root, args.onefile, args.debug)
    
    if success:
        print("\n" + "=" * 60)
        print("BUILD SUCCESSFUL")
        print("=" * 60)
        print("\nTo run the application:")
        if args.onefile:
            print("  dist/LViT-Detection.exe")
        else:
            print("  dist/LViT-Detection/LViT-Detection.exe")
    else:
        sys.exit(1)


if __name__ == '__main__':
    main()
