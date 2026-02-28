#!/usr/bin/env python
"""
COCO Dataset Download Script.

This script downloads the COCO 2017 dataset for training and evaluation.

Usage:
    python download_coco.py                    # Download full dataset
    python download_coco.py --val-only         # Download validation set only
    python download_coco.py --data-dir custom/path  # Custom download directory
"""

import os
import sys

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.download import download_coco_dataset


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Download COCO 2017 Dataset',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/coco',
        help='Directory to store dataset'
    )
    parser.add_argument(
        '--train-only', action='store_true',
        help='Download only training set'
    )
    parser.add_argument(
        '--val-only', action='store_true',
        help='Download only validation set (recommended for quick start)'
    )
    parser.add_argument(
        '--keep-zip', action='store_true',
        help='Keep zip files after extraction'
    )
    
    args = parser.parse_args()
    
    download_train = not args.val_only
    download_val = not args.train_only
    
    print("COCO 2017 Dataset Downloader")
    print("=" * 50)
    print(f"Download directory: {os.path.abspath(args.data_dir)}")
    print(f"Download training set: {download_train}")
    print(f"Download validation set: {download_val}")
    print()
    
    if download_train:
        print("Note: Training set is ~18GB, this may take a while.")
        print("Use --val-only for a quick start with validation set (~778MB).")
        print()
    
    download_coco_dataset(
        data_dir=args.data_dir,
        download_train=download_train,
        download_val=download_val,
        cleanup_zip=not args.keep_zip
    )
