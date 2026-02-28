"""
COCO Dataset Download Utilities.

This module provides functions to automatically download and extract
the COCO dataset if it is not present locally.
"""

import os
import sys
import zipfile
import hashlib
from typing import Optional
from urllib.request import urlretrieve
from tqdm import tqdm


# COCO 2017 Dataset URLs
COCO_URLS = {
    'train2017': {
        'url': 'http://images.cocodataset.org/zips/train2017.zip',
        'size': '18GB',
        'md5': None  # MD5 check optional for large files
    },
    'val2017': {
        'url': 'http://images.cocodataset.org/zips/val2017.zip',
        'size': '778MB',
        'md5': None
    },
    'annotations': {
        'url': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',
        'size': '241MB',
        'md5': None
    }
}


class DownloadProgressBar(tqdm):
    """Progress bar for downloads."""
    
    def update_to(self, b: int = 1, bsize: int = 1, tsize: Optional[int] = None):
        """
        Update progress bar.
        
        Args:
            b: Number of blocks transferred so far.
            bsize: Size of each block (in bytes).
            tsize: Total size (in bytes).
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_file(url: str, dest_path: str, desc: str = "Downloading") -> str:
    """
    Download a file with progress bar.
    
    Args:
        url: URL to download from.
        dest_path: Destination file path.
        desc: Description for progress bar.
        
    Returns:
        Path to downloaded file.
    """
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=desc) as t:
        urlretrieve(url, filename=dest_path, reporthook=t.update_to)
        
    return dest_path


def extract_zip(zip_path: str, extract_dir: str, desc: str = "Extracting") -> None:
    """
    Extract a zip file with progress bar.
    
    Args:
        zip_path: Path to zip file.
        extract_dir: Directory to extract to.
        desc: Description for progress bar.
    """
    os.makedirs(extract_dir, exist_ok=True)
    
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        members = zip_ref.namelist()
        
        with tqdm(total=len(members), desc=desc) as pbar:
            for member in members:
                zip_ref.extract(member, extract_dir)
                pbar.update(1)


def download_coco_dataset(
    data_dir: str = 'data/coco',
    download_train: bool = True,
    download_val: bool = True,
    download_annotations: bool = True,
    cleanup_zip: bool = True
) -> None:
    """
    Download COCO 2017 dataset.
    
    Args:
        data_dir: Directory to store the dataset.
        download_train: Whether to download training images.
        download_val: Whether to download validation images.
        download_annotations: Whether to download annotations.
        cleanup_zip: Whether to delete zip files after extraction.
    """
    os.makedirs(data_dir, exist_ok=True)
    zip_dir = os.path.join(data_dir, 'zips')
    os.makedirs(zip_dir, exist_ok=True)
    
    print("=" * 60)
    print("COCO 2017 Dataset Download")
    print("=" * 60)
    print(f"Download directory: {os.path.abspath(data_dir)}")
    print()
    
    # Download annotations first (required for both train and val)
    if download_annotations:
        ann_dir = os.path.join(data_dir, 'annotations')
        if not os.path.exists(os.path.join(ann_dir, 'instances_train2017.json')):
            print("Downloading annotations...")
            zip_path = os.path.join(zip_dir, 'annotations_trainval2017.zip')
            
            if not os.path.exists(zip_path):
                download_file(
                    COCO_URLS['annotations']['url'],
                    zip_path,
                    f"annotations ({COCO_URLS['annotations']['size']})"
                )
            
            print("Extracting annotations...")
            extract_zip(zip_path, data_dir, "Extracting annotations")
            
            if cleanup_zip:
                os.remove(zip_path)
                print("Cleaned up annotations zip file.")
            print("Annotations downloaded successfully!")
            print()
        else:
            print("Annotations already exist, skipping...")
            print()
    
    # Download validation images
    if download_val:
        val_dir = os.path.join(data_dir, 'val2017')
        if not os.path.exists(val_dir) or len(os.listdir(val_dir)) < 5000:
            print("Downloading validation images...")
            zip_path = os.path.join(zip_dir, 'val2017.zip')
            
            if not os.path.exists(zip_path):
                download_file(
                    COCO_URLS['val2017']['url'],
                    zip_path,
                    f"val2017 ({COCO_URLS['val2017']['size']})"
                )
            
            print("Extracting validation images...")
            extract_zip(zip_path, data_dir, "Extracting val2017")
            
            if cleanup_zip:
                os.remove(zip_path)
                print("Cleaned up val2017 zip file.")
            print("Validation images downloaded successfully!")
            print()
        else:
            print("Validation images already exist, skipping...")
            print()
    
    # Download training images
    if download_train:
        train_dir = os.path.join(data_dir, 'train2017')
        if not os.path.exists(train_dir) or len(os.listdir(train_dir)) < 118000:
            print("Downloading training images...")
            print(f"WARNING: Training set is large ({COCO_URLS['train2017']['size']}). This may take a while.")
            zip_path = os.path.join(zip_dir, 'train2017.zip')
            
            if not os.path.exists(zip_path):
                download_file(
                    COCO_URLS['train2017']['url'],
                    zip_path,
                    f"train2017 ({COCO_URLS['train2017']['size']})"
                )
            
            print("Extracting training images...")
            extract_zip(zip_path, data_dir, "Extracting train2017")
            
            if cleanup_zip:
                os.remove(zip_path)
                print("Cleaned up train2017 zip file.")
            print("Training images downloaded successfully!")
            print()
        else:
            print("Training images already exist, skipping...")
            print()
    
    # Cleanup zip directory if empty
    if cleanup_zip and os.path.exists(zip_dir) and not os.listdir(zip_dir):
        os.rmdir(zip_dir)
    
    print("=" * 60)
    print("COCO dataset download complete!")
    print("=" * 60)


def check_coco_dataset(
    data_dir: str = 'data/coco',
    check_train: bool = True,
    check_val: bool = True
) -> dict:
    """
    Check if COCO dataset exists and is complete.
    
    Args:
        data_dir: COCO data directory.
        check_train: Whether to check training set.
        check_val: Whether to check validation set.
        
    Returns:
        Dictionary with status of each component.
    """
    status = {
        'annotations_train': False,
        'annotations_val': False,
        'images_train': False,
        'images_val': False
    }
    
    # Check annotations
    ann_dir = os.path.join(data_dir, 'annotations')
    if os.path.exists(os.path.join(ann_dir, 'instances_train2017.json')):
        status['annotations_train'] = True
    if os.path.exists(os.path.join(ann_dir, 'instances_val2017.json')):
        status['annotations_val'] = True
        
    # Check images
    if check_train:
        train_dir = os.path.join(data_dir, 'train2017')
        if os.path.exists(train_dir) and len(os.listdir(train_dir)) >= 118000:
            status['images_train'] = True
            
    if check_val:
        val_dir = os.path.join(data_dir, 'val2017')
        if os.path.exists(val_dir) and len(os.listdir(val_dir)) >= 5000:
            status['images_val'] = True
            
    return status


def ensure_coco_dataset(
    data_dir: str = 'data/coco',
    require_train: bool = True,
    require_val: bool = True,
    auto_download: bool = True
) -> bool:
    """
    Ensure COCO dataset is available, download if necessary.
    
    Args:
        data_dir: COCO data directory.
        require_train: Whether training set is required.
        require_val: Whether validation set is required.
        auto_download: Whether to automatically download if missing.
        
    Returns:
        True if dataset is available, False otherwise.
    """
    status = check_coco_dataset(data_dir, require_train, require_val)
    
    need_download = False
    
    if require_train and (not status['annotations_train'] or not status['images_train']):
        need_download = True
        
    if require_val and (not status['annotations_val'] or not status['images_val']):
        need_download = True
        
    if need_download:
        if auto_download:
            print("\nCOCO dataset not found or incomplete.")
            print("Starting automatic download...\n")
            download_coco_dataset(
                data_dir=data_dir,
                download_train=require_train and not status['images_train'],
                download_val=require_val and not status['images_val'],
                download_annotations=not status['annotations_train'] or not status['annotations_val']
            )
            return True
        else:
            print("\nCOCO dataset not found. Please download manually or set auto_download=True")
            return False
            
    return True


if __name__ == '__main__':
    """Command line interface for downloading COCO dataset."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Download COCO 2017 Dataset')
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
        help='Download only validation set'
    )
    parser.add_argument(
        '--keep-zip', action='store_true',
        help='Keep zip files after extraction'
    )
    
    args = parser.parse_args()
    
    download_train = not args.val_only
    download_val = not args.train_only
    
    download_coco_dataset(
        data_dir=args.data_dir,
        download_train=download_train,
        download_val=download_val,
        cleanup_zip=not args.keep_zip
    )
