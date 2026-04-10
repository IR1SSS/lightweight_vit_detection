#!/usr/bin/env python
"""
COCO 2017 数据集下载脚本。

Usage:
    python scripts/download_coco.py --output-dir data/coco
    python scripts/download_coco.py --output-dir data/coco --skip-train  # 仅下载验证集
"""

import argparse
import os
import sys
from pathlib import Path
from urllib.request import urlretrieve
import zipfile
import shutil


# COCO 2017 数据集下载链接
COCO_URLS = {
    "train2017": "http://images.cocodataset.org/zips/train2017.zip",
    "val2017": "http://images.cocodataset.org/zips/val2017.zip",
    "annotations": "http://images.cocodataset.org/annotations/annotations_trainval2017.zip",
}

# 文件大小信息（用于显示）
FILE_SIZES = {
    "train2017": "~18GB",
    "val2017": "~1GB",
    "annotations": "~241MB",
}


def download_progress(block_num, block_size, total_size):
    """下载进度回调函数。"""
    downloaded = block_num * block_size
    percent = min(downloaded / total_size * 100, 100)
    mb_downloaded = downloaded / (1024 * 1024)
    mb_total = total_size / (1024 * 1024)
    print(f"\r下载进度: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end="")
    if percent >= 100:
        print()


def download_file(url: str, output_path: Path, desc: str = None):
    """下载文件。"""
    if desc:
        print(f"正在下载 {desc}...")
    print(f"URL: {url}")
    print(f"保存到: {output_path}")
    
    try:
        urlretrieve(url, output_path, download_progress)
        print("下载完成!")
        return True
    except Exception as e:
        print(f"下载失败: {e}")
        return False


def extract_zip(zip_path: Path, output_dir: Path):
    """解压 ZIP 文件。"""
    print(f"正在解压 {zip_path.name}...")
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(output_dir)
        print("解压完成!")
        return True
    except Exception as e:
        print(f"解压失败: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="下载 COCO 2017 数据集")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/coco",
        help="数据集保存目录",
    )
    parser.add_argument(
        "--skip-train",
        action="store_true",
        help="跳过训练集下载（仅下载验证集和标注）",
    )
    parser.add_argument(
        "--skip-val",
        action="store_true",
        help="跳过验证集下载",
    )
    parser.add_argument(
        "--skip-annotations",
        action="store_true",
        help="跳过标注文件下载",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="保留压缩包文件",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="仅显示下载信息，不实际下载",
    )
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("COCO 2017 数据集下载脚本")
    print("=" * 60)
    print(f"输出目录: {output_dir.absolute()}")
    print()
    
    # 确定要下载的内容
    to_download = []
    if not args.skip_train:
        to_download.append("train2017")
    if not args.skip_val:
        to_download.append("val2017")
    if not args.skip_annotations:
        to_download.append("annotations")
    
    if not to_download:
        print("没有需要下载的内容（所有选项都被跳过）")
        return
    
    # 显示下载计划
    print("下载计划:")
    for name in to_download:
        print(f"  - {name}: {FILE_SIZES.get(name, '未知大小')}")
    print()
    
    if args.dry_run:
        print("Dry-run 模式，不实际下载")
        return
    
    # 下载和解压
    for name in to_download:
        url = COCO_URLS[name]
        zip_path = output_dir / f"{name}.zip"
        
        # 检查是否已存在
        extract_dir = output_dir / name
        if name == "annotations":
            # 标注文件解压后是 annotations 目录
            extract_dir = output_dir / "annotations"
        
        if extract_dir.exists() and any(extract_dir.iterdir()):
            print(f"[跳过] {name} 已存在: {extract_dir}")
            print()
            continue
        
        # 下载
        success = download_file(url, zip_path, f"{name} ({FILE_SIZES.get(name, '')})")
        
        if not success:
            continue
        
        # 解压
        success = extract_zip(zip_path, output_dir)
        
        # 清理压缩包
        if success and not args.keep_zip:
            print(f"清理压缩包: {zip_path}")
            zip_path.unlink()
        
        print()
    
    # 显示最终目录结构
    print("=" * 60)
    print("下载完成！目录结构:")
    print("=" * 60)
    
    for item in output_dir.iterdir():
        if item.is_dir():
            file_count = len(list(item.iterdir()))
            print(f"  {item.name}/ ({file_count} 项)")
        else:
            print(f"  {item.name}")
    
    print()
    print("数据集准备完成，可以开始训练了！")


if __name__ == "__main__":
    main()
