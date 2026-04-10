# 轻量化视觉Transformer实时物体检测系统

[![Python 3.8+](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 1.12+](https://img.shields.io/badge/PyTorch-1.12%2B-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

基于轻量化视觉Transformer的实时物体检测系统，在保持高精度的同时实现 >30 FPS 的实时性能，适用于移动设备和边缘计算场景。

## 特性

- **轻量化骨干网络**: 支持 MobileViT-Small/Base、EfficientFormerV2 等轻量化架构
- **高效注意力机制**: 线性注意力、池化注意力，复杂度从 O(n²) 降至 O(n)
- **知识蒸馏框架**: 支持响应蒸馏、特征蒸馏、关系蒸馏等多种策略
- **模型优化**: 量化感知训练(QAT)、动态/静态量化、结构化剪枝、ONNX/TensorRT 导出
- **实时推理**: 支持图像、视频流实时检测，可视化结果

## 项目结构

```
Light Weight ViT/
├── configs/                    # 配置文件
│   ├── model/                  # 模型配置
│   ├── train/                  # 训练配置
│   └── deploy/                 # 部署配置
├── src/                        # 源代码
│   ├── models/                 # 模型定义 (backbone/neck/head)
│   ├── distillation/           # 知识蒸馏
│   ├── optimization/           # 模型优化 (量化/剪枝/导出)
│   ├── inference/              # 推理引擎
│   ├── data/                   # 数据处理
│   └── utils/                  # 工具函数
├── tools/                      # CLI 工具
│   ├── train.py                # 训练
│   ├── evaluate.py             # 评估
│   ├── export_model.py         # 导出
│   ├── quantize.py             # 量化
│   ├── prune.py                # 剪枝
│   └── demo.py                 # 演示
└── tests/                      # 单元测试
```

## 安装

```bash
# 克隆项目
git clone https://github.com/your-repo/lightweight-vit-detection.git
cd lightweight-vit-detection

# 创建虚拟环境
python -m venv .venv
.\.venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 安装项目
pip install -e .
```

## 项目流程

完整的模型开发流程：**训练 → 评估 → 优化 → 导出**

### Step 1: 准备数据集

下载 COCO 2017 数据集到 `data/coco/` 目录：

```powershell
python scripts/download_coco.py --output-dir data/coco
```

数据目录结构：
```
data/coco/
├── train2017/
├── val2017/
└── annotations/
    ├── instances_train2017.json
    └── instances_val2017.json
```

### Step 2: 蒸馏训练

使用知识蒸馏训练轻量化学生模型：

```powershell
# 开始训练 (默认50 epochs)
python tools/train.py --config configs/train/distillation.yaml

# 恢复训练
python tools/train.py --config configs/train/distillation.yaml --resume outputs/distillation/checkpoint_epoch_0010.pth
```

训练输出位于 `outputs/distillation/`：
- `best_model.pth` - 最佳模型
- `checkpoint_epoch_*.pth` - 周期性检查点
- `train.log` - 训练日志

### Step 3: 模型评估

评估模型精度：

```powershell
python tools/evaluate.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth
```

### Step 4: 模型量化 (可选)

量化可减小模型体积并加速推理：

```powershell
# 动态量化 (最快，无需校准数据)
python tools/quantize.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --method dynamic

# 静态量化 (精度更高，需校准数据)
python tools/quantize.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --method static --calibration-data data/coco/val2017

# 量化感知训练 QAT (精度最高)
python tools/quantize.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --method qat --train-data data/coco/train2017 --epochs 10
```

### Step 5: 模型剪枝 (可选)

剪枝可移除冗余参数：

```powershell
# L1 非结构化剪枝
python tools/prune.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --method l1_unstructured --ratio 0.3

# L1 结构化剪枝 (通道剪枝，适合硬件加速)
python tools/prune.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --method l1_structured --ratio 0.2

# 迭代剪枝 + 微调 (推荐，渐进式剪枝)
python tools/prune.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --method iterative --ratio 0.5 --iterations 5 --finetune-epochs 5 --train-data data/coco/train2017
```

### Step 6: 导出 ONNX

导出模型用于部署：

```powershell
# 导出 ONNX (推荐)
python tools/export_model.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --format onnx --simplify --verify

# 导出 TorchScript
python tools/export_model.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --format torchscript
```

### Step 7: 运行演示

```powershell
# 图像检测
python tools/demo.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --source path/to/image.jpg --show

# 视频检测
python tools/demo.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --source path/to/video.mp4 --output output.mp4

# 实时摄像头检测
python tools/demo.py --config configs/model/mobilevit_small.yaml --weights outputs/distillation/best_model.pth --source 0
```

## 模型配置

| 模型 | 参数量 | 模型大小 | mAP@0.5 | FPS |
|------|--------|----------|---------|-----|
| MobileViT-Small | ~5.6M | ~22 MB | >0.75 | >30 |
| EfficientFormerV2-S1 | ~6.8M | ~26 MB | >0.76 | >35 |

## 知识蒸馏配置

在 `configs/train/distillation.yaml` 中配置：

```yaml
distillation:
  strategies:
    response: { enabled: true, weight: 1.0, temperature: 4.0 }
    feature: { enabled: true, weight: 0.5, layers: [2, 3, 4] }
    relation: { enabled: true, weight: 0.3 }
  loss_weights:
    detection_loss: 1.0
    distill_loss: 1.0
```

## 性能基准

| 模型 | FPS | 大小 | mAP@0.5 |
|------|-----|------|---------|
| YOLOv5s | 140 | 28 MB | 0.68 |
| SSD-MobileNetV2 | 60 | 17 MB | 0.62 |
| EfficientDet-D0 | 55 | 15 MB | 0.67 |
| **MobileViT-S (Ours)** | **30+** | **<50 MB** | **>0.75** |

## 开发指南

```powershell
# 运行测试
pytest tests/ -v

# 代码格式化
black src/ tools/ tests/
```

## 参考文献

1. MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
2. EfficientFormer: Vision Transformers at MobileNet Speed
3. Knowledge Distillation: A Survey

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
