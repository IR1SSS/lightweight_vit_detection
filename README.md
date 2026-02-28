# Lightweight Vision Transformer for Real-time Object Detection

轻量化视觉Transformer实时物体检测系统

## 项目简介

本项目实现了一个基于轻量化Vision Transformer (ViT)的实时物体检测系统，适用于移动端和边缘设备部署。主要特点：

- **轻量化设计**: 基于MobileViT架构，结合CNN和Transformer的优势
- **高效推理**: 目标在RTX 4070达到>30 FPS
- **知识蒸馏**: 支持从大模型到小模型的知识迁移
- **多后端部署**: 支持PyTorch、ONNX、TensorRT等多种部署方式

### 性能目标

| 指标 | 目标值 |
|------|--------|
| mAP@0.5 | ≥ 0.75 |
| 推理速度 | > 30 FPS (RTX 3060) |
| 模型大小 | < 50 MB |

## 安装

### 环境要求

- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (GPU推理)

### 安装步骤

```bash
# 克隆项目
git clone https://github.com/yourusername/lightweight-vit-detection.git
cd lightweight-vit-detection

# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 安装项目（开发模式）
pip install -e .
```

### 安装开发依赖

```bash
pip install -e ".[dev]"
```

## 快速开始


### 1. 训练模型

```bash
# 基础训练
python scripts/train.py --config configs/model/mobilevit.yaml  --epochs 300

# 使用知识蒸馏
python scripts/distill.py --config configs/training/distillation.yaml
```

### 2. 评估模型

```bash
python scripts/evaluate.py \
    --model outputs/best_model.pth \
    --config configs/model/mobilevit.yaml
```


## 项目结构

```
lightweight_vit_detection/
├── configs/                    # 配置文件
│   ├── model/                  # 模型配置
│   ├── training/               # 训练配置
│   └── deployment/             # 部署配置
├── src/                        # 源代码
│   ├── models/                 # 模型定义
│   ├── distillation/           # 知识蒸馏
│   ├── data/                   # 数据加载
│   ├── training/               # 训练相关
│   ├── quantization/           # 量化
│   ├── deployment/             # 部署
│   ├── utils/                  # 工具函数
│   └── applications/           # 应用示例
├── scripts/                    # 脚本
│   ├── train.py                # 训练脚本
│   ├── distill.py              # 蒸馏脚本
│   ├── inference.py            # 推理脚本
│   └── evaluate.py             # 评估脚本
├── tests/                      # 测试
├── data/                       # 数据集目录
├── models/                     # 预训练模型
├── outputs/                    # 输出目录
└── docs/                       # 文档
```

## 数据准备

### COCO数据集

下载COCO数据集并按以下结构组织：

```
data/
└── coco/
    ├── train2017/
    ├── val2017/
    └── annotations/
        ├── instances_train2017.json
        └── instances_val2017.json
```

## 配置说明

主要配置文件示例 (`configs/model/mobilevit.yaml`):

```yaml
model:
  name: "mobilevit_detection"
  input:
    image_size: [640, 640]
  backbone:
    name: "mobilevit_s"
  head:
    type: "retina"
    num_classes: 80
```

## 测试

```bash
# 运行所有测试
pytest tests/

# 运行特定测试
pytest tests/unit/test_models.py -v
```

## 导出模型

```bash
# 导出ONNX
python -c "
from src.deployment import export_to_onnx
from src.models import build_mobilevit
from src.utils import load_config

config = load_config('configs/model/mobilevit.yaml')
model = build_mobilevit(config)
export_to_onnx(model, 'outputs/model.onnx')
"
```

## 技术栈

- **深度学习框架**: PyTorch 1.12+
- **计算机视觉**: OpenCV 4.5+, Pillow
- **优化工具**: ONNX Runtime, TensorRT 8.0+
- **评估工具**: pycocotools

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request!

## 参考文献

1. MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer
2. EfficientFormer: Vision Transformers at MobileNet Speed
3. Knowledge Distillation: A Survey
