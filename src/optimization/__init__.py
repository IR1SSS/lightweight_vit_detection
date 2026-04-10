"""
Model optimization modules for the Lightweight ViT Detection System.
"""

from .quantization import (
    QuantizationAwareTraining,
    calibrate_model,
    quantize_model_dynamic,
    quantize_model_static,
    # FX graph mode quantization (PyTorch 1.8+)
    get_vit_qconfig_mapping,
    quantize_fx_static,
    quantize_fx_qat,
    quantize_fx_dynamic,
    ViTQuantizer,
    FX_QUANT_AVAILABLE,
)
from .pruning import (
    ModelPruner,
    L1UnstructuredPruner,
    L1StructuredPruner,
    ChannelPruner,
    iterative_pruning,
    get_pruning_statistics,
)
from .export import (
    export_onnx,
    export_torchscript,
    export_tensorrt,
    optimize_onnx,
    verify_onnx,
)

__all__ = [
    # Quantization
    "QuantizationAwareTraining",
    "calibrate_model",
    "quantize_model_dynamic",
    "quantize_model_static",
    # FX Quantization
    "get_vit_qconfig_mapping",
    "quantize_fx_static",
    "quantize_fx_qat",
    "quantize_fx_dynamic",
    "ViTQuantizer",
    "FX_QUANT_AVAILABLE",
    # Pruning
    "ModelPruner",
    "L1UnstructuredPruner",
    "L1StructuredPruner",
    "ChannelPruner",
    "iterative_pruning",
    "get_pruning_statistics",
    # Export
    "export_onnx",
    "export_torchscript",
    "export_tensorrt",
    "optimize_onnx",
    "verify_onnx",
]
