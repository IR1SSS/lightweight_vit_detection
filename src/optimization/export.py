"""
Model export utilities for deployment.
Supports ONNX, TorchScript, and TensorRT export.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn


def export_onnx(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int] = (3, 320, 320),
    batch_size: int = 1,
    opset_version: int = 14,
    dynamic_axes: Optional[Dict] = None,
    input_names: List[str] = ["input"],
    output_names: List[str] = ["boxes", "scores", "labels"],
    simplify: bool = True,
    **kwargs,
) -> str:
    """
    Export model to ONNX format.
    
    Args:
        model: Model to export
        output_path: Output file path
        input_shape: Input shape (C, H, W)
        batch_size: Batch size
        opset_version: ONNX opset version
        dynamic_axes: Dynamic axes configuration
        input_names: Input tensor names
        output_names: Output tensor names
        simplify: Simplify ONNX model
        **kwargs: Additional export arguments
        
    Returns:
        Path to exported model
    """
    model.eval()
    
    # Move model to CPU for ONNX export (ensures portability)
    model = model.cpu()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape)
    
    # Default dynamic axes
    if dynamic_axes is None:
        dynamic_axes = {
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "boxes": {0: "batch_size"},
            "scores": {0: "batch_size"},
            "labels": {0: "batch_size"},
        }
    
    # Export (use dynamo=False for better compatibility with complex models)
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        opset_version=opset_version,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,  # Use legacy exporter for better compatibility
        **kwargs,
    )
    
    # Simplify if requested
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            onnx_model = onnx.load(output_path)
            onnx_model, _ = onnx_simplify(onnx_model)
            onnx.save(onnx_model, output_path)
            print(f"Simplified ONNX model saved to {output_path}")
        except ImportError:
            print("onnx-simplifier not available, skipping simplification")
    
    return output_path


def export_torchscript(
    model: nn.Module,
    output_path: str,
    input_shape: Tuple[int, int, int] = (3, 320, 320),
    batch_size: int = 1,
    method: str = "trace",
    **kwargs,
) -> str:
    """
    Export model to TorchScript format.
    
    Args:
        model: Model to export
        output_path: Output file path
        input_shape: Input shape (C, H, W)
        batch_size: Batch size
        method: Export method ("trace" or "script")
        **kwargs: Additional arguments
        
    Returns:
        Path to exported model
    """
    model.eval()
    
    # Move model to CPU for TorchScript export (ensures portability)
    model = model.cpu()
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, *input_shape)
    
    # Export
    if method == "trace":
        scripted_model = torch.jit.trace(model, dummy_input, **kwargs)
    elif method == "script":
        scripted_model = torch.jit.script(model, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Save
    scripted_model.save(output_path)
    print(f"TorchScript model saved to {output_path}")
    
    return output_path


def export_tensorrt(
    onnx_path: str,
    output_path: str,
    precision: str = "fp16",
    max_batch_size: int = 1,
    max_workspace_size: int = 1 << 30,  # 1GB
    **kwargs,
) -> str:
    """
    Convert ONNX model to TensorRT engine.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output path for TensorRT engine
        precision: Precision ("fp32", "fp16", "int8")
        max_batch_size: Maximum batch size
        max_workspace_size: Maximum workspace size in bytes
        **kwargs: Additional arguments
        
    Returns:
        Path to TensorRT engine
    """
    try:
        import tensorrt as trt
        import pycuda.driver as cuda
    except ImportError:
        raise ImportError("TensorRT and PyCUDA required for TensorRT export")
    
    # Create logger
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    
    # Create builder
    builder = trt.Builder(TRT_LOGGER)
    builder.max_batch_size = max_batch_size
    builder.max_workspace_size = max_workspace_size
    
    # Set precision flags
    if precision == "fp16":
        builder.fp16_mode = True
    elif precision == "int8":
        builder.int8_mode = True
        # Need calibration dataset for INT8
        if "calibration_cache" not in kwargs:
            print("Warning: INT8 mode requires calibration data")
    
    # Parse ONNX
    network = builder.create_network()
    parser = trt.OnnxParser(network, TRT_LOGGER)
    
    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            raise RuntimeError("Failed to parse ONNX model")
    
    # Build engine
    engine = builder.build_cuda_engine(network)
    
    # Save engine
    with open(output_path, "wb") as f:
        f.write(engine.serialize())
    
    print(f"TensorRT engine saved to {output_path}")
    
    return output_path


def optimize_onnx(
    onnx_path: str,
    output_path: Optional[str] = None,
    optimizations: Optional[List[str]] = None,
) -> str:
    """
    Optimize ONNX model.
    
    Args:
        onnx_path: Path to ONNX model
        output_path: Output path (default: overwrite input)
        optimizations: List of optimizations to apply
        
    Returns:
        Path to optimized model
    """
    import onnx
    from onnx import optimizer
    
    # Load model
    model = onnx.load(onnx_path)
    
    # Get available passes
    all_passes = optimizer.get_available_passes()
    
    # Default optimizations
    if optimizations is None:
        optimizations = [
            "eliminate_identity",
            "eliminate_nop_transpose",
            "fuse_add_bias_into_conv",
            "fuse_bn_into_conv",
            "fuse_consecutive_concats",
            "fuse_consecutive_reduce_unsqueeze",
            "fuse_consecutive_squeezes",
            "fuse_consecutive_transposes",
            "fuse_pad_into_conv",
            "fuse_transpose_into_gemm",
        ]
    
    # Apply optimizations
    for opt in optimizations:
        if opt in all_passes:
            model = optimizer.optimize(model, [opt])
            print(f"Applied optimization: {opt}")
        else:
            print(f"Optimization not available: {opt}")
    
    # Save
    if output_path is None:
        output_path = onnx_path
    
    onnx.save(model, output_path)
    print(f"Optimized model saved to {output_path}")
    
    return output_path


def verify_onnx(
    onnx_path: str,
    model: nn.Module,
    input_shape: Tuple[int, int, int] = (3, 320, 320),
    batch_size: int = 1,
    rtol: float = 1e-3,
    atol: float = 1e-5,
) -> bool:
    """
    Verify ONNX model produces same outputs as PyTorch model.
    
    Args:
        onnx_path: Path to ONNX model
        model: Original PyTorch model
        input_shape: Input shape for testing
        batch_size: Batch size for testing
        rtol: Relative tolerance
        atol: Absolute tolerance
        
    Returns:
        True if outputs match
    """
    import onnxruntime as ort
    import numpy as np
    
    # Move model to CPU for verification (ONNX runs on CPU)
    model = model.cpu()
    
    # Create test input
    dummy_input = torch.randn(batch_size, *input_shape)
    
    # PyTorch inference
    model.eval()
    with torch.no_grad():
        torch_output = model(dummy_input)
    
    # ONNX inference
    session = ort.InferenceSession(onnx_path)
    input_name = session.get_inputs()[0].name
    onnx_output = session.run(None, {input_name: dummy_input.numpy()})
    
    # Compare outputs
    if isinstance(torch_output, tuple):
        torch_output = torch_output[0]
    
    onnx_output = onnx_output[0]
    torch_output = torch_output.numpy()
    
    # Check shape
    if torch_output.shape != onnx_output.shape:
        print(f"Shape mismatch: PyTorch {torch_output.shape} vs ONNX {onnx_output.shape}")
        return False
    
    # Check values
    if not np.allclose(torch_output, onnx_output, rtol=rtol, atol=atol):
        max_diff = np.abs(torch_output - onnx_output).max()
        print(f"Output mismatch: max difference = {max_diff}")
        return False
    
    print("ONNX model verification passed!")
    return True


class ModelExporter:
    """
    Unified model exporter supporting multiple formats.
    """
    
    def __init__(
        self,
        model: nn.Module,
        input_shape: Tuple[int, int, int] = (3, 320, 320),
    ):
        """
        Initialize exporter.
        
        Args:
            model: Model to export
            input_shape: Input shape (C, H, W)
        """
        self.model = model
        self.input_shape = input_shape
    
    def export_all(
        self,
        output_dir: str,
        formats: List[str] = ["onnx", "torchscript"],
        **kwargs,
    ) -> Dict[str, str]:
        """
        Export model to multiple formats.
        
        Args:
            output_dir: Output directory
            formats: List of formats to export
            **kwargs: Additional arguments
            
        Returns:
            Dictionary mapping format to output path
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        outputs = {}
        
        for fmt in formats:
            if fmt == "onnx":
                path = output_dir / "model.onnx"
                outputs[fmt] = export_onnx(self.model, str(path), self.input_shape, **kwargs)
            elif fmt == "torchscript":
                path = output_dir / "model.pt"
                outputs[fmt] = export_torchscript(self.model, str(path), input_shape=self.input_shape, **kwargs)
            else:
                print(f"Unknown format: {fmt}")
        
        return outputs
