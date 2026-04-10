"""
Quantization utilities for model optimization.
Implements quantization-aware training and post-training quantization.

Supports:
- Dynamic quantization (post-training)
- Static quantization with calibration (post-training)
- Quantization-aware training (QAT)
- FX graph mode quantization (PyTorch 1.8+)
"""

from typing import Any, Callable, Dict, List, Optional, Union

import torch
import torch.nn as nn

# PyTorch量化API在不同版本中位置不同
# PyTorch 2.0+: torch.ao.quantization
# PyTorch < 2.0: torch.quantization
try:
    from torch.ao.quantization import (
        quantize_dynamic,
        QConfig,
        default_qconfig,
        default_dynamic_qconfig,
        get_default_qconfig,
        get_default_qat_qconfig,
        MinMaxObserver,
        MovingAverageMinMaxObserver,
        PerChannelMinMaxObserver,
    )
    from torch.ao.quantization.quantize_fx import (
        prepare_fx,
        convert_fx,
        prepare_qat_fx,
    )
    from torch.ao.quantization import get_default_qconfig_mapping, get_default_qat_qconfig_mapping
    FX_QUANT_AVAILABLE = True
except ImportError:
    try:
        from torch.ao.quantization import (
            quantize_dynamic,
            QConfig,
            default_qconfig,
            default_dynamic_qconfig,
            get_default_qconfig,
            get_default_qat_qconfig,
            MinMaxObserver,
            MovingAverageMinMaxObserver,
            PerChannelMinMaxObserver,
        )
        from torch.ao.quantization.fx import prepare_fx, convert_fx, prepare_qat_fx
        from torch.ao.quantization import get_default_qconfig_mapping, get_default_qat_qconfig_mapping
        FX_QUANT_AVAILABLE = True
    except ImportError:
        from torch.quantization import (
            quantize_dynamic,
            QConfig,
            default_qconfig,
            default_dynamic_qconfig,
            get_default_qconfig,
            get_default_qat_qconfig,
            MinMaxObserver,
            MovingAverageMinMaxObserver,
            PerChannelMinMaxObserver,
        )
        prepare_fx = None
        convert_fx = None
        prepare_qat_fx = None
        get_default_qconfig_mapping = None
        get_default_qat_qconfig_mapping = None
        FX_QUANT_AVAILABLE = False


class QuantizationAwareTraining:
    """
    Quantization-Aware Training (QAT) wrapper.
    
    Prepares model for QAT and handles the quantization workflow.
    """
    
    def __init__(
        self,
        model: nn.Module,
        qconfig: Optional[QConfig] = None,
        backend: str = "fbgemm",
    ):
        """
        Initialize QAT wrapper.
        
        Args:
            model: Model to quantize
            qconfig: Quantization configuration
            backend: Quantization backend ("fbgemm" for x86, "qnnpack" for ARM)
        """
        self.model = model
        self.backend = backend
        self.qconfig = qconfig or get_default_qat_qconfig(backend)
        self.prepared_model = None
        self.quantized_model = None
        
    def prepare(self) -> nn.Module:
        """
        Prepare model for QAT by inserting fake quantization modules.
        
        Returns:
            Model prepared for QAT
        """
        # Set qconfig
        self.model.qconfig = self.qconfig
        
        # Prepare for QAT
        self.prepared_model = torch.quantization.prepare_qat(
            self.model,
            inplace=False,
        )
        
        return self.prepared_model
    
    def convert(self) -> nn.Module:
        """
        Convert prepared model to quantized model.
        
        Returns:
            Quantized model
        """
        if self.prepared_model is None:
            raise RuntimeError("Model must be prepared before conversion")
        
        # Set to eval mode for conversion
        self.prepared_model.eval()
        
        # Convert to quantized model
        self.quantized_model = torch.quantization.convert(
            self.prepared_model,
            inplace=False,
        )
        
        return self.quantized_model
    
    def get_prepared_model(self) -> nn.Module:
        """Get the prepared model."""
        return self.prepared_model
    
    def get_quantized_model(self) -> nn.Module:
        """Get the quantized model."""
        return self.quantized_model


def quantize_model_dynamic(
    model: nn.Module,
    dtype: torch.dtype = torch.qint8,
) -> nn.Module:
    """
    Apply dynamic quantization to model.
    
    Args:
        model: Model to quantize
        dtype: Target data type
        
    Returns:
        Quantized model
    """
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=dtype,
    )
    return quantized_model


def calibrate_model(
    model: nn.Module,
    calibration_loader,
    prepare_fn: Optional[Callable] = None,
    backend: str = "fbgemm",
) -> nn.Module:
    """
    Calibrate model for static quantization.
    
    Args:
        model: Model to calibrate
        calibration_loader: Data loader for calibration
        prepare_fn: Custom prepare function
        backend: Quantization backend
        
    Returns:
        Calibrated model ready for conversion
    """
    # Set backend
    torch.backends.quantized.engine = backend
    
    # Prepare model
    model.eval()
    model.qconfig = get_default_qconfig(backend)
    
    if prepare_fn is not None:
        model = prepare_fn(model)
    else:
        model = torch.quantization.prepare(model, inplace=False)
    
    # Run calibration
    with torch.no_grad():
        for batch in calibration_loader:
            if isinstance(batch, dict):
                images = batch.get("images", batch.get("image"))
            else:
                images = batch[0]
            
            if isinstance(images, torch.Tensor):
                model(images)
    
    return model


def quantize_model_static(
    model: nn.Module,
    calibration_loader,
    backend: str = "fbgemm",
) -> nn.Module:
    """
    Apply static quantization with calibration.
    
    Args:
        model: Model to quantize
        calibration_loader: Data loader for calibration
        backend: Quantization backend
        
    Returns:
        Quantized model
    """
    # Calibrate
    model = calibrate_model(model, calibration_loader, backend=backend)
    
    # Convert
    quantized_model = torch.quantization.convert(model, inplace=False)
    
    return quantized_model


class QuantStubWrapper(nn.Module):
    """
    Wrapper to add QuantStub and DeQuantStub to a model.
    """
    
    def __init__(self, model: nn.Module):
        """
        Initialize wrapper.
        
        Args:
            model: Model to wrap
        """
        super().__init__()
        self.quant = torch.quantization.QuantStub()
        self.model = model
        self.dequant = torch.quantization.DeQuantStub()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = self.quant(x)
        x = self.model(x)
        x = self.dequant(x)
        return x


class FusedModuleWrapper:
    """
    Utility to fuse modules for quantization.
    """
    
    @staticmethod
    def fuse_conv_bn(
        model: nn.Module,
        inplace: bool = True,
    ) -> nn.Module:
        """
        Fuse Conv2d and BatchNorm2d layers.
        
        Args:
            model: Model to fuse
            inplace: Whether to modify in place
            
        Returns:
            Model with fused layers
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)
        
        torch.quantization.fuse_modules(model, [], inplace=True)
        return model
    
    @staticmethod
    def fuse_conv_bn_relu(
        model: nn.Module,
        inplace: bool = True,
    ) -> nn.Module:
        """
        Fuse Conv2d, BatchNorm2d, and ReLU layers.
        
        Args:
            model: Model to fuse
            inplace: Whether to modify in place
            
        Returns:
            Model with fused layers
        """
        if not inplace:
            import copy
            model = copy.deepcopy(model)
        
        # Find and fuse patterns
        for name, module in model.named_modules():
            if isinstance(module, nn.Sequential):
                if len(module) >= 3:
                    if (isinstance(module[0], nn.Conv2d) and
                        isinstance(module[1], nn.BatchNorm2d) and
                        isinstance(module[2], (nn.ReLU, nn.ReLU6))):
                        torch.quantization.fuse_modules(
                            model, [f"{name}.0", f"{name}.1", f"{name}.2"],
                            inplace=True
                        )
        
        return model


# ============================================================================
# FX Graph Mode Quantization (PyTorch 1.8+)
# ============================================================================

def get_vit_qconfig_mapping(backend: str = "fbgemm"):
    """
    Get quantization config mapping optimized for ViT models.
    
    ViT models use LayerNorm instead of BatchNorm, requiring special handling.
    
    Args:
        backend: Quantization backend ("fbgemm" or "qnnpack")
        
    Returns:
        QConfigMapping for ViT models
    """
    if not FX_QUANT_AVAILABLE:
        raise RuntimeError("FX quantization is not available in this PyTorch version")
    
    # Get default mapping
    qconfig_mapping = get_default_qconfig_mapping(backend)
    
    # Custom qconfig for attention layers (per-channel quantization for better accuracy)
    attention_qconfig = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
        weight=PerChannelMinMaxObserver.with_args(qscheme=torch.per_channel_symmetric),
    )
    
    # Custom qconfig for LayerNorm (more sensitive to quantization)
    layernorm_qconfig = QConfig(
        activation=MovingAverageMinMaxObserver.with_args(qscheme=torch.per_tensor_affine),
        weight=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric),
    )
    
    # Apply custom configs for ViT-specific layers
    for name in ["attention", "attn", "qkv", "proj", "norm", "layernorm", "layer_norm"]:
        qconfig_mapping.set_module_name(name, attention_qconfig if "attn" in name else layernorm_qconfig)
    
    return qconfig_mapping


def quantize_fx_static(
    model: nn.Module,
    calibration_loader,
    backend: str = "fbgemm",
    input_shape: tuple = (1, 3, 320, 320),
) -> nn.Module:
    """
    Apply FX graph mode static quantization.
    
    This method is recommended for ViT models as it handles:
    - Dynamic control flow
    - Complex module hierarchies
    - LayerNorm quantization
    
    Args:
        model: Model to quantize
        calibration_loader: Calibration data loader
        backend: Quantization backend
        input_shape: Input shape for tracing
        
    Returns:
        Quantized model
    """
    if not FX_QUANT_AVAILABLE:
        raise RuntimeError("FX quantization requires PyTorch 1.8+")
    
    model.eval()
    
    # Get ViT-optimized config mapping
    qconfig_mapping = get_vit_qconfig_mapping(backend)
    
    # Create example input
    example_input = torch.randn(*input_shape)
    
    # Prepare model
    prepared_model = prepare_fx(model, qconfig_mapping, example_input)
    
    # Run calibration
    print("Running calibration...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(calibration_loader):
            if isinstance(batch, dict):
                images = batch.get("images", batch.get("image"))
            else:
                images = batch[0]
            
            if isinstance(images, torch.Tensor):
                prepared_model(images)
            
            if batch_idx % 50 == 0:
                print(f"  Calibration batch {batch_idx}")
    
    # Convert to quantized model
    quantized_model = convert_fx(prepared_model)
    
    return quantized_model


def quantize_fx_qat(
    model: nn.Module,
    train_loader,
    epochs: int,
    lr: float = 1e-5,
    backend: str = "fbgemm",
    input_shape: tuple = (1, 3, 320, 320),
    device: str = "cpu",
) -> nn.Module:
    """
    Apply FX graph mode quantization-aware training.
    
    This is the recommended approach for quantizing ViT models as it:
    - Maintains accuracy through fine-tuning
    - Handles complex model architectures
    - Supports custom quantization configs
    
    Args:
        model: Model to quantize
        train_loader: Training data loader
        epochs: Number of training epochs
        lr: Learning rate
        backend: Quantization backend
        input_shape: Input shape for tracing
        device: Device to use
        
    Returns:
        Quantized model
    """
    if not FX_QUANT_AVAILABLE:
        raise RuntimeError("FX quantization requires PyTorch 1.8+")
    
    # Get ViT-optimized config mapping
    qconfig_mapping = get_vit_qat_qconfig_mapping(backend)
    
    # Create example input
    example_input = torch.randn(*input_shape)
    
    # Prepare model for QAT
    prepared_model = prepare_qat_fx(model, qconfig_mapping, example_input)
    prepared_model.to(device)
    prepared_model.train()
    
    # Setup optimizer
    optimizer = torch.optim.AdamW(prepared_model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Training loop
    print("Starting QAT training...")
    for epoch in range(epochs):
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(train_loader):
            if isinstance(batch, dict):
                images = batch.get("images", batch.get("image"))
            else:
                images = batch[0]
            
            images = images.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            outputs = prepared_model(images)
            
            # Compute loss (simplified)
            if isinstance(outputs, tuple):
                loss = sum(o.mean() for o in outputs if isinstance(o, torch.Tensor) and o.requires_grad)
            else:
                loss = outputs.mean() if isinstance(outputs, torch.Tensor) else torch.tensor(0.0, requires_grad=True)
            
            if loss.requires_grad:
                loss.backward()
                optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        scheduler.step()
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
    
    # Convert to quantized model
    prepared_model.eval()
    quantized_model = convert_fx(prepared_model)
    
    return quantized_model


def quantize_fx_dynamic(
    model: nn.Module,
    backend: str = "fbgemm",
    input_shape: tuple = (1, 3, 320, 320),
) -> nn.Module:
    """
    Apply FX graph mode dynamic quantization.
    
    Simpler than static quantization, suitable for:
    - Quick quantization without calibration
    - Models where static quantization is challenging
    - Deployment scenarios with varying input distributions
    
    Args:
        model: Model to quantize
        backend: Quantization backend
        input_shape: Input shape for tracing
        
    Returns:
        Quantized model
    """
    if not FX_QUANT_AVAILABLE:
        raise RuntimeError("FX quantization requires PyTorch 1.8+")
    
    model.eval()
    
    # For dynamic quantization, we use the standard API
    quantized_model = torch.ao.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8,
    )
    
    return quantized_model


class ViTQuantizer:
    """
    Specialized quantizer for Vision Transformer models.
    
    Handles the unique challenges of quantizing ViT models:
    - LayerNorm instead of BatchNorm
    - Attention mechanisms
    - Patch embedding layers
    """
    
    def __init__(
        self,
        model: nn.Module,
        backend: str = "fbgemm",
        input_shape: tuple = (1, 3, 320, 320),
    ):
        """
        Initialize ViT quantizer.
        
        Args:
            model: ViT model to quantize
            backend: Quantization backend
            input_shape: Input shape for tracing
        """
        self.model = model
        self.backend = backend
        self.input_shape = input_shape
        self.prepared_model = None
        self.quantized_model = None
        
    def prepare_static(self) -> nn.Module:
        """Prepare model for static quantization."""
        if not FX_QUANT_AVAILABLE:
            raise RuntimeError("FX quantization requires PyTorch 1.8+")
        
        self.model.eval()
        qconfig_mapping = get_vit_qconfig_mapping(self.backend)
        example_input = torch.randn(*self.input_shape)
        self.prepared_model = prepare_fx(self.model, qconfig_mapping, example_input)
        return self.prepared_model
    
    def prepare_qat(self) -> nn.Module:
        """Prepare model for QAT."""
        if not FX_QUANT_AVAILABLE:
            raise RuntimeError("FX quantization requires PyTorch 1.8+")
        
        qconfig_mapping = get_vit_qat_qconfig_mapping(self.backend)
        example_input = torch.randn(*self.input_shape)
        self.prepared_model = prepare_qat_fx(self.model, qconfig_mapping, example_input)
        return self.prepared_model
    
    def calibrate(self, calibration_loader) -> None:
        """Run calibration on prepared model."""
        if self.prepared_model is None:
            raise RuntimeError("Model must be prepared first")
        
        self.prepared_model.eval()
        with torch.no_grad():
            for batch in calibration_loader:
                if isinstance(batch, dict):
                    images = batch.get("images", batch.get("image"))
                else:
                    images = batch[0]
                self.prepared_model(images)
    
    def convert(self) -> nn.Module:
        """Convert prepared model to quantized model."""
        if self.prepared_model is None:
            raise RuntimeError("Model must be prepared first")
        
        self.prepared_model.eval()
        self.quantized_model = convert_fx(self.prepared_model)
        return self.quantized_model
    
    def get_quantized_model(self) -> Optional[nn.Module]:
        """Get the quantized model."""
        return self.quantized_model
