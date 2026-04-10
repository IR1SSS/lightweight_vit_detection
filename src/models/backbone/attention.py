"""
Lightweight attention mechanisms for Vision Transformers.
Implements linear attention, pool attention, and efficient attention variants.
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(nn.Module):
    """
    Standard Multi-Head Self-Attention mechanism.
    Complexity: O(n^2) where n is sequence length.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        """
        Initialize multi-head attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias in QKV projection
            attn_drop: Attention dropout rate
            proj_drop: Output projection dropout rate
        """
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} not divisible by num_heads {num_heads}"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, N, C)
            mask: Attention mask (B, N, N)
            
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float("-inf"))
        
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class LinearAttention(nn.Module):
    """
    Linear Attention mechanism with O(n) complexity.
    Uses kernel-based approximation to avoid computing full attention matrix.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dim_head: int = 64,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize linear attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            dim_head: Dimension per head
            qkv_bias: Whether to use bias
            dropout: Dropout rate
        """
        super().__init__()
        inner_dim = num_heads * dim_head
        
        self.num_heads = num_heads
        self.dim_head = dim_head
        self.scale = dim_head ** -0.5
        
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=qkv_bias)
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, N, C)
            mask: Not used in linear attention
            
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape
        
        # Compute Q, K, V
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(
            lambda t: t.reshape(B, N, self.num_heads, self.dim_head).transpose(1, 2),
            qkv
        )
        
        # Apply ELU kernel function for linear attention
        q = F.elu(q) + 1
        k = F.elu(k) + 1
        
        # Compute attention: (B, H, D, N) @ (B, H, N, D) -> (B, H, D, D)
        kv = k.transpose(-2, -1) @ v
        qkv = q @ kv
        
        # Normalize
        k_sum = k.sum(dim=-2, keepdim=True).transpose(-2, -1)
        q_sum = q @ k_sum
        qkv = qkv / (q_sum + 1e-6)
        
        # Reshape output
        x = qkv.transpose(1, 2).reshape(B, N, -1)
        x = self.to_out(x)
        
        return x


class PoolAttention(nn.Module):
    """
    Pooling-based Attention mechanism from EfficientFormer.
    Uses average pooling to replace key-value computation, achieving O(n) complexity.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        pool_size: int = 3,
        stride: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialize pool attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias
            pool_size: Size of pooling window
            stride: Stride for pooling
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Q projection
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Pooling for K and V
        self.pool_k = nn.AvgPool2d(pool_size, stride=stride, padding=pool_size // 2)
        self.pool_v = nn.AvgPool2d(pool_size, stride=stride, padding=pool_size // 2)
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, N, C) or (B, C, H, W)
            H: Height (if x is flattened)
            W: Width (if x is flattened)
            
        Returns:
            Output tensor (B, N, C)
        """
        if H is None or W is None:
            # Assume square feature map
            B, N, C = x.shape
            H = W = int(math.sqrt(N))
        else:
            B, N, C = x.shape
        
        # Reshape to 2D feature map
        x_2d = x.transpose(1, 2).reshape(B, C, H, W)
        
        # Compute Q
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute K and V via pooling
        k = self.pool_k(x_2d).reshape(B, C, -1).transpose(1, 2)
        v = self.pool_v(x_2d).reshape(B, C, -1).transpose(1, 2)
        
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class EfficientAttention(nn.Module):
    """
    Efficient Attention with memory-efficient implementation.
    Combines the best of linear and pool attention.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.0,
        sr_ratio: int = 1,
    ):
        """
        Initialize efficient attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias
            dropout: Dropout rate
            sr_ratio: Spatial reduction ratio for K and V
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.sr_ratio = sr_ratio
        
        # Q, K, V projections
        self.to_q = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_k = nn.Linear(dim, dim, bias=qkv_bias)
        self.to_v = nn.Linear(dim, dim, bias=qkv_bias)
        
        # Spatial reduction
        if sr_ratio > 1:
            self.sr = nn.Sequential(
                nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio),
                nn.BatchNorm2d(dim),
            )
        
        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
    
    def forward(
        self,
        x: torch.Tensor,
        H: Optional[int] = None,
        W: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, N, C)
            H: Height of feature map
            W: Width of feature map
            
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape
        
        if H is None or W is None:
            H = W = int(math.sqrt(N))
        
        # Compute Q
        q = self.to_q(x).reshape(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute K and V with spatial reduction
        if self.sr_ratio > 1:
            x_2d = x.transpose(1, 2).reshape(B, C, H, W)
            x_sr = self.sr(x_2d).reshape(B, C, -1).transpose(1, 2)
            k = self.to_k(x_sr)
            v = self.to_v(x_sr)
        else:
            k = self.to_k(x)
            v = self.to_v(x)
        
        k = k.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        # Apply attention
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x


class FlashAttention(nn.Module):
    """
    Flash Attention wrapper (requires PyTorch 2.0+ with CUDA).
    Falls back to standard attention if not available.
    """
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = True,
        dropout: float = 0.0,
    ):
        """
        Initialize flash attention.
        
        Args:
            dim: Input dimension
            num_heads: Number of attention heads
            qkv_bias: Whether to use bias
            dropout: Dropout rate (ignored in flash attention)
        """
        super().__init__()
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout)
        
        # Check if scaled_dot_product_attention is available
        self.use_flash = hasattr(F, "scaled_dot_product_attention")
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, N, C)
            mask: Attention mask
            
        Returns:
            Output tensor (B, N, C)
        """
        B, N, C = x.shape
        
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        if self.use_flash and x.is_cuda:
            # Use flash attention
            x = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        else:
            # Standard attention
            attn = (q @ k.transpose(-2, -1)) * self.scale
            if mask is not None:
                attn = attn.masked_fill(mask == 0, float("-inf"))
            attn = attn.softmax(dim=-1)
            x = attn @ v
        
        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x
