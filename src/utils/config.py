"""
Configuration management module.
Handles YAML config loading, merging, and validation.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from omegaconf import DictConfig, OmegaConf


class Config:
    """
    Configuration class that wraps OmegaConf DictConfig.
    Provides easy access to nested configuration values.
    """
    
    def __init__(self, cfg: Optional[Union[Dict, DictConfig]] = None):
        """
        Initialize configuration.
        
        Args:
            cfg: Configuration dictionary or DictConfig
        """
        if cfg is None:
            self._cfg = OmegaConf.create()
        elif isinstance(cfg, dict):
            self._cfg = OmegaConf.create(cfg)
        elif isinstance(cfg, DictConfig):
            self._cfg = cfg
        else:
            raise TypeError(f"Unsupported config type: {type(cfg)}")
    
    def __getattr__(self, name: str) -> Any:
        """Allow attribute-style access to config values."""
        if name.startswith("_"):
            return super().__getattribute__(name)
        value = getattr(self._cfg, name, None)
        if isinstance(value, DictConfig):
            return Config(value)
        return value
    
    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access to config values."""
        value = self._cfg[key]
        if isinstance(value, DictConfig):
            return Config(value)
        return value
    
    def __contains__(self, key: str) -> bool:
        """Check if key exists in config."""
        return key in self._cfg
    
    def __repr__(self) -> str:
        return f"Config({self._cfg})"
    
    def keys(self):
        """Return config keys."""
        return self._cfg.keys()
    
    def values(self):
        """Return config values."""
        return self._cfg.values()
    
    def items(self):
        """Return config items."""
        return self._cfg.items()
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get config value with default."""
        value = OmegaConf.select(self._cfg, key, default=default)
        if isinstance(value, DictConfig):
            return Config(value)
        return value
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to plain dictionary."""
        return OmegaConf.to_container(self._cfg, resolve=True)
    
    @property
    def raw(self) -> DictConfig:
        """Get raw OmegaConf DictConfig."""
        return self._cfg


def load_config(
    config_path: Union[str, Path],
    *overrides: Union[str, Dict[str, Any]]
) -> Config:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML config file
        *overrides: Override values (strings in "key=value" format or dicts)
        
    Returns:
        Config object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = OmegaConf.load(f)
    
    # Apply overrides
    for override in overrides:
        if isinstance(override, str):
            # Parse "key=value" format
            if "=" in override:
                key, value = override.split("=", 1)
                OmegaConf.update(cfg, key, _parse_value(value))
        elif isinstance(override, dict):
            cfg = OmegaConf.merge(cfg, override)
    
    # Resolve interpolations
    OmegaConf.resolve(cfg)
    
    return Config(cfg)


def merge_configs(*configs: Union[Config, Dict, DictConfig]) -> Config:
    """
    Merge multiple configurations.
    Later configs override earlier ones.
    
    Args:
        *configs: Configurations to merge
        
    Returns:
        Merged Config object
    """
    merged = OmegaConf.create()
    
    for cfg in configs:
        if isinstance(cfg, Config):
            merged = OmegaConf.merge(merged, cfg.raw)
        elif isinstance(cfg, (dict, DictConfig)):
            merged = OmegaConf.merge(merged, cfg)
        else:
            raise TypeError(f"Unsupported config type: {type(cfg)}")
    
    return Config(merged)


def save_config(config: Config, save_path: Union[str, Path]) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Config object to save
        save_path: Path to save YAML file
    """
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(save_path, "w", encoding="utf-8") as f:
        OmegaConf.save(config.raw, f)


def _parse_value(value: str) -> Any:
    """
    Parse string value to appropriate Python type.
    
    Args:
        value: String value to parse
        
    Returns:
        Parsed value
    """
    # Boolean
    if value.lower() in ("true", "yes", "on"):
        return True
    if value.lower() in ("false", "no", "off"):
        return False
    
    # None
    if value.lower() in ("none", "null"):
        return None
    
    # Integer
    try:
        return int(value)
    except ValueError:
        pass
    
    # Float
    try:
        return float(value)
    except ValueError:
        pass
    
    # List (comma-separated)
    if "," in value and value.startswith("["):
        # Parse list format
        inner = value[1:-1].strip()
        return [_parse_value(v.strip()) for v in inner.split(",")]
    
    # String (default)
    return value


def validate_config(config: Config, required_keys: List[str]) -> bool:
    """
    Validate that config contains required keys.
    
    Args:
        config: Config to validate
        required_keys: List of required key paths (e.g., "model.backbone.name")
        
    Returns:
        True if all required keys exist
        
    Raises:
        ValueError: If required key is missing
    """
    missing = []
    for key in required_keys:
        if config.get(key) is None:
            missing.append(key)
    
    if missing:
        raise ValueError(f"Missing required config keys: {missing}")
    
    return True
