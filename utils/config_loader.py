"""
Configuration Loader for Lightweight ViT Detection System.

This module provides utilities for loading and managing YAML configuration files.
"""

import os
from typing import Dict, Any, Optional, List
from copy import deepcopy

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to YAML configuration file.
        
    Returns:
        Configuration dictionary.
        
    Raises:
        FileNotFoundError: If config file doesn't exist.
        yaml.YAMLError: If YAML parsing fails.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    if config is None:
        config = {}
        
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """
    Save configuration to YAML file.
    
    Args:
        config: Configuration dictionary to save.
        save_path: Path to save YAML file.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    with open(save_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)


def merge_configs(
    base_config: Dict[str, Any],
    override_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Override values take precedence over base values.
    
    Args:
        base_config: Base configuration dictionary.
        override_config: Override configuration dictionary.
        
    Returns:
        Merged configuration dictionary.
    """
    result = deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_configs(result[key], value)
        else:
            result[key] = deepcopy(value)
            
    return result


def load_config_with_defaults(
    config_path: str,
    default_config_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Load configuration with default values.
    
    Merges user config with default config, where user values override defaults.
    
    Args:
        config_path: Path to user configuration file.
        default_config_path: Optional path to default configuration file.
        
    Returns:
        Merged configuration dictionary.
    """
    config = load_config(config_path)
    
    if default_config_path is not None and os.path.exists(default_config_path):
        default_config = load_config(default_config_path)
        config = merge_configs(default_config, config)
        
    return config


def get_nested_value(
    config: Dict[str, Any],
    key_path: str,
    default: Any = None
) -> Any:
    """
    Get nested value from configuration using dot notation.
    
    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to value (e.g., 'model.backbone.name').
        default: Default value if path doesn't exist.
        
    Returns:
        Value at key path or default.
        
    Example:
        >>> config = {'model': {'backbone': {'name': 'mobilevit'}}}
        >>> get_nested_value(config, 'model.backbone.name')
        'mobilevit'
    """
    keys = key_path.split('.')
    value = config
    
    for key in keys:
        if isinstance(value, dict) and key in value:
            value = value[key]
        else:
            return default
            
    return value


def set_nested_value(
    config: Dict[str, Any],
    key_path: str,
    value: Any
) -> Dict[str, Any]:
    """
    Set nested value in configuration using dot notation.
    
    Args:
        config: Configuration dictionary.
        key_path: Dot-separated path to value.
        value: Value to set.
        
    Returns:
        Modified configuration dictionary.
    """
    keys = key_path.split('.')
    current = config
    
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
        
    current[keys[-1]] = value
    
    return config


class ConfigManager:
    """
    Configuration manager for handling multiple configs.
    
    Provides convenient access to configuration values and
    supports config inheritance and overrides.
    """
    
    def __init__(
        self,
        config_dir: str = 'configs',
        default_config: Optional[str] = None
    ) -> None:
        """
        Initialize config manager.
        
        Args:
            config_dir: Base directory for configuration files.
            default_config: Optional default configuration file.
        """
        self.config_dir = config_dir
        self.configs: Dict[str, Dict[str, Any]] = {}
        
        if default_config is not None:
            self.load('default', default_config)
            
    def load(self, name: str, config_path: str) -> Dict[str, Any]:
        """
        Load configuration by name.
        
        Args:
            name: Name to reference this config.
            config_path: Path to configuration file.
            
        Returns:
            Loaded configuration.
        """
        full_path = os.path.join(self.config_dir, config_path)
        
        if not os.path.exists(full_path):
            full_path = config_path
            
        config = load_config(full_path)
        self.configs[name] = config
        
        return config
        
    def get(self, name: str, key_path: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration or value by name.
        
        Args:
            name: Configuration name.
            key_path: Optional dot-separated path to specific value.
            default: Default value if not found.
            
        Returns:
            Configuration or specific value.
        """
        if name not in self.configs:
            return default
            
        config = self.configs[name]
        
        if key_path is None:
            return config
            
        return get_nested_value(config, key_path, default)
        
    def merge(self, name: str, override_name: str) -> Dict[str, Any]:
        """
        Merge two configurations.
        
        Args:
            name: Base configuration name.
            override_name: Override configuration name.
            
        Returns:
            Merged configuration.
        """
        base = self.configs.get(name, {})
        override = self.configs.get(override_name, {})
        
        return merge_configs(base, override)
