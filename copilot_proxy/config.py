"""Configuration management for Copilot proxy."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

CONFIG_DIR = Path.home() / ".copilot-proxy"
CONFIG_FILE = CONFIG_DIR / "config.json"


def is_first_run() -> bool:
    """Check if this is the first run (no config file exists)."""
    return not CONFIG_FILE.exists()


def get_config_dir() -> Path:
    """Get the configuration directory path."""
    return CONFIG_DIR


def get_config_file() -> Path:
    """Get the configuration file path."""
    return CONFIG_FILE


def ensure_config_dir() -> Path:
    """Ensure the configuration directory exists."""
    CONFIG_DIR.mkdir(exist_ok=True)
    return CONFIG_DIR


def load_config() -> Dict[str, Any]:
    """Load configuration from file.

    Returns:
        Dictionary containing configuration, or empty dict if file doesn't exist.
    """
    if not CONFIG_FILE.exists():
        return {}

    try:
        with open(CONFIG_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def save_config(config: Dict[str, Any]) -> None:
    """Save configuration to file.

    Args:
        config: Dictionary containing configuration to save.
    """
    ensure_config_dir()

    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2)


def get_api_key() -> Optional[str]:
    """Get API key from config file.

    Returns:
        API key if found in config, None otherwise.
    """
    config = load_config()
    api_key = config.get("api_key")
    if api_key:
        return api_key.strip()
    return None


def set_api_key(api_key: str) -> None:
    """Save API key to config file.

    Args:
        api_key: The API key to save.
    """
    config = load_config()
    config["api_key"] = api_key.strip()
    save_config(config)


def get_base_url() -> Optional[str]:
    """Get base URL from config file.

    Returns:
        Base URL if found in config, None otherwise.
    """
    config = load_config()
    base_url = config.get("base_url")
    if base_url:
        return base_url.strip()
    return None


def set_base_url(base_url: str) -> None:
    """Save base URL to config file.

    Args:
        base_url: The base URL to save.
    """
    config = load_config()
    config["base_url"] = base_url.strip()
    save_config(config)


def ensure_complete_config() -> None:
    """Ensure all configuration values exist in config, including defaults."""
    if not CONFIG_FILE.exists():
        return

    config = load_config()
    updated = False

    # Ensure base_url exists (save default if not set)
    if "base_url" not in config:
        config["base_url"] = "https://api.z.ai/api/coding/paas/v4"
        updated = True

    # Ensure context_length exists
    if "context_length" not in config:
        config["context_length"] = 128000
        updated = True

    # Ensure default host exists
    if "default_host" not in config:
        config["default_host"] = "127.0.0.1"
        updated = True

    # Ensure default port exists
    if "default_port" not in config:
        config["default_port"] = 11434
        updated = True

    if updated:
        save_config(config)


def get_context_length(model_name: Optional[str] = None) -> int:
    """Get context length from config file.

    Args:
        model_name: Optional model name to get model-specific context length.
                   If None, returns global default.

    Returns:
        Context length for the specified model, or global default (128000).
    """
    config = load_config()
    
    # If model_name is provided, check for model-specific setting
    if model_name:
        model_configs = config.get("models", {})
        model_config = model_configs.get(model_name, {})
        if "context_length" in model_config:
            return model_config["context_length"]
    
    # Fall back to global context_length
    return config.get("context_length", 128000)


def set_context_length(context_length: int, model_name: Optional[str] = None) -> None:
    """Save context length to config file.

    Args:
        context_length: The context length to save.
        model_name: Optional model name to set model-specific context length.
                   If None, sets global default.
    """
    config = load_config()
    
    if model_name:
        # Set model-specific context length
        if "models" not in config:
            config["models"] = {}
        if model_name not in config["models"]:
            config["models"][model_name] = {}
        config["models"][model_name]["context_length"] = context_length
    else:
        # Set global context length
        config["context_length"] = context_length
    
    save_config(config)


def get_model_name() -> Optional[str]:
    """Get model name from config file.

    Returns:
        Model name if found in config, None otherwise.
    """
    config = load_config()
    model_name = config.get("model_name")
    if model_name:
        return model_name.strip()
    return None


def set_model_name(model_name: str) -> None:
    """Save model name to config file.

    Args:
        model_name: The model name to save.
    """
    config = load_config()
    config["model_name"] = model_name.strip()
    save_config(config)


def get_temperature() -> Optional[float]:
    """Get temperature from config file.

    Returns:
        Temperature if found in config, None otherwise.
    """
    config = load_config()
    return config.get("temperature")


def set_temperature(temperature: float) -> None:
    """Save temperature to config file.

    Args:
        temperature: The temperature to save.
    """
    config = load_config()
    config["temperature"] = temperature
    save_config(config)


def get_max_output_tokens(model_name: Optional[str] = None) -> Optional[int]:
    """Get max output tokens from config file.

    Args:
        model_name: Optional model name to get model-specific max output tokens.
                   If None, returns global default.

    Returns:
        Max output tokens for the specified model, or None if not set.
    """
    config = load_config()
    
    # If model_name is provided, check for model-specific setting
    if model_name:
        model_configs = config.get("models", {})
        model_config = model_configs.get(model_name, {})
        if "max_output_tokens" in model_config:
            return model_config["max_output_tokens"]
    
    # Fall back to global max_output_tokens (may be None)
    return config.get("max_output_tokens")


def set_max_output_tokens(max_output_tokens: int, model_name: Optional[str] = None) -> None:
    """Save max output tokens to config file.

    Args:
        max_output_tokens: The max output tokens to save.
        model_name: Optional model name to set model-specific max output tokens.
                   If None, sets global default.
    """
    config = load_config()
    
    if model_name:
        # Set model-specific max output tokens
        if "models" not in config:
            config["models"] = {}
        if model_name not in config["models"]:
            config["models"][model_name] = {}
        config["models"][model_name]["max_output_tokens"] = max_output_tokens
    else:
        # Set global max output tokens
        config["max_output_tokens"] = max_output_tokens
    
    save_config(config)