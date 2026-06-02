# Copyright (c) 2026
# Licensed under the MIT License

"""Configuration management for qlib_data."""

import os
from pathlib import Path
from typing import Optional, Dict, Any

from .logging import get_logger


logger = get_logger(__name__)


class Config:
    """Global configuration singleton for qlib_data."""

    _instance = None
    _config: Dict[str, Any] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._config = {
                "provider_uri": None,
                "region": "CN",
                "auto_mmap": True,
                "mem_cache_size": "6GB",
            }
            logger.debug("config.singleton_created", defaults=list(cls._instance._config))
        return cls._instance

    def __getitem__(self, key: str) -> Any:
        return self._config.get(key)

    def __setitem__(self, key: str, value: Any) -> None:
        logger.debug("config.set", key=key, value=value)
        self._config[key] = value

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def update(self, **kwargs) -> None:
        logger.debug("config.update", keys=sorted(kwargs))
        self._config.update(kwargs)

    @property
    def provider_uri(self) -> Optional[str]:
        return self._config.get("provider_uri")

    @property
    def provider_path(self) -> Optional[Path]:
        uri = self.provider_uri
        if uri is None:
            return None
        return Path(uri).expanduser().resolve()

    def initialize(self, provider_uri: Optional[str] = None, **kwargs) -> None:
        """
        Initialize qlib_data configuration.

        Parameters
        ----------
        provider_uri : str, optional
            Path to the QLib data directory
        **kwargs : dict
            Additional configuration options
        """
        if provider_uri is not None:
            resolved = str(Path(provider_uri).expanduser().resolve())
            self._config["provider_uri"] = resolved
            logger.info("config.provider_uri_set", provider_uri=resolved)
        self._config.update(kwargs)
        if kwargs:
            logger.info("config.initialized", extra=kwargs)

    def reset(self) -> None:
        """Reset configuration to defaults."""
        logger.info("config.reset")
        self._config.clear()
        self._config.update({
            "provider_uri": None,
            "region": "CN",
            "auto_mmap": True,
            "mem_cache_size": "6GB",
        })


# Global config instance
C = Config()
logger.debug("config.module_loaded")
