# Copyright (c) 2026
# Licensed under the MIT License

"""Structured logging configuration based on :mod:`structlog`.

This module provides a small helper that configures :mod:`structlog` so all
loggers in the package produce consistent, structured output.  The
configuration is intentionally lightweight: it is safe to import and call
``configure_logging`` multiple times — only the first call has an effect.

The default renderer is :class:`structlog.dev.ConsoleRenderer` which prints
human-friendly colored output to stderr.  The level can be controlled via
the ``QLIB_DATA_LOG_LEVEL`` environment variable (defaults to ``INFO``).
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional

import structlog


_CONFIGURED = False


def configure_logging(level: Optional[str] = None, force: bool = False) -> None:
    """Configure :mod:`structlog` for the whole package.

    Parameters
    ----------
    level : str, optional
        Log level name (e.g. ``"DEBUG"``, ``"INFO"``).  When ``None`` the
        value of the ``QLIB_DATA_LOG_LEVEL`` environment variable is used,
        falling back to ``"INFO"``.
    force : bool
        If ``True`` reconfigure even if logging has already been configured.
    """

    global _CONFIGURED
    if _CONFIGURED and not force:
        return

    log_level_name = (level or os.environ.get("QLIB_DATA_LOG_LEVEL") or "INFO").upper()
    log_level = getattr(logging, log_level_name, logging.INFO)

    # Bridge Python's standard logging into structlog so anything that
    # uses ``logging.getLogger`` (e.g. third party libraries) still appears
    # in a sensible format.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="%Y-%m-%d %H:%M:%S", utc=False),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    _CONFIGURED = True


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Return a configured :mod:`structlog` logger.

    The logger is bound to ``name`` so it is easy to filter by module
    (e.g. ``qlib_data.provider.feature``).  Configuration is performed
    lazily on the first call.
    """

    configure_logging()
    return structlog.get_logger(name)


__all__ = ["configure_logging", "get_logger"]
