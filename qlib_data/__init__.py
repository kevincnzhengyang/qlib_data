# Copyright (c) 2026
# Licensed under the MIT License

"""
QLib Data Layer - Independent data management for quantitative finance.

This package extracts the data layer from QLib, providing:
- Calendar (trading days) management
- Instrument (stock codes) management
- Feature data storage and retrieval
- Binary data format compatible with QLib

The public API is intentionally narrow and consists of three entry points:

- :func:`init` -- configure the data directory
- :func:`load_dataset` -- read feature data for instruments
- :func:`dump_dataset` -- convert CSV data into the on-disk QLib format

Example
-------
>>> import qlib_data
>>> qlib_data.init("~/.qlib_data/hk_data")
>>> df = qlib_data.load_dataset(["HK.00100"], ["$close", "$open"], "2026-01-01", "2026-06-01")
>>> qlib_data.dump_dataset("data.csv", "~/.qlib_data/hk_data")
"""

from .config import C
from .logging import configure_logging, get_logger
from .data import load_dataset, calendar, instruments

# Import dump functionality
from .dump import dump_dataset


# Configure structlog as early as possible.  This is a no-op if the user has
# already configured logging or set QLIB_DATA_LOG_LEVEL.
configure_logging()
logger = get_logger("qlib_data")


def init(provider_uri: str = None, **kwargs) -> None:
    """
    Initialize the qlib_data module.

    Parameters
    ----------
    provider_uri : str
        Path to the QLib data directory (containing calendars/, features/, instruments/)
    **kwargs : dict
        Additional configuration options

    Example
    -------
    >>> import qlib_data
    >>> qlib_data.init("/path/to/qlib_data")
    >>> df = qlib_data.load_dataset(["HK.00100"], ["$close"], "2026-01-01", "2026-06-01")
    """
    logger.info("qlib_data.init", provider_uri=provider_uri, extra=kwargs)
    C.initialize(provider_uri=provider_uri, **kwargs)


__all__ = [
    "init",
    "load_dataset",
    "dump_dataset",
    "calendar",
    "instruments",
    "C",
    "configure_logging",
    "get_logger",
]
