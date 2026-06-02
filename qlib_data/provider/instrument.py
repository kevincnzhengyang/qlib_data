# Copyright (c) 2026
# Licensed under the MIT License

"""Instrument provider for stock code management."""

from typing import List, Optional, Dict, Tuple

import pandas as pd

from ..config import C
from ..logging import get_logger
from ..storage.bin_storage import InstrumentStorage


logger = get_logger(__name__)


class InstrumentProvider:
    """Instrument provider for accessing stock codes."""

    def __init__(self, provider_uri: str = None):
        self._provider_uri = provider_uri
        self._instruments_cache = None
        logger.debug("instrument.provider_created", provider_uri=provider_uri)

    @property
    def provider_uri(self) -> str:
        return self._provider_uri or C.provider_uri

    def _get_storage(self) -> InstrumentStorage:
        """Get instrument storage instance."""
        return InstrumentStorage(provider_uri=self.provider_uri)

    def _load_instruments(self) -> Dict[str, List[Tuple[str, str]]]:
        """Load instruments from storage."""
        if self._instruments_cache is None:
            storage = self._get_storage()
            self._instruments_cache = storage.read()
            logger.debug("instrument.cache_loaded", count=len(self._instruments_cache))
        return self._instruments_cache

    def instruments(self) -> List[str]:
        """Get list of all instrument codes.

        Returns
        -------
        list
            List of instrument codes
        """
        codes = list(self._load_instruments().keys())
        logger.debug("instrument.list_returned", count=len(codes))
        return codes

    def get_instrument_date_range(self, instrument: str) -> Tuple[str, str]:
        """Get the date range for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument code

        Returns
        -------
        tuple
            (start_date, end_date)
        """
        periods = self._load_instruments().get(instrument, [])
        if not periods:
            logger.warning("instrument.date_range_missing", instrument=instrument)
            return None, None
        return periods[0][0], periods[-1][1]

    def is_valid_instrument(self, instrument: str) -> bool:
        """Check if an instrument exists.

        Parameters
        ----------
        instrument : str
            Instrument code

        Returns
        -------
        bool
            True if instrument exists
        """
        valid = instrument in self._load_instruments()
        logger.debug("instrument.validity_check", instrument=instrument, valid=valid)
        return valid

    def list_instruments(self) -> List[Dict]:
        """Get list of all instruments with metadata.

        Returns
        -------
        list
            List of dicts with 'symbol', 'start_date', 'end_date'
        """
        result = []
        for symbol, periods in self._load_instruments().items():
            for start_date, end_date in periods:
                result.append({
                    "symbol": symbol,
                    "start_date": start_date,
                    "end_date": end_date,
                })
        logger.debug("instrument.metadata_listed", count=len(result))
        return result

    def reload(self) -> None:
        """Reload instruments from storage."""
        logger.info("instrument.reload")
        self._instruments_cache = None


# Global instance
_instrument_provider = None


def get_instrument_provider() -> InstrumentProvider:
    """Get the global instrument provider instance."""
    global _instrument_provider
    if _instrument_provider is None:
        logger.debug("instrument.provider_singleton_create")
        _instrument_provider = InstrumentProvider()
    return _instrument_provider
