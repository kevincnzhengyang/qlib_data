# Copyright (c) 2026
# Licensed under the MIT License

"""Feature provider for accessing feature data."""

from typing import Union, List, Optional

import numpy as np
import pandas as pd

from ..config import C
from ..logging import get_logger
from ..storage.bin_storage import BinStorage, CalendarStorage


logger = get_logger(__name__)


class FeatureProvider:
    """Feature provider for accessing feature data."""

    def __init__(self, provider_uri: str = None):
        self._provider_uri = provider_uri
        self._feature_cache = {}
        logger.debug("feature.provider_created", provider_uri=provider_uri)

    @property
    def provider_uri(self) -> str:
        return self._provider_uri or C.provider_uri

    def _get_storage(self, instrument: str, field: str, freq: str) -> BinStorage:
        """Get feature storage instance."""
        key = (instrument, field, freq)
        if key not in self._feature_cache:
            logger.debug("feature.storage_create", instrument=instrument, field=field, freq=freq)
            self._feature_cache[key] = BinStorage(instrument, field, freq, provider_uri=self.provider_uri)
        return self._feature_cache[key]

    def feature(self, instrument: str, field: str, start_index: int, end_index: int, freq: str) -> pd.Series:
        """Get feature data for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument code
        field : str
            Feature field name (e.g., "close", "open")
        start_index : int
            Start calendar index
        end_index : int
            End calendar index
        freq : str
            Frequency

        Returns
        -------
        pd.Series
            Feature data with calendar index
        """
        storage = self._get_storage(instrument, field, freq)

        if not storage.uri.exists():
            logger.warning("feature.missing", instrument=instrument, field=field, freq=freq, uri=str(storage.uri))
            return pd.Series(dtype=np.float32)

        calendar_storage = CalendarStorage(freq=freq, provider_uri=self.provider_uri)
        calendar_list = calendar_storage.read()

        if not calendar_list:
            logger.warning("feature.empty_calendar", instrument=instrument, field=field, freq=freq)
            return pd.Series(dtype=np.float32)

        # Get data from storage
        data = storage.read_all()

        if data.empty:
            logger.warning("feature.empty_data", instrument=instrument, field=field, freq=freq)
            return pd.Series(dtype=np.float32)

        # Calculate relative positions in the data
        bin_start = int(data.index[0])
        bin_len = len(data)

        # Convert absolute calendar indices to relative positions
        rel_start = max(0, start_index - bin_start)
        rel_end = min(bin_len - 1, end_index - bin_start)

        if rel_start > rel_end or rel_start >= bin_len:
            logger.debug(
                "feature.range_miss",
                instrument=instrument,
                field=field,
                start_index=start_index,
                end_index=end_index,
                bin_start=bin_start,
                bin_len=bin_len,
            )
            return pd.Series(dtype=np.float32)

        # Slice data
        result = data.iloc[rel_start:rel_end + 1].copy()

        # Map indices to calendar dates
        try:
            new_index = [calendar_list[idx] for idx in result.index]
            result.index = pd.Index(new_index)
        except IndexError:
            # Fallback: use RangeIndex
            logger.warning("feature.index_mapping_failed", instrument=instrument, field=field)

        logger.debug(
            "feature.loaded",
            instrument=instrument,
            field=field,
            start_index=start_index,
            end_index=end_index,
            rel_start=rel_start,
            rel_end=rel_end,
            count=len(result),
        )
        return result

    def get_fields(self, instrument: str, freq: str = "day") -> List[str]:
        """Get all available fields for an instrument.

        Parameters
        ----------
        instrument : str
            Instrument code
        freq : str
            Frequency

        Returns
        -------
        list
            List of field names
        """
        from pathlib import Path
        uri = Path(self.provider_uri) / "features" / instrument.lower()
        if not uri.exists():
            logger.warning("feature.dir_missing", instrument=instrument, uri=str(uri))
            return []
        fields = []
        for f in uri.glob(f"*.{freq.lower()}.bin"):
            fields.append(f.stem.rsplit(".", 1)[0])  # Remove .{freq}.bin
        logger.debug("feature.fields_listed", instrument=instrument, freq=freq, count=len(fields))
        return fields


class PITFeatureProvider:
    """Point-in-Time Feature Provider for period data."""

    def __init__(self, provider_uri: str = None):
        self._provider_uri = provider_uri

    @property
    def provider_uri(self) -> str:
        return self._provider_uri or C.provider_uri

    def period_feature(self, instrument: str, field: str, start_index: int, end_index: int, cur_time: pd.Timestamp, period: int = None) -> pd.Series:
        """Get period feature data.

        This is used for point-in-time data where the same period (e.g., quarter)
        may have multiple observations over time.

        Parameters
        ----------
        instrument : str
            Instrument code
        field : str
            Feature field name
        start_index : int
            Start index
        end_index : int
            End index
        cur_time : pd.Timestamp
            Current observation time
        period : int, optional
            Specific period to retrieve

        Returns
        -------
        pd.Series
            Period feature data
        """
        logger.warning("pit.not_implemented", instrument=instrument, field=field)
        raise NotImplementedError("PIT features are not yet implemented")


# Global instance
_feature_provider = None


def get_feature_provider() -> FeatureProvider:
    """Get the global feature provider instance."""
    global _feature_provider
    if _feature_provider is None:
        logger.debug("feature.provider_singleton_create")
        _feature_provider = FeatureProvider()
    return _feature_provider


def features(
    instruments: Union[str, List[str]],
    fields: Union[str, List[str]],
    start_time: str = None,
    end_time: str = None,
    freq: str = "day",
) -> pd.DataFrame:
    """Get features for instruments.

    This is the main interface for accessing feature data, similar to qlib.data.D.features.

    Parameters
    ----------
    instruments : str or list
        Instrument code or list of codes
    fields : str or list
        Field expression(s) or list of field expressions
        Examples: "$close", ["$close", "$open"], "Ref($close, -1)"
    start_time : str, optional
        Start time
    end_time : str, optional
        End time
    freq : str
        Frequency (default: "day")

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [instrument, datetime] + field names

    Example
    -------
    >>> df = features(["HK.00100"], ["$close", "$open"], "2026-01-01", "2026-06-01")
    >>> df = features("HK.00100", "Ref($close, -1)/$close - 1")
    """
    from ..expression.parser import parse_fields
    from ..provider.calendar import get_calendar_provider

    logger.info(
        "features.request",
        instruments=instruments,
        fields=fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
    )

    # Normalize instruments
    if isinstance(instruments, str):
        instruments = [instruments]

    # Normalize fields
    if isinstance(fields, str):
        fields = [fields]

    # Parse expressions
    expressions = parse_fields(fields)
    logger.debug("features.expressions_parsed", count=len(expressions), types=[type(e).__name__ for e in expressions])

    # Get calendar and locate time range
    calendar_provider = get_calendar_provider()

    if start_time is None and end_time is None:
        start_index = 0
        end_index = len(calendar_provider._get_storage(freq, False).read()) - 1
    else:
        _, _, start_index, end_index = calendar_provider.locate_index(
            start_time or "1900-01-01",
            end_time or "2100-12-31",
            freq,
            False,
        )
    logger.info("features.index_range", start_index=start_index, end_index=end_index)

    # Fetch data for each instrument and expression
    feature_provider = get_feature_provider()
    result_dfs = []

    for instrument in instruments:
        logger.debug("features.instrument_start", instrument=instrument)
        instrument_data = {}

        for expr in expressions:
            col_name = str(expr)
            series = expr.load(instrument, start_index, end_index, freq)
            instrument_data[col_name] = series
            logger.debug("features.expression_loaded", instrument=instrument, expression=col_name, rows=len(series))

        if instrument_data:
            df = pd.DataFrame(instrument_data)
            df["instrument"] = instrument
            result_dfs.append(df)

    if not result_dfs:
        logger.warning("features.empty_result", instruments=instruments, fields=fields)
        return pd.DataFrame(columns=["instrument"] + [str(f) for f in fields])

    result = pd.concat(result_dfs, ignore_index=False)
    result = result.reset_index().rename(columns={"index": "datetime"})
    logger.info("features.returned", rows=len(result), columns=list(result.columns))
    return result
