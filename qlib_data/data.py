# Copyright (c) 2026
# Licensed under the MIT License

"""Main data access module - provides the load_dataset() function."""

import re
from typing import Union, List
import pandas as pd

from .config import C
from .logging import get_logger


logger = get_logger(__name__)


# Pattern that matches a *simple* field reference, i.e. either ``close`` or
# ``$close``.  The character class is kept in sync with the one used in
# :mod:`qlib_data.expression.parser` so the interface accepts the same
# identifiers as the expression engine.
_SIMPLE_FIELD_PATTERN = re.compile(
    r"^\$?[\w\u3001\uff1a\uff08\uff09]+$"
)


def _strip_field_prefix(field: str) -> str:
    """Return the bare field name (without a leading ``$``)."""
    return field[1:] if field.startswith("$") else field


def _resolve_index_range(start_time, end_time, freq):
    """Compute calendar index range used by :func:`load_dataset`."""
    from .provider.calendar import get_calendar_provider

    calendar_provider = get_calendar_provider()
    if start_time is None and end_time is None:
        calendar_list = calendar_provider._get_storage(freq, False).read()
        return 0, len(calendar_list) - 1
    _, _, start_index, end_index = calendar_provider.locate_index(
        start_time or "1900-01-01",
        end_time or "2100-12-31",
        freq,
        False,
    )
    return start_index, end_index


def load_dataset(
    instruments: Union[str, List[str]],
    fields: Union[str, List[str]],
    start_time: str = None,
    end_time: str = None,
    freq: str = "day",
) -> pd.DataFrame:
    """Load feature data for one or more instruments.

    This is the main interface for reading the on-disk dataset produced by
    :func:`qlib_data.dump_dataset`, and is the read-side counterpart of the
    public API alongside :func:`qlib_data.init` and
    :func:`qlib_data.dump_dataset`.

    Field strings may be specified either with or without the leading ``$``
    used by the expression engine.  Both forms are equivalent::

        >>> qlib_data.load_dataset(["HK.00100"], ["$close"])  # noqa: E501
        >>> qlib_data.load_dataset(["HK.00100"], ["close"])

    A field that contains operators (e.g. ``"Ref($close, -1)"``) is dispatched
    to the expression engine, while a bare field name is looked up directly
    in the binary storage.  The returned :class:`~pandas.DataFrame` always
    uses the canonical ``$name`` form for column labels.

    Parameters
    ----------
    instruments : str or list
        Instrument code or list of codes
    fields : str or list
        Field name(s) or expression(s).
        Examples: ``"close"``, ``["close", "open"]``, ``"Ref($close, -1)"``
    start_time : str, optional
        Start time (e.g., "2026-01-01")
    end_time : str, optional
        End time (e.g., "2026-06-01")
    freq : str
        Frequency (default: "day")

    Returns
    -------
    pd.DataFrame
        DataFrame with columns [instrument, datetime] + field names

    Example
    -------
    >>> import qlib_data
    >>> qlib_data.init("~/.qlib_data/hk_data")
    >>> df = qlib_data.load_dataset(["HK.00100"], ["$close", "$open"], "2026-01-01", "2026-06-01")
    >>> df = qlib_data.load_dataset("HK.00100", "Ref($close, -1)/$close - 1")
    """
    logger.info(
        "data.load_dataset",
        instruments=instruments,
        fields=fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        provider_uri=C.provider_uri,
    )

    if isinstance(instruments, str):
        instruments = [instruments]
    if isinstance(fields, str):
        fields = [fields]

    start_index, end_index = _resolve_index_range(start_time, end_time, freq)
    logger.info("load_dataset.index_range", start_index=start_index, end_index=end_index)

    from .provider.feature import get_feature_provider
    from .expression.parser import parse_fields

    feature_provider = get_feature_provider()
    result_dfs = []

    for instrument in instruments:
        logger.debug("load_dataset.instrument_start", instrument=instrument)
        instrument_data = {}

        for field in fields:
            if isinstance(field, str) and _SIMPLE_FIELD_PATTERN.match(field):
                # Simple field reference: look it up directly in the storage
                # layer.  We intentionally bypass the expression engine here
                # so the bare field name (without ``$``) is what reaches the
                # on-disk ``.bin`` files.
                bare_name = _strip_field_prefix(field)
                col_name = "$" + bare_name
                series = feature_provider.feature(
                    instrument, bare_name, start_index, end_index, freq
                )
                instrument_data[col_name] = series
                logger.debug(
                    "load_dataset.simple_field_loaded",
                    instrument=instrument,
                    field=bare_name,
                    col=col_name,
                    rows=len(series),
                )
            else:
                # Anything that contains operators / function calls is
                # delegated to the expression engine.
                for expr in parse_fields([field]):
                    col_name = str(expr)
                    series = expr.load(instrument, start_index, end_index, freq)
                    instrument_data[col_name] = series
                    logger.debug(
                        "load_dataset.expression_loaded",
                        instrument=instrument,
                        expression=col_name,
                        rows=len(series),
                    )

        if instrument_data:
            df = pd.DataFrame(instrument_data)
            df["instrument"] = instrument
            result_dfs.append(df)

    if not result_dfs:
        logger.warning("load_dataset.empty_result", instruments=instruments, fields=fields)
        return pd.DataFrame(columns=["instrument"] + [str(f) for f in fields])

    result = pd.concat(result_dfs, ignore_index=False)
    result = result.reset_index().rename(columns={"index": "datetime"})
    logger.info("load_dataset.returned", rows=len(result), columns=list(result.columns))
    return result


def calendar(
    start_time: str = None,
    end_time: str = None,
    freq: str = "day",
    future: bool = False,
) -> List[pd.Timestamp]:
    """Get trading calendar.

    Parameters
    ----------
    start_time : str, optional
        Start time
    end_time : str, optional
        End time
    freq : str
        Frequency
    future : bool
        Include future

    Returns
    -------
    list
        List of timestamps
    """
    logger.info("data.calendar", start_time=start_time, end_time=end_time, freq=freq, future=future)
    from .provider.calendar import get_calendar_provider
    return get_calendar_provider().calendar(start_time, end_time, freq, future)


def instruments() -> List[str]:
    """Get list of all instruments.

    Returns
    -------
    list
        List of instrument codes
    """
    logger.info("data.instruments")
    from .provider.instrument import get_instrument_provider
    return get_instrument_provider().instruments()
