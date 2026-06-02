# Copyright (c) 2026
# Licensed under the MIT License

"""Binary storage implementation for QLib data format."""

from pathlib import Path
from typing import Union, Iterable, List, Tuple, Optional, Dict

import numpy as np
import pandas as pd

from ..config import C
from ..logging import get_logger


logger = get_logger(__name__)


class CalendarStorage:
    """Calendar storage for trading days."""

    # Calendar / timestamp formats per frequency.  ``day`` and ``tick`` are
    # the two frequencies with first-class support; any other value is
    # treated as a high-frequency bucket and uses a second-resolution
    # format.  The actual parsing of the on-disk file is delegated to
    # :class:`pandas.Timestamp` which is format-agnostic, so the entries
    # only need to round-trip through one of these strings.
    DAILY_FORMAT = "%Y-%m-%d"
    TICK_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"

    FREQ_FORMATS: "dict[str, str]" = {
        "day": DAILY_FORMAT,
        "tick": TICK_FORMAT,
    }
    DEFAULT_HIGH_FREQ_FORMAT = HIGH_FREQ_FORMAT

    def __init__(self, freq: str = "day", future: bool = False, provider_uri: str = None):
        self.freq = freq
        self.future = future
        self._provider_uri = provider_uri

    @property
    def provider_uri(self) -> Optional[Path]:
        uri = self._provider_uri or C.provider_uri
        if uri is None:
            return None
        return Path(uri)

    @property
    def uri(self) -> Path:
        if self.provider_uri is None:
            raise ValueError("provider_uri is not set")
        freq_str = self.freq
        future_str = "_future" if self.future else ""
        return self.provider_uri / "calendars" / f"{freq_str}{future_str}.txt"

    def read(self) -> List[pd.Timestamp]:
        """Read calendar from file."""
        if not self.uri.exists():
            logger.warning("calendar.missing", uri=str(self.uri))
            return []
        with open(self.uri, "r") as f:
            dates = []
            for line in f:
                line = line.strip()
                if line:
                    dates.append(pd.Timestamp(line))
        logger.debug("calendar.read", uri=str(self.uri), count=len(dates))
        return dates

    def write(self, dates: Iterable[str]) -> None:
        """Write calendar to file."""
        self.uri.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(self.uri, "w") as f:
            for date in dates:
                f.write(f"{date}\n")
                count += 1
        logger.info("calendar.write", uri=str(self.uri), count=count)

    def __len__(self) -> int:
        return len(self.read())

    def __contains__(self, date: pd.Timestamp) -> bool:
        return date in self.read()


class InstrumentStorage:
    """Instrument storage for stock codes."""

    SEP = "\t"

    def __init__(self, freq: str = "day", provider_uri: str = None):
        self.freq = freq
        self._provider_uri = provider_uri

    @property
    def provider_uri(self) -> Optional[Path]:
        uri = self._provider_uri or C.provider_uri
        if uri is None:
            return None
        return Path(uri)

    @property
    def uri(self) -> Path:
        if self.provider_uri is None:
            raise ValueError("provider_uri is not set")
        return self.provider_uri / "instruments" / f"all.txt"

    def read(self) -> Dict[str, List[Tuple[str, str]]]:
        """Read instruments from file.

        Returns
        -------
        dict
            {instrument_code: [(start_date, end_date), ...]}
        """
        if not self.uri.exists():
            logger.warning("instruments.missing", uri=str(self.uri))
            return {}
        instruments = {}
        with open(self.uri, "r") as f:
            for line in f:
                parts = line.strip().split(self.SEP)
                if len(parts) >= 3:
                    code, start_date, end_date = parts[0], parts[1], parts[2]
                    instruments.setdefault(code, []).append((start_date, end_date))
        logger.debug("instruments.read", uri=str(self.uri), count=len(instruments))
        return instruments

    def write(self, data: Dict[str, List[Tuple[str, str]]]) -> None:
        """Write instruments to file."""
        self.uri.parent.mkdir(parents=True, exist_ok=True)
        count = 0
        with open(self.uri, "w") as f:
            for code, periods in data.items():
                for start_date, end_date in periods:
                    f.write(f"{code}{self.SEP}{start_date}{self.SEP}{end_date}\n")
                    count += 1
        logger.info("instruments.write", uri=str(self.uri), instruments=len(data), rows=count)

    def get_instruments(self) -> List[str]:
        """Get list of all instrument codes."""
        return list(self.read().keys())


class BinStorage:
    """Feature storage in QLib bin format.

    Format:
    - File: {provider_uri}/features/{instrument}/{field}.{freq}.bin
    - Content: [start_index, value1, value2, ...] as float32 (little-endian)
    """

    DTYPE = "<f"  # little-endian float32

    def __init__(self, instrument: str, field: str, freq: str = "day", provider_uri: str = None):
        self.instrument = instrument
        self.field = field
        self.freq = freq
        self._provider_uri = provider_uri

    @property
    def provider_uri(self) -> Optional[Path]:
        uri = self._provider_uri or C.provider_uri
        if uri is None:
            return None
        return Path(uri)

    @property
    def uri(self) -> Path:
        if self.provider_uri is None:
            raise ValueError("provider_uri is not set")
        return (
            self.provider_uri
            / "features"
            / self.instrument.lower()
            / f"{self.field.lower()}.{self.freq.lower()}.bin"
        )

    @property
    def start_index(self) -> Optional[int]:
        """Get the starting calendar index."""
        if not self.uri.exists():
            logger.warning("bin.missing_start", uri=str(self.uri))
            return None
        with open(self.uri, "rb") as f:
            data = np.frombuffer(f.read(4), dtype=self.DTYPE)
            return int(data[0])

    @property
    def end_index(self) -> Optional[int]:
        """Get the ending calendar index."""
        if not self.uri.exists():
            logger.warning("bin.missing_end", uri=str(self.uri))
            return None
        return self.start_index + len(self) - 1

    def __len__(self) -> int:
        """Get number of data points."""
        if not self.uri.exists():
            return 0
        with open(self.uri, "rb") as f:
            data = np.fromfile(f, dtype=self.DTYPE)
            return len(data) - 1  # subtract 1 for start_index

    def __getitem__(self, key: Union[int, slice]) -> Union[float, pd.Series]:
        """Get feature value(s).

        Parameters
        ----------
        key : int or slice
            Index or slice of calendar positions

        Returns
        -------
        float or pd.Series
            Single value or series of values
        """
        if not self.uri.exists():
            logger.warning("bin.read_missing", uri=str(self.uri), key=str(key))
            if isinstance(key, int):
                return np.nan
            return pd.Series(dtype=np.float32)

        with open(self.uri, "rb") as f:
            data = np.fromfile(f, dtype=self.DTYPE)

        start_idx = int(data[0])
        values = data[1:]

        if isinstance(key, int):
            abs_index = start_idx + key
            if abs_index < start_idx or abs_index >= start_idx + len(values):
                return np.nan
            return values[key]

        elif isinstance(key, slice):
            result = values[key]
            new_index = pd.RangeIndex(start=start_idx + (key.start or 0), stop=start_idx + len(values), step=key.step or 1)
            return pd.Series(result, index=new_index)

        return pd.Series(dtype=np.float32)

    def __setitem__(self, key: slice, values: Union[np.ndarray, pd.Series, List]) -> None:
        """Write feature values.

        Parameters
        ----------
        key : slice
            Slice specifying start position
        values : array-like
            Values to write
        """
        self.uri.parent.mkdir(parents=True, exist_ok=True)

        if isinstance(values, pd.Series):
            values = values.values
        values = np.array(values, dtype=np.float32)

        if not self.uri.exists():
            start_index = key.start or 0
            with open(self.uri, "wb") as f:
                np.hstack([[start_index], values]).astype(self.DTYPE).tofile(f)
            logger.debug("bin.write_new", uri=str(self.uri), start_index=start_index, count=len(values))
        else:
            existing_data = self[:]
            existing_start = self.start_index

            # Create full array with existing data
            total_len = len(existing_data) + existing_start
            full_data = np.full(total_len, np.nan, dtype=np.float32)
            full_data[existing_start - existing_start:existing_start - existing_start + len(existing_data)] = existing_data.values

            # Insert new values
            new_start = key.start or 0
            full_data[new_start:new_start + len(values)] = values

            # Find new start index
            first_valid = np.argmax(~np.isnan(full_data))
            last_valid = len(full_data) - np.argmax(~np.isnan(full_data[::-1])) - 1

            with open(self.uri, "wb") as f:
                np.hstack([[first_valid], full_data[first_valid:last_valid + 1]]).astype(self.DTYPE).tofile(f)
            logger.debug(
                "bin.write_extend",
                uri=str(self.uri),
                new_start=new_start,
                first_valid=int(first_valid),
                last_valid=int(last_valid),
                count=len(values),
            )

    def read_all(self) -> pd.Series:
        """Read all feature values as a Series."""
        if not self.uri.exists():
            logger.warning("bin.read_all_missing", uri=str(self.uri))
            return pd.Series(dtype=np.float32)
        with open(self.uri, "rb") as f:
            data = np.fromfile(f, dtype=self.DTYPE)
        start_idx = int(data[0])
        return pd.Series(data[1:], index=pd.RangeIndex(start=start_idx, stop=start_idx + len(data) - 1))

    def write_all(self, data: Union[np.ndarray, pd.Series], start_index: int = 0) -> None:
        """Write all feature values.

        Parameters
        ----------
        data : array-like
            Feature values
        start_index : int
            Starting calendar index
        """
        self.uri.parent.mkdir(parents=True, exist_ok=True)
        if isinstance(data, pd.Series):
            start_index = data.index[0]
            data = data.values
        with open(self.uri, "wb") as f:
            np.hstack([[start_index], data]).astype(self.DTYPE).tofile(f)
        logger.debug("bin.write_all", uri=str(self.uri), start_index=start_index, count=len(data))


def read_calendars(provider_uri: str, freq: str = "day", future: bool = False) -> List[pd.Timestamp]:
    """Read trading calendar from provider."""
    storage = CalendarStorage(freq=freq, future=future, provider_uri=provider_uri)
    return storage.read()


def read_instruments(provider_uri: str) -> Dict[str, List[Tuple[str, str]]]:
    """Read instruments from provider."""
    storage = InstrumentStorage(provider_uri=provider_uri)
    return storage.read()


def read_feature(instrument: str, field: str, freq: str, provider_uri: str) -> pd.Series:
    """Read a single feature from provider."""
    storage = BinStorage(instrument, field, freq, provider_uri)
    return storage.read_all()
