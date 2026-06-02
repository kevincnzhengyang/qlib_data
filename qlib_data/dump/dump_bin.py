# Copyright (c) 2026
# Licensed under the MIT License

"""Data dump utilities for converting CSV to QLib bin format."""

from pathlib import Path
from typing import Union, Iterable, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

from ..logging import get_logger


logger = get_logger(__name__)


class DumpDataAll:
    """Dump CSV data to QLib bin format.

    This class handles conversion of CSV files containing OHLCV data
    into QLib's binary format for efficient storage and retrieval.

    Two operating modes are supported:

    * **Full overwrite** (``incremental=False``, default) -- every dump
      replaces the calendar, instruments and feature binaries that already
      live under ``qlib_dir``.
    * **Incremental** (``incremental=True``) -- the new CSV rows are merged
      with whatever is already on disk.  For calendars and instruments the
      union of dates/ranges is taken; for features the new values win on
      overlapping dates and missing entries are filled from disk.
    """

    CALENDARS_DIR = "calendars"
    FEATURES_DIR = "features"
    INSTRUMENTS_DIR = "instruments"
    INSTRUMENTS_FILE = "all.txt"

    # Calendar / timestamp formats per frequency.  ``day`` and ``tick`` are
    # the two frequencies with first-class support; any other value falls
    # back to a second-resolution high-frequency format.
    DAILY_FORMAT = "%Y-%m-%d"
    TICK_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
    HIGH_FREQ_FORMAT = "%Y-%m-%d %H:%M:%S"

    FREQ_FORMATS: "dict[str, str]" = {
        "day": DAILY_FORMAT,
        "tick": TICK_FORMAT,
    }
    DEFAULT_HIGH_FREQ_FORMAT = HIGH_FREQ_FORMAT

    SUPPORTED_FREQS: Tuple[str, ...] = ("day", "tick")

    SEP = "\t"

    def __init__(
        self,
        csv_path: Union[str, Path],
        qlib_dir: Union[str, Path],
        freq: str = "day",
        date_field: str = "date",
        symbol_field: str = "symbol",
        include_fields: Optional[List[str]] = None,
        exclude_fields: Optional[List[str]] = None,
        max_workers: int = 4,
        incremental: bool = False,
    ):
        """
        Parameters
        ----------
        csv_path : str or Path
            Path to CSV file or directory containing CSV files
        qlib_dir : str or Path
            Output directory for QLib data
        freq : str
            Frequency.  ``"day"`` and ``"tick"`` are the two frequencies
            with first-class support; ``"day"`` is the default.  Any other
            string is treated as a high-frequency bucket and uses a
            second-resolution timestamp format.
        date_field : str
            Name of the date column in CSV
        symbol_field : str
            Name of the symbol/instrument column in CSV
        include_fields : list, optional
            Fields to include (None = all except excluded)
        exclude_fields : list, optional
            Fields to exclude (None = none)
        max_workers : int
            Number of parallel workers
        incremental : bool
            If ``True``, merge the new CSV data with whatever is already
            stored on disk instead of overwriting it.  See the class
            docstring for the merge rules.
        """
        if freq is None:
            freq = "day"
        self.freq = freq.lower()
        self.date_field = date_field
        self.symbol_field = symbol_field
        self.include_fields = include_fields
        self.exclude_fields = exclude_fields or []
        self.max_workers = max_workers
        self.incremental = incremental

        self.csv_path = Path(csv_path).expanduser()
        self.qlib_dir = Path(qlib_dir).expanduser()
        self.calendar_format = self._resolve_calendar_format(self.freq)

        # Find CSV files
        if self.csv_path.is_dir():
            self.csv_files = sorted(self.csv_path.glob("*.csv"))
        else:
            self.csv_files = [self.csv_path]

        logger.info(
            "dump.initialized",
            csv_path=str(self.csv_path),
            qlib_dir=str(self.qlib_dir),
            freq=self.freq,
            csv_count=len(self.csv_files),
            include_fields=self.include_fields,
            exclude_fields=self.exclude_fields,
            incremental=self.incremental,
            calendar_format=self.calendar_format,
        )

    @classmethod
    def _resolve_calendar_format(cls, freq: str) -> str:
        """Return the timestamp format used for the calendar file."""
        return cls.FREQ_FORMATS.get(freq, cls.DEFAULT_HIGH_FREQ_FORMAT)

    def _format_timestamp(self, ts: pd.Timestamp) -> str:
        """Render a timestamp using the format selected for ``freq``."""
        return pd.Timestamp(ts).strftime(self.calendar_format)

    def dump(self) -> None:
        """Execute the dump process."""
        logger.info(
            "dump.start",
            csv_count=len(self.csv_files),
            incremental=self.incremental,
        )
        # Read all CSV files.  When the CSV carries a ``symbol_field``
        # column we group rows by symbol so a single file can contain
        # several instruments.  Otherwise the file stem is used as the
        # symbol, preserving the previous behaviour.
        all_data = []
        for csv_file in self.csv_files:
            df = self._read_csv(csv_file)
            if df is None or df.empty:
                continue
            if self.symbol_field in df.columns:
                for symbol, sub in df.groupby(self.symbol_field, sort=True):
                    all_data.append((str(symbol).lower(), sub.copy()))
            else:
                all_data.append((csv_file.stem.lower(), df))

        if not all_data:
            logger.warning("dump.empty", message="No data to dump")
            print("No data to dump")
            return

        # Read existing data when running in incremental mode.  Reading is
        # skipped on the first incremental dump (no files on disk yet).
        existing_calendar: List[pd.Timestamp] = []
        existing_instruments: dict = {}
        if self.incremental:
            existing_calendar = self._read_existing_calendar()
            existing_instruments = self._read_existing_instruments()
            logger.info(
                "dump.existing_loaded",
                calendar_count=len(existing_calendar),
                instrument_count=len(existing_instruments),
            )

        # Extract and save calendars
        new_calendars = self._extract_calendars(all_data)
        if self.incremental and existing_calendar:
            calendars = sorted(set(existing_calendar) | set(new_calendars))
        else:
            calendars = new_calendars
        logger.info(
            "dump.calendars_extracted",
            new_count=len(new_calendars),
            merged_count=len(calendars),
        )
        self._save_calendars(calendars)

        # Extract and save instruments
        new_instruments = self._extract_instruments(all_data)
        if self.incremental and existing_instruments:
            instruments = self._merge_instruments(existing_instruments, new_instruments)
        else:
            instruments = new_instruments
        logger.info(
            "dump.instruments_extracted",
            new_count=len(new_instruments),
            merged_count=len(instruments),
        )
        self._save_instruments(instruments)

        # Save features (merges with existing bin files when incremental).
        self._save_features(all_data, existing_calendar=existing_calendar)

        logger.info(
            "dump.completed",
            instruments=len(all_data),
            target=str(self.qlib_dir),
            incremental=self.incremental,
        )
        print(f"Dumped {len(all_data)} instruments to {self.qlib_dir}")

    def _read_csv(self, file_path: Path) -> Optional[pd.DataFrame]:
        """Read a CSV file."""
        try:
            df = pd.read_csv(file_path)
            if self.date_field in df.columns:
                # ``pd.to_datetime`` will accept both date-only strings
                # (``2026-01-09``) and full timestamps with microseconds
                # (``2026-01-09 09:30:00.123456``) which covers both
                # ``day`` and ``tick`` frequencies.
                df[self.date_field] = pd.to_datetime(df[self.date_field])
            logger.debug(
                "dump.csv_loaded",
                file=str(file_path),
                rows=len(df),
                cols=len(df.columns),
            )
            return df
        except Exception as e:
            logger.error(
                "dump.csv_failed", file=str(file_path), error=str(e), exc_info=True
            )
            print(f"Error reading {file_path}: {e}")
            return None

    def _extract_calendars(self, all_data: List) -> List[pd.Timestamp]:
        """Extract all unique dates."""
        all_dates = set()
        for _, df in all_data:
            if self.date_field in df.columns:
                all_dates.update(df[self.date_field].dropna())
        return sorted(all_dates)

    def _extract_instruments(self, all_data: List) -> dict:
        """Extract instrument date ranges."""
        instruments = {}
        for symbol, df in all_data:
            if self.date_field in df.columns:
                dates = df[self.date_field].dropna()
                if not dates.empty:
                    instruments[symbol] = [(dates.min(), dates.max())]
        return instruments

    def _read_existing_calendar(self) -> List[pd.Timestamp]:
        """Read existing calendar from disk (empty list if missing)."""
        cal_file = self.qlib_dir / self.CALENDARS_DIR / f"{self.freq}.txt"
        if not cal_file.exists():
            return []
        with open(cal_file, "r") as f:
            return [pd.Timestamp(line.strip()) for line in f if line.strip()]

    def _read_existing_instruments(self) -> dict:
        """Read existing instruments from disk (empty dict if missing)."""
        inst_file = self.qlib_dir / self.INSTRUMENTS_DIR / self.INSTRUMENTS_FILE
        if not inst_file.exists():
            return {}
        instruments: dict = {}
        with open(inst_file, "r") as f:
            for line in f:
                parts = line.strip().split(self.SEP)
                if len(parts) >= 3:
                    code, start_date, end_date = parts[0], parts[1], parts[2]
                    instruments.setdefault(code, []).append(
                        (pd.Timestamp(start_date), pd.Timestamp(end_date))
                    )
        return instruments

    @staticmethod
    def _merge_instruments(existing: dict, new: dict) -> dict:
        """Combine instrument date ranges; the new range wins on overlap."""
        merged: dict = {symbol: list(periods) for symbol, periods in existing.items()}
        for symbol, periods in new.items():
            new_start = min(p[0] for p in periods)
            new_end = max(p[1] for p in periods)
            if symbol in merged and merged[symbol]:
                old_start = min(p[0] for p in merged[symbol])
                old_end = max(p[1] for p in merged[symbol])
                merged_start = min(old_start, new_start)
                merged_end = max(old_end, new_end)
            else:
                merged_start = new_start
                merged_end = new_end
            merged[symbol] = [(merged_start, merged_end)]
        return merged

    def _save_calendars(self, calendars: List[pd.Timestamp]) -> None:
        """Save calendar to file."""
        cal_dir = self.qlib_dir / self.CALENDARS_DIR
        cal_dir.mkdir(parents=True, exist_ok=True)

        cal_file = cal_dir / f"{self.freq}.txt"
        with open(cal_file, "w") as f:
            for cal in calendars:
                f.write(f"{self._format_timestamp(cal)}\n")
        logger.info("dump.calendar_written", file=str(cal_file), count=len(calendars))

    def _save_instruments(self, instruments: dict) -> None:
        """Save instruments to file."""
        inst_dir = self.qlib_dir / self.INSTRUMENTS_DIR
        inst_dir.mkdir(parents=True, exist_ok=True)

        inst_file = inst_dir / self.INSTRUMENTS_FILE
        with open(inst_file, "w") as f:
            for symbol, periods in instruments.items():
                for start_date, end_date in periods:
                    f.write(
                        f"{symbol}{self.SEP}"
                        f"{self._format_timestamp(start_date)}{self.SEP}"
                        f"{self._format_timestamp(end_date)}\n"
                    )
        logger.info(
            "dump.instruments_written",
            file=str(inst_file),
            count=len(instruments),
        )

    def _save_features(
        self,
        all_data: List,
        existing_calendar: Optional[List[pd.Timestamp]] = None,
    ) -> None:
        """Save features to binary files.

        When ``incremental`` is True and a bin file already exists, the
        existing values are kept for dates that the new CSV does not cover
        and the new values win for overlapping / new dates.
        """
        feat_dir = self.qlib_dir / self.FEATURES_DIR
        feat_dir.mkdir(parents=True, exist_ok=True)

        # Get fields to save
        _, first_df = all_data[0]
        all_fields = [
            c
            for c in first_df.columns
            if c not in [self.date_field, self.symbol_field]
        ]
        fields_to_save = self._filter_fields(all_fields)
        logger.debug(
            "dump.fields_resolved",
            candidates=len(all_fields),
            selected=len(fields_to_save),
        )

        # Process each instrument
        for symbol, df in all_data:
            new_dates = sorted(df[self.date_field].dropna().unique())
            new_data_indexed = df.set_index(self.date_field)
            saved_count = 0
            skipped_count = 0

            for field in fields_to_save:
                if field not in df.columns:
                    logger.warning(
                        "dump.field_missing", symbol=symbol, field=field
                    )
                    skipped_count += 1
                    continue

                feat_file_dir = feat_dir / symbol.lower()
                feat_file_dir.mkdir(parents=True, exist_ok=True)
                feat_file = (
                    feat_file_dir / f"{field.lower()}.{self.freq}.bin"
                )

                new_values_by_date = dict(
                    zip(
                        new_dates,
                        new_data_indexed[field].reindex(new_dates).values,
                    )
                )

                if self.incremental and feat_file.exists():
                    merged_values = self._merge_feature_values(
                        feat_file=feat_file,
                        new_values_by_date=new_values_by_date,
                        new_dates=new_dates,
                        existing_calendar=existing_calendar or [],
                    )
                else:
                    merged_values = (
                        new_data_indexed[field]
                        .reindex(new_dates)
                        .values.astype(np.float32)
                    )

                start_idx = 0
                values = np.asarray(merged_values, dtype=np.float32)
                with open(feat_file, "wb") as f:
                    np.hstack([[start_idx], values]).astype("<f").tofile(f)
                saved_count += 1

            logger.info(
                "dump.instrument_features_written",
                symbol=symbol,
                fields=saved_count,
                skipped=skipped_count,
                rows=len(new_dates),
            )

    def _merge_feature_values(
        self,
        feat_file: Path,
        new_values_by_date: dict,
        new_dates: List[pd.Timestamp],
        existing_calendar: List[pd.Timestamp],
    ) -> np.ndarray:
        """Merge the existing bin file with the new values for one field.

        The existing per-instrument dates are reconstructed by aligning the
        bin file's ``start_index`` and length with the global calendar.  This
        matches the convention used by :class:`qlib_data.FeatureProvider`
        when reading the data back.
        """

        with open(feat_file, "rb") as f:
            raw = np.fromfile(f, dtype="<f")
        if raw.size == 0:
            existing_dates: List[pd.Timestamp] = []
            existing_values = np.array([], dtype=np.float32)
        else:
            start_idx = int(raw[0])
            existing_values = raw[1:].astype(np.float32)
            if existing_calendar and 0 <= start_idx:
                end_idx = min(start_idx + len(existing_values), len(existing_calendar))
                existing_dates = list(existing_calendar[start_idx:end_idx])
                # Trim the existing values to the slice we were able to
                # back-derive from the global calendar.
                if len(existing_dates) < len(existing_values):
                    existing_values = existing_values[: len(existing_dates)]
            else:
                # Fall back to a position-only calendar; new values for
                # matching positions still win.
                existing_dates = list(range(len(existing_values)))

        merged_dates = sorted(set(existing_dates) | set(new_dates))
        merged_values = np.empty(len(merged_dates), dtype=np.float32)
        for i, d in enumerate(merged_dates):
            if isinstance(d, pd.Timestamp) and d in new_values_by_date:
                merged_values[i] = new_values_by_date[d]
            elif d in existing_dates:
                merged_values[i] = existing_values[existing_dates.index(d)]
            else:
                merged_values[i] = np.nan
        return merged_values

    def _filter_fields(self, fields: List[str]) -> List[str]:
        """Filter fields based on include/exclude lists."""
        if self.include_fields:
            return [f for f in fields if f in self.include_fields]
        return [f for f in fields if f not in self.exclude_fields]


def dump_dataset(
    csv_path: Union[str, Path],
    qlib_dir: Union[str, Path],
    freq: str = "day",
    date_field: str = "date",
    symbol_field: str = "symbol",
    include_fields: Optional[List[str]] = None,
    exclude_fields: Optional[List[str]] = None,
    incremental: bool = False,
) -> None:
    """Dump CSV data to QLib bin format.

    This is the main entry point for converting CSV data to QLib format.

    Parameters
    ----------
    csv_path : str or Path
        Path to CSV file or directory containing CSV files
    qlib_dir : str or Path
        Output directory for QLib data
    freq : str
        Frequency of the data.  ``"day"`` and ``"tick"`` are the two
        frequencies with first-class support.  Any other value is treated
        as a high-frequency bucket and uses a second-resolution timestamp
        format.
    date_field : str
        Name of the date column in CSV
    symbol_field : str
        Name of the symbol/instrument column in CSV
    include_fields : list, optional
        Fields to include (None = all)
    exclude_fields : list, optional
        Fields to exclude
    incremental : bool
        If ``True`` (default ``False``) merge the new CSV data with
        whatever is already stored under ``qlib_dir`` instead of
        overwriting it.  See :class:`DumpDataAll` for the merge rules.

    Example
    -------
    >>> dump_dataset(
    ...     csv_path="tests/HK.00100.csv",
    ...     qlib_dir="~/.qlib_data/hk_data",
    ...     freq="day",
    ...     date_field="date",
    ...     symbol_field="symbol",
    ... )

    Tick data::

    >>> dump_dataset(
    ...     csv_path="ticks/HK.00100.csv",
    ...     qlib_dir="~/.qlib_data/hk_ticks",
    ...     freq="tick",
    ... )

    Append new rows to an existing dataset::

    >>> dump_dataset(
    ...     csv_path="ticks/HK.00100_extra.csv",
    ...     qlib_dir="~/.qlib_data/hk_ticks",
    ...     freq="tick",
    ...     incremental=True,
    ... )
    """
    dumper = DumpDataAll(
        csv_path=csv_path,
        qlib_dir=qlib_dir,
        freq=freq,
        date_field=date_field,
        symbol_field=symbol_field,
        include_fields=include_fields,
        exclude_fields=exclude_fields,
        incremental=incremental,
    )
    dumper.dump()