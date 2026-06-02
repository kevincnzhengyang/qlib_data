# qlib_data

> Independent data layer extracted from QLib - OHLCV binary storage, calendar/instrument management and an expression engine. No ML dependencies.

## Overview

`qlib_data` is the data management half of [QLib](https://github.com/microsoft/qlib), repackaged as a small standalone library. It provides:

- **Binary storage** of OHLCV and feature data in QLib-compatible `.bin` format
- **Trading calendar** and **instrument** (stock code) management
- **Expression system** with `Ref`, `Mean`, `Std`, `Add`, `Sub`, arithmetic / comparison / rolling operators
- **CSV → Bin conversion** via `dump_dataset()` (full overwrite or incremental merge)
- **Two built-in frequencies**: `day` (`%Y-%m-%d`) and `tick` (`%Y-%m-%d %H:%M:%S.%f`)
- Structured logging via [`structlog`](https://www.structlog.org/) (level controllable with `QLIB_DATA_LOG_LEVEL`)

## Installation

This is a **private library** - install it editable from the source tree:

```bash
pip install -r requirements.txt
pip install -e .
```

## Quick Start

The public API is intentionally narrow and consists of three entry points:

- `init()` -- configure the data directory
- `load_dataset()` -- read feature data for instruments
- `dump_dataset()` -- convert CSV data into the on-disk QLib format

```python
import qlib_data

# 1. Convert CSV data to QLib binary format
qlib_data.dump_dataset(
    csv_path="data/HK.00100.csv",
    qlib_dir="~/.qlib_data/hk_data",
    freq="day",
    date_field="date",
    symbol_field="symbol",
)

# 2. Point qlib_data at the data directory
qlib_data.init("~/.qlib_data/hk_data")

# 3. Load features
df = qlib_data.load_dataset(
    instruments=["HK.00100"],
    fields=["$close", "$open", "$volume"],
    start_time="2026-01-01",
    end_time="2026-06-01",
)
print(df.head())
```

## API Reference

### `init(provider_uri, **kwargs)`
Configure qlib_data with the on-disk data directory.

### `dump_dataset(csv_path, qlib_dir, freq="day", ..., incremental=False)`
Convert one or more CSV files into the on-disk QLib format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `csv_path` | str/Path | - | CSV file or directory of CSVs |
| `qlib_dir` | str/Path | - | Output directory |
| `freq` | str | `"day"` | Frequency: `"day"` or `"tick"` |
| `date_field` | str | `"date"` | Date column in the CSV |
| `symbol_field` | str | `"symbol"` | Symbol column in the CSV |
| `include_fields` | list | None | Whitelist of fields |
| `exclude_fields` | list | None | Blacklist of fields |
| `incremental` | bool | `False` | Merge with existing data instead of full overwrite |

When `incremental=True`, the new CSV rows are merged with the on-disk dataset:
calendar entries are unioned, instrument date ranges are extended and
feature binaries are extended with the new dates (new values win on overlap).

### `load_dataset(instruments, fields, start_time=None, end_time=None, freq="day")`
Read features for the given instruments.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `instruments` | str/list | - | Instrument code(s) |
| `fields` | str/list | - | Field name(s) or expression(s) |
| `start_time` | str | None | Start date / timestamp |
| `end_time` | str | None | End date / timestamp |
| `freq` | str | `"day"` | Frequency: `"day"` or `"tick"` |

Field names may be given with or without the leading `$`:

```python
qlib_data.load_dataset(["HK.00100"], ["$close"])   # OK
qlib_data.load_dataset(["HK.00100"], ["close"])    # also OK
```

### `calendar(start_time=None, end_time=None, freq="day", future=False)`
List the trading timestamps in `[start_time, end_time]`.

### `instruments()`
Return all instrument codes known to the dataset.

## Expression Syntax

### Field reference
```python
"$close"          # ordinary field
"$$fwd_eps"       # point-in-time field
```

### Operators
```python
"Ref($close, -1)"                              # previous bar's close
"Mean($close, 5)"                              # 5-bar moving average
"Std($close, 10)"                              # 10-bar rolling std
"Ref($close, -1) / $close - 1"                # bar-over-bar return
"($close - Mean($close, 20)) / Std($close, 20)"  # z-score
"If(GT($close, $open), $close - $open, 0)"    # conditional
```

## Storage Format

```
qlib_dir/
├── calendars/
│   └── {freq}.txt                  # one timestamp per line
├── features/
│   └── {instrument}/
│       └── {field}.{freq}.bin      # float32 little-endian
└── instruments/
    └── all.txt                     # symbol<TAB>start<TAB>end
```

Each `.bin` file is `[start_index, value_1, value_2, ...]` packed as little-endian
float32.  `start_index` is the position of `value_1` in the global calendar.

## Architecture

```
qlib_data/
├── __init__.py            # init(), re-exports
├── config.py              # Global Config singleton (C)
├── logging.py             # structlog configuration
├── data.py                # load_dataset(), calendar(), instruments()
├── expression/
│   ├── base.py            # Expression, Feature, PFeature
│   ├── ops.py             # Ref, Mean, Add, Sub, ...
│   └── parser.py          # parse_field()
├── provider/
│   ├── calendar.py        # CalendarProvider
│   ├── instrument.py      # InstrumentProvider
│   └── feature.py         # FeatureProvider
├── storage/
│   └── bin_storage.py     # BinStorage, CalendarStorage, InstrumentStorage
└── dump/
    └── dump_bin.py        # DumpDataAll, dump_dataset()
```

## Development

```bash
make install-dev   # pip install -e ".[dev]"
make test          # run pytest
make lint          # pyflakes
make clean         # remove pyc / cache files
```

## Dependencies

- Python 3.13+
- `numpy`, `pandas`, `structlog`
- Dev: `pytest`, `pytest-cov`

## License

MIT
