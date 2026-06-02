# AGENTS.md

> Guidance for AI agents working on the `qlib_data` codebase.

## Project Overview

`qlib_data` is an **independent data layer extracted from QLib** (Microsoft's quantitative investment platform). It provides:

- **Binary storage** of OHLCV and feature data (QLib-compatible `.bin` format)
- **Trading calendar** management
- **Instrument** (stock code) management
- **Expression system** with operators (`Ref`, `RollingMean`, etc.)
- **CSV → Bin conversion** via `dump_dataset()`

**Constraints (from `architecture.md`):**
- No dependency on `qlib` or `pyqlib`
- No ML libraries (no `torch`, `tensorflow`, `xgboost`, etc.)
- Keep QLib's binary format compatibility
- Keep expression syntax compatibility

## Quick Reference

### Essential Commands

```bash
# Install dependencies
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest tests/ -v

# Quick test (dumps and reads back HK.00100.csv)
python tests/test_qlib_data.py
```

### Public API

```python
import qlib_data

# Initialize (path to data dir)
qlib_data.init("/path/to/qlib_dir")

# Convert CSV → Bin
qlib_data.dump_dataset("data.csv", "/path/to/qlib_dir", freq="day")

# Load features
df = qlib_data.features(
    instruments=["HK.00100"],
    fields=["$close", "Ref($close, -1)"],
    start_time="2026-01-01",
    end_time="2026-06-01",
)

# Get calendar
dates = qlib_data.calendar("2026-01-01", "2026-06-01")

# Get instruments
symbols = qlib_data.instruments()
```

## Code Organization

```
qlib_data/
├── __init__.py              # init() + public API
├── config.py                # Global Config singleton (C)
├── data.py                  # features(), calendar(), instruments()
├── expression/
│   ├── base.py              # Expression, Feature, PFeature
│   ├── ops.py               # Ref, Mean, Std, Add, Sub, ... + Operators namespace
│   └── parser.py            # parse_field() - converts "$close" → Feature("close")
├── provider/
│   ├── calendar.py          # CalendarProvider + get_calendar_provider()
│   ├── instrument.py        # InstrumentProvider
│   └── feature.py           # FeatureProvider + features() function
├── storage/
│   └── bin_storage.py       # BinStorage, CalendarStorage, InstrumentStorage
└── dump/
    └── dump_bin.py          # DumpDataAll + dump_dataset()
```

## Key Patterns & Gotchas

### 1. Expression Evaluation

The parser converts strings to Expression objects via `eval()`. The eval scope MUST include:
- `Operators` namespace (for `Ref`, `Mean`, etc.)
- `Feature` and `PFeature` classes (for `$name` → `Feature("name")`)

See `expression/parser.py:65-70` for the eval globals.

### 2. Bin File Format

Each `.bin` file contains:
- First 4 bytes: `start_index` (int32, little-endian) — calendar position of first value
- Remaining bytes: `value_1, value_2, ...` (float32, little-endian)

**Critical**: The dump writes index `0` as the start. The reader must map `data_index → calendar_list[data_index]`.

### 3. Config Singleton

`C` in `config.py` is a **singleton**. After `init()`, all providers read from `C.provider_uri`. Don't create multiple Config instances.

### 4. Provider Global Instances

Calendar/Instrument/Feature providers have global singleton getters:
```python
from qlib_data.provider.calendar import get_calendar_provider
cp = get_calendar_provider()  # Returns the same instance every time
```

### 5. Calendar Date Format

- Daily: `"%Y-%m-%d"` (e.g., `"2026-01-09"`)
- High-freq: `"%Y-%m-%d %H:%M:%S"`

The dump auto-selects based on `freq` parameter.

### 6. Feature Provider Index Mapping

The `feature()` method in `FeatureProvider` reads the bin file's data (indexed 0..N-1), then maps each index to a calendar date via `calendar_list[index]`. The `start_index`/`end_index` parameters are calendar positions, not relative positions in the bin file.

See `provider/feature.py:33-90` — uses relative slicing `data.iloc[rel_start:rel_end+1]`.

## Expression Syntax

| Syntax | Meaning |
|--------|---------|
| `$close` | Reference field "close" |
| `$$fwd_eps` | Point-in-time field |
| `Ref($close, -1)` | Previous day's close |
| `Mean($close, 5)` | 5-day moving average |
| `Std($close, 10)` | 10-day rolling std |
| `$close + $open` | Element-wise add |
| `Ref($close, -1) / $close - 1` | Daily return |
| `GT($close, $open)` | Greater-than comparison |
| `If(GT($close, $open), $close, $open)` | Conditional |

## Adding a New Operator

1. Add class to `expression/ops.py` (inherit from `ExpressionOps`, `PairOperator`, `Rolling`, etc.)
2. Register in `Operators` namespace at bottom of `ops.py`
3. Implement `_load_internal()` and (if needed) `get_longest_back_rolling()` and `get_extended_window_size()`

Example:
```python
class Median(Rolling):
    def __init__(self, feature, N):
        super().__init__(feature, N, "median")
```

Add to `Operators`:
```python
class Operators:
    ...
    Median = Median
```

## Testing

Test data: `tests/HK.00100.csv` — Hong Kong stock with 94 daily bars (2026-01-09 to 2026-06-01) containing 44 fields (OHLCV + technical indicators).

Run tests with: `python -m pytest tests/test_qlib_data.py -v`

## Known Issues

1. **PIT (Point-in-Time) features** — `PITFeatureProvider` is a stub. Only static features are fully supported.
2. **No parallel dump** — `dump_dataset` runs sequentially. For large datasets, consider adding `concurrent.futures`.
3. **No resampling** — Calendar resampling (e.g., day → week) is not implemented.

## Reference

- Original QLib source: `reference/qlib/qlib/data/`
- Architecture: `architecture.md` (Chinese)
- QLib docs: https://qlib.readthedocs.io/
