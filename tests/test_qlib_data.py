# Copyright (c) 2026
# Licensed under the MIT License

"""Tests for qlib_data package."""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import qlib_data

data_path = ""


def test_dump_and_read():
    """Test dumping CSV data and reading it back."""
    # Create temp directory for test data
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(__file__).parent / "HK.00100.csv"
        # qlib_dir = Path(tmpdir) / "test_data"
        qlib_dir = Path(data_path)

        print(f"CSV path: {csv_path}")
        print(f"Using temporary directory: {qlib_dir}")

        # Dump the CSV data
        qlib_data.dump_dataset(
            csv_path=csv_path,
            qlib_dir=qlib_dir,
            freq="day",
            date_field="date",
            symbol_field="symbol",
        )

        # Verify directories created
        assert (qlib_dir / "calendars").exists()
        assert (qlib_dir / "instruments").exists()
        assert (qlib_dir / "features").exists()

        # Verify calendar
        cal_file = qlib_dir / "calendars" / "day.txt"
        assert cal_file.exists()
        with open(cal_file) as f:
            dates = [line.strip() for line in f if line.strip()]
        assert len(dates) > 0
        assert dates[0] == "2026-01-09"  # First date in CSV

        # Verify instruments
        inst_file = qlib_dir / "instruments" / "all.txt"
        assert inst_file.exists()
        with open(inst_file) as f:
            lines = [line.strip() for line in f if line.strip()]
        assert any("hk.00100" in line.lower() for line in lines)

        # Verify features
        feat_dir = qlib_dir / "features" / "hk.00100"
        assert feat_dir.exists()
        bin_files = list(feat_dir.glob("*.bin"))
        assert len(bin_files) > 0

        # Initialize and read back
        qlib_data.init(str(qlib_dir))

        # Test reading features
        df = qlib_data.load_dataset(["hk.00100"], ["$close"], "2026-01-01", "2026-03-01")
        assert not df.empty
        assert "instrument" in df.columns
        assert "datetime" in df.columns
        assert "$close" in df.columns

        # Test reading multiple features
        df2 = qlib_data.load_dataset("hk.00100", ["$close", "$open", "$volume"])
        assert "$close" in df2.columns
        assert "$open" in df2.columns
        assert "$volume" in df2.columns

        print("All tests passed!")


def test_expression_parsing():
    """Test expression parsing."""
    from qlib_data.expression.parser import parse_field, parse_fields

    # Test simple field
    expr = parse_field("$close")
    assert str(expr) == "$close"

    # Test expression with operator
    expr = parse_field("Ref($close, -1)")
    assert "Ref" in str(expr)

    # Test list of fields
    exprs = parse_fields(["$close", "$open"])
    assert len(exprs) == 2


def test_config():
    global data_path

    """Test configuration."""
    config = qlib_data.C
    config.reset()
    assert config.provider_uri is None

    config.initialize("/tmp/test")
    assert config.provider_uri == "/tmp/test"
    data_path = config.provider_uri


def _build_extra_csv(tmpdir: Path) -> Path:
    """Build a CSV containing only the two extra dates for HK.00100."""
    extra_rows = pd.DataFrame(
        {
            "date": ["2026-06-02", "2026-06-03"],
            "symbol": ["HK.00100", "HK.00100"],
            "open": [700.0, 710.0],
            "high": [720.0, 730.0],
            "low": [690.0, 700.0],
            "close": [715.0, 725.0],
            "volume": [1234567, 2345678],
        }
    )
    extra_path = tmpdir / "HK.00100_extra.csv"
    extra_rows.to_csv(extra_path, index=False)
    return extra_path


def test_incremental_dump():
    """``incremental=True`` must merge new data with the on-disk dataset."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        qlib_dir = tmpdir / "incr"
        csv_path = Path(__file__).parent / "HK.00100.csv"

        # First dump: full.
        qlib_data.dump_dataset(
            csv_path=csv_path,
            qlib_dir=qlib_dir,
            freq="day",
            date_field="date",
            symbol_field="symbol",
        )
        cal_file = qlib_dir / "calendars" / "day.txt"
        with open(cal_file) as f:
            original_dates = [line.strip() for line in f if line.strip()]
        assert len(original_dates) == 94

        # Second dump: incremental with extra rows.
        extra_path = _build_extra_csv(tmpdir)
        qlib_data.dump_dataset(
            csv_path=extra_path,
            qlib_dir=qlib_dir,
            freq="day",
            date_field="date",
            symbol_field="symbol",
            incremental=True,
        )

        with open(cal_file) as f:
            merged_dates = [line.strip() for line in f if line.strip()]
        # The extra CSV carries the original 15 rows + 2 new dates;
        # the merged calendar should contain the original 94 dates plus
        # the two new ones.
        assert len(merged_dates) == 96
        assert "2026-06-02" in merged_dates
        assert "2026-06-03" in merged_dates
        assert merged_dates[0] == original_dates[0]

        # The bin file should now hold 96 values and the new values for
        # 2026-06-02 / 2026-06-03 must equal the CSV.
        feat_file = qlib_dir / "features" / "hk.00100" / "close.day.bin"
        assert feat_file.exists()
        with open(feat_file, "rb") as f:
            raw = np.fromfile(f, dtype="<f")
        assert len(raw) == 96 + 1  # 1 for start_index
        assert int(raw[0]) == 0
        # The last two values correspond to 2026-06-02 and 2026-06-03.
        assert raw[-2] == 715.0
        assert raw[-1] == 725.0

        # The first value (oldest date) must still be preserved.
        assert raw[1] == 345.0

        # Instruments file should still have HK.00100 with an extended
        # end date covering the new rows.
        with open(qlib_dir / "instruments" / "all.txt") as f:
            line = f.readline().strip()
        parts = line.split("\t")
        assert parts[0] == "hk.00100"
        assert parts[1] == "2026-01-09"
        assert parts[2] == "2026-06-03"


def test_tick_frequency():
    """``freq="tick"`` must use the tick timestamp format on disk."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        qlib_dir = tmpdir / "ticks"
        qlib_dir.mkdir(parents=True, exist_ok=True)

        # Build a tiny tick dataset in-memory.
        rows = [
            ("2026-01-09 09:30:00.123456", "HK.00100", 100.0, 101.0, 99.5, 100.5, 1000),
            ("2026-01-09 09:30:00.500000", "HK.00100", 100.5, 102.0, 100.0, 101.5, 2000),
            ("2026-01-09 09:30:01.000001", "HK.00100", 101.5, 103.0, 101.0, 102.5, 1500),
        ]
        df = pd.DataFrame(rows, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        csv_path = tmpdir / "HK.00100_ticks.csv"
        df.to_csv(csv_path, index=False)

        qlib_data.dump_dataset(
            csv_path=csv_path,
            qlib_dir=qlib_dir,
            freq="tick",
            date_field="date",
            symbol_field="symbol",
        )

        cal_file = qlib_dir / "calendars" / "tick.txt"
        assert cal_file.exists()
        with open(cal_file) as f:
            entries = [line.strip() for line in f if line.strip()]
        # Tick format: "%Y-%m-%d %H:%M:%S.%f" (microsecond resolution)
        assert entries == [
            "2026-01-09 09:30:00.123456",
            "2026-01-09 09:30:00.500000",
            "2026-01-09 09:30:01.000001",
        ]

        feat_file = qlib_dir / "features" / "hk.00100" / "close.tick.bin"
        assert feat_file.exists()
        with open(feat_file, "rb") as f:
            raw = np.fromfile(f, dtype="<f")
        assert int(raw[0]) == 0
        assert list(raw[1:]) == [100.5, 101.5, 102.5]

        qlib_data.init(str(qlib_dir))
        out = qlib_data.load_dataset(
            ["hk.00100"], ["$close"], freq="tick"
        )
        assert not out.empty
        assert len(out) == 3
        assert out["$close"].tolist() == [100.5, 101.5, 102.5]


def test_incremental_tick_append():
    """Appending tick rows with ``incremental=True`` must extend the bin."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        qlib_dir = tmpdir / "ticks_incr"
        qlib_dir.mkdir(parents=True, exist_ok=True)

        rows1 = [
            ("2026-01-09 09:30:00.100000", "HK.00100", 100.0, 101.0, 99.5, 100.5, 1000),
            ("2026-01-09 09:30:00.200000", "HK.00100", 100.5, 102.0, 100.0, 101.5, 2000),
        ]
        rows2 = [
            ("2026-01-09 09:30:00.300000", "HK.00100", 101.5, 103.0, 101.0, 102.5, 1500),
            ("2026-01-09 09:30:00.400000", "HK.00100", 102.5, 104.0, 102.0, 103.5, 1700),
        ]
        df1 = pd.DataFrame(rows1, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        df2 = pd.DataFrame(rows2, columns=["date", "symbol", "open", "high", "low", "close", "volume"])
        p1 = tmpdir / "ticks1.csv"
        p2 = tmpdir / "ticks2.csv"
        df1.to_csv(p1, index=False)
        df2.to_csv(p2, index=False)

        qlib_data.dump_dataset(csv_path=p1, qlib_dir=qlib_dir, freq="tick")
        qlib_data.dump_dataset(
            csv_path=p2, qlib_dir=qlib_dir, freq="tick", incremental=True
        )

        with open(qlib_dir / "calendars" / "tick.txt") as f:
            entries = [line.strip() for line in f if line.strip()]
        assert len(entries) == 4
        assert entries[2] == "2026-01-09 09:30:00.300000"

        feat_file = qlib_dir / "features" / "hk.00100" / "close.tick.bin"
        with open(feat_file, "rb") as f:
            raw = np.fromfile(f, dtype="<f")
        assert len(raw) == 4 + 1
        assert list(raw[1:]) == [100.5, 101.5, 102.5, 103.5]


if __name__ == "__main__":
    test_config()
    test_expression_parsing()
    test_dump_and_read()
    test_incremental_dump()
    test_tick_frequency()
    test_incremental_tick_append()
