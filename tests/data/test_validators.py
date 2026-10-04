from __future__ import annotations

from datetime import date, datetime

import pandas as pd
import pytest
import yaml

from src.data.validators import DataValidationError, _parse_date, validate_daily_frame


@pytest.fixture
def valid_frame():
    return pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01T00:00:00+00:00", "2020-01-02T00:00:00+00:00"]),
            "open": [100.0, 102.0],
            "high": [110.0, 110.5],
            "low": [95.0, 100.0],
            "close": [108.0, 109.0],
            "volume": [2000, 2100],
            "open_interest": [0, 0],
        }
    )


def test_validate_daily_frame_success(valid_frame):
    result = validate_daily_frame(valid_frame, from_date="2020-01-01", to_date="2020-01-02")
    assert result["valid"] is True
    assert result["record_count"] == 2


def test_parse_date_accepts_yaml_loaded_date_object():
    loaded = yaml.safe_load("start_date: 2020-01-01\nend_date: 2020-01-02\n")
    assert _parse_date(loaded["start_date"]) == datetime(2020, 1, 1)
    assert _parse_date(loaded["end_date"]) == datetime(2020, 1, 2)

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2020-01-01T00:00:00+00:00", "2020-01-02T00:00:00+00:00"]),
            "open": [100.0, 102.0],
            "high": [110.0, 110.5],
            "low": [95.0, 100.0],
            "close": [108.0, 109.0],
            "volume": [2000, 2100],
            "open_interest": [0, 0],
        }
    )
    result = validate_daily_frame(frame, from_date=date(2020, 1, 1), to_date=date(2020, 1, 2))
    assert result["valid"] is True


def test_validate_daily_frame_rejects_invalid_ohlc(valid_frame):
    valid_frame.loc[0, "high"] = 90.0
    with pytest.raises(DataValidationError):
        validate_daily_frame(valid_frame)


def test_validate_daily_frame_rejects_negative_volume(valid_frame):
    valid_frame.loc[0, "volume"] = -1
    with pytest.raises(DataValidationError):
        validate_daily_frame(valid_frame)
