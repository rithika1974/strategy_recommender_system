from __future__ import annotations

import sqlite3

import pandas as pd

from src.augmentation.rolling_windows import (
    AugmentationConfig,
    DateSplit,
    build_augmentation_windows,
    generate_window_metadata,
)


def config() -> AugmentationConfig:
    return AugmentationConfig(
        window_length=60,
        training_stride=30,
        evaluation_stride=60,
        train=DateSplit(pd.Timestamp("2020-01-01").date(), pd.Timestamp("2020-12-31").date()),
        validation=DateSplit(pd.Timestamp("2021-01-01").date(), pd.Timestamp("2021-12-31").date()),
        test=DateSplit(pd.Timestamp("2022-01-01").date(), None),
        regime_source="window_end",
        null_handling="reject_active_feature_nulls",
        ignored_null_columns=("repo_rate", "cpi"),
        seed=42,
    )


def feature_frame(symbols: tuple[str, ...] = ("AAA", "BBB")) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        for split_start in ("2020-01-01", "2021-01-01", "2022-01-01"):
            dates = pd.date_range(split_start, periods=120, freq="D")
            frames.append(
                pd.DataFrame(
                    {
                        "symbol": symbol,
                        "date": dates.strftime("%Y-%m-%d"),
                        "feature_value": range(1, 121),
                        "repo_rate": pd.NA,
                        "cpi": pd.NA,
                        "market_trend_regime": [
                            "BULL" if value % 2 else "BEAR" for value in range(1, 121)
                        ],
                        "volatility_regime": [
                            "HIGH" if value % 3 else "LOW" for value in range(1, 121)
                        ],
                    }
                )
            )
    return pd.concat(frames, ignore_index=True)


def test_window_geometry_stride_overlap_count_and_split_safety():
    features = feature_frame()
    windows, summary = generate_window_metadata(features, config())

    assert len(windows) == 14
    assert summary["candidate_windows"] == 14
    assert summary["rejected_windows"] == 0
    assert {window["window_length"] for window in windows} == {60}
    assert {window["stride"] for window in windows if window["split"] == "train"} == {30}
    assert {
        window["stride"] for window in windows if window["split"] in {"validation", "test"}
    } == {60}

    for symbol in ("AAA", "BBB"):
        train = [w for w in windows if w["symbol"] == symbol and w["split"] == "train"]
        validation = [
            w for w in windows if w["symbol"] == symbol and w["split"] == "validation"
        ]
        test = [w for w in windows if w["symbol"] == symbol and w["split"] == "test"]
        assert pd.Timestamp(train[1]["window_start_date"]) == pd.Timestamp("2020-01-31")
        assert pd.Timestamp(train[0]["window_end_date"]) == pd.Timestamp("2020-02-29")
        assert pd.Timestamp(validation[1]["window_start_date"]) > pd.Timestamp(
            validation[0]["window_end_date"]
        )
        assert pd.Timestamp(test[1]["window_start_date"]) > pd.Timestamp(
            test[0]["window_end_date"]
        )

    split_bounds = {
        "train": ("2020-01-01", "2020-12-31"),
        "validation": ("2021-01-01", "2021-12-31"),
        "test": ("2022-01-01", "9999-12-31"),
    }
    for window in windows:
        lower, upper = split_bounds[window["split"]]
        assert lower <= window["window_start_date"] <= window["window_end_date"] <= upper
        source = features[
            (features["symbol"] == window["symbol"])
            & (features["date"] >= window["window_start_date"])
            & (features["date"] <= window["window_end_date"])
        ]
        assert len(source) == 60
        assert source["date"].is_monotonic_increasing


def test_windows_stay_within_symbols_use_end_regime_and_are_deterministic():
    features = feature_frame()
    first, _ = generate_window_metadata(features.sample(frac=1, random_state=7), config())
    second, _ = generate_window_metadata(features, config())

    assert first == second
    assert len({window["sample_id"] for window in first}) == len(first)
    for window in first:
        end = features[
            (features["symbol"] == window["symbol"])
            & (features["date"] == window["window_end_date"])
        ].iloc[0]
        assert window["market_trend_regime"] == end["market_trend_regime"]
        assert window["volatility_regime"] == end["volatility_regime"]


def test_active_null_rejects_window_but_repo_and_cpi_nulls_are_ignored():
    features = feature_frame(("AAA",)).iloc[:60].copy()
    accepted, accepted_summary = generate_window_metadata(features, config())
    assert len(accepted) == 1
    assert accepted_summary["rejected_windows"] == 0

    features.loc[10, "feature_value"] = pd.NA
    rejected, rejected_summary = generate_window_metadata(features, config())
    assert rejected == []
    assert rejected_summary["rejected_windows"] == 1
    assert rejected_summary["rejection_reasons"] == {"active_feature_null": 1}
    assert rejected_summary["rejection_columns"] == {"feature_value": 1}


def test_database_rebuild_is_compact_idempotent_and_preserves_features(tmp_path):
    database = tmp_path / "market_data.db"
    config_path = tmp_path / "augmentation.yaml"
    config_path.write_text(
        """augmentation:
  window_length: 60
  training_stride: 30
  evaluation_stride: 60
  train: {start_date: '2020-01-01', end_date: '2020-12-31'}
  validation: {start_date: '2021-01-01', end_date: '2021-12-31'}
  test: {start_date: '2022-01-01', end_date: null}
  regime_source: window_end
  null_handling: reject_active_feature_nulls
  ignored_null_columns: [repo_rate, cpi]
  seed: 42
""",
        encoding="utf-8",
    )
    original = feature_frame()
    with sqlite3.connect(database) as connection:
        original.to_sql("features", connection, index=False)
        before = connection.execute(
            "SELECT * FROM features ORDER BY symbol, date"
        ).fetchall()

    first = build_augmentation_windows(database, config_path)
    second = build_augmentation_windows(database, config_path)

    assert first == second
    assert first["total_windows"] == 14
    with sqlite3.connect(database) as connection:
        after = connection.execute(
            "SELECT * FROM features ORDER BY symbol, date"
        ).fetchall()
        stored = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT sample_id) FROM augmentation_windows"
        ).fetchone()
        augmentation_columns = [
            row[1] for row in connection.execute("PRAGMA table_info(augmentation_windows)")
        ]
    assert before == after
    assert stored == (14, 14)
    assert "feature_value" not in augmentation_columns
