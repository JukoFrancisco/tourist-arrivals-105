import numpy as np
import pandas as pd
import pytest

from backend.pipeline import (
    ROOT, clean_data, encoded_frame, evaluate_model, prepare_data,
    read_csv, score, select_features, validate_forecast_inputs,
)


@pytest.fixture
def lab():
    raw, meta = read_csv(ROOT / "data" / "tourist_arrivals.csv")
    clean, report = clean_data(raw)
    return raw, meta, clean, report


def test_supplied_export_and_monthly_grid(lab):
    raw, meta, clean, report = lab
    assert meta["header_lines_skipped"] == 2
    assert len(raw) == 314 and len(clean) == 312
    assert report["duplicates_removed"] == 2
    assert clean.date.is_unique
    assert np.array_equal(clean.date.to_numpy(), pd.date_range("2000-01-01", "2025-12-01", freq="MS").to_numpy())
    assert clean.arrivals.isna().sum() == 3
    assert clean.arrivals.max() > 30_000_000  # Extreme source observations are retained.


def test_future_data_cannot_change_selection_or_scalers(lab):
    _, _, clean, _ = lab
    first = select_features(clean)
    prep = prepare_data(clean, first)
    changed = clean.copy()
    start = first["val_start"]
    changed.loc[start:, "arrivals"] *= 1_000
    changed.loc[start:, "rainfall_mm"] = 999_999
    second = select_features(changed)
    prep_changed = prepare_data(changed, second)
    pd.testing.assert_frame_equal(first["spearman"], second["spearman"])
    assert first["selected"] == second["selected"]
    np.testing.assert_equal(prep["scaler_X"].data_max_, prep_changed["scaler_X"].data_max_)
    np.testing.assert_equal(prep["scaler_y"].data_max_, prep_changed["scaler_y"].data_max_)
    np.testing.assert_equal(prep["sequences"]["fit"]["X"], prep_changed["sequences"]["fit"]["X"])


def test_window_target_alignment_and_missing_targets(lab):
    _, _, clean, _ = lab
    selection = select_features(clean)
    prep = prepare_data(clean, selection)
    for part in prep["sequences"].values():
        for j, idx in enumerate(part["indices"]):
            assert np.isfinite(clean.arrivals.iloc[idx])
            expected = prep["scaler_X"].transform(prep["inputs"].iloc[idx-12:idx])
            np.testing.assert_allclose(part["X"][j], expected, rtol=1e-6, atol=1e-6)
            target = prep["scaler_y"].inverse_transform(part["y"][j:j+1])[0, 0]
            assert target == pytest.approx(clean.arrivals.iloc[idx], rel=1e-5, abs=5)
            assert part["dates"][j] == clean.date.iloc[idx]


def test_all_methods_share_dates_and_baselines(lab):
    _, _, clean, _ = lab
    prep = prepare_data(clean, select_features(clean))
    seq = prep["sequences"]["test"]
    predict = lambda model, X, scaler: clean.arrivals.iloc[seq["indices"]].to_numpy()
    result = evaluate_model(None, prep, clean, predict)
    table = result["predictions"]
    assert result["metrics"].loc["LSTM", "MAE"] == 0
    for _, row in table.iterrows():
        i = clean.index[clean.date.eq(row.date)][0]
        assert row["Naive"] == clean.arrivals.iloc[i-1]
        assert row["Seasonal naive"] == clean.arrivals.iloc[i-12]


def test_zero_actual_metrics_and_constant_targets():
    result = score([0, 10], [2, 8])
    assert result["MAE"] == 2
    assert result["MAPE (%)"] == 20
    assert np.isnan(score([0, 0], [1, 1])["MAPE (%)"])
    assert np.isnan(score([3, 3], [2, 2])["R²"])


def test_fixed_category_columns_and_no_constant_selection(lab):
    _, _, clean, _ = lab
    encoded = encoded_frame(clean, True)
    assert set(["season_wet", "season_transition", "monsoon_habagat", "monsoon_transition"]).issubset(encoded)
    selection = select_features(clean, include_categories=True)
    assert selection["selected"]
    changed = clean.copy()
    changed["is_holiday_peak"] = 0
    result = select_features(changed)
    assert "is_holiday_peak" not in result["selected"]


def test_forecast_rejects_bad_values_and_dates(lab):
    _, _, clean, _ = lab
    sample = clean[["date", "rainfall_mm", "humidity_pct"]].tail(12).copy()
    validate_forecast_inputs(sample, ["rainfall_mm", "humidity_pct"], 12)
    sample.iloc[-1, sample.columns.get_loc("rainfall_mm")] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_forecast_inputs(sample, ["rainfall_mm", "humidity_pct"], 12)
