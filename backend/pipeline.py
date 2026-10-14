"""Pure data operations shared by the eight Streamlit pages."""
from __future__ import annotations

import hashlib
import io
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

ROOT = Path(__file__).resolve().parents[1]
DATA_URL = "https://drive.google.com/drive/folders/1b8KgyEbT1E6EkpEEKOkiOe29JnHLFbml?usp=sharing"
CANDIDATES = [
    "quarter", "is_holiday_peak", "temp_mean_c", "temp_min_c", "temp_max_c",
    "rainfall_mm", "rainy_days", "humidity_pct", "typhoon_count",
    "typhoon_max_wind_kt", "storm_signal_days", "pm25_ugm3", "wave_height_m",
]
EXPECTED = ["date", "year", "month", "quarter", "season", "monsoon",
            *CANDIDATES[1:], "arrivals"]
CATEGORY_LEVELS = {
    "season": ["Dry", "Wet", "Transition"],
    "monsoon": ["Amihan", "Habagat", "Transition"],
}
LABELS = {
    "quarter": "Quarter", "is_holiday_peak": "Holiday peak",
    "temp_mean_c": "Mean temperature (°C)", "temp_min_c": "Minimum temperature (°C)",
    "temp_max_c": "Maximum temperature (°C)", "rainfall_mm": "Rainfall (mm)",
    "rainy_days": "Rainy days", "humidity_pct": "Humidity (%)",
    "typhoon_count": "Typhoon count", "typhoon_max_wind_kt": "Typhoon wind (kt)",
    "storm_signal_days": "Storm signal days", "pm25_ugm3": "PM2.5 (µg/m³)",
    "wave_height_m": "Wave height (m)",
}


def read_csv(source):
    """The supplied export has two title lines above the real CSV header."""
    content = source if isinstance(source, bytes) else Path(source).read_bytes()
    text = content.decode("utf-8-sig")
    lines = text.splitlines()
    offset = next((i for i, line in enumerate(lines[:20])
                   if line.strip().lower().startswith("date,")), None)
    if offset is None:
        raise ValueError("CSV header must contain date, year, month and the lab columns.")
    df = pd.read_csv(io.StringIO("\n".join(lines[offset:])))
    df.columns = df.columns.str.strip().str.lower()
    if df.columns.duplicated().any():
        raise ValueError("Duplicate column names are not supported.")
    missing = sorted(set(EXPECTED) - set(df.columns))
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    df = df[EXPECTED].copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    if df["date"].isna().any():
        raise ValueError("Some dates cannot be read. Correct them before loading.")
    if not (df["date"].dt.day == 1).all():
        raise ValueError("Dates must be the first day of each month.")
    if df.empty:
        raise ValueError("The CSV has no observations.")
    invalid_numeric = {}
    for col in ["year", "month", *CANDIDATES, "arrivals"]:
        before = df[col].notna()
        converted = pd.to_numeric(df[col], errors="coerce")
        invalid_numeric[col] = int((before & converted.isna()).sum())
        df[col] = converted.replace([-999, -9999, np.inf, -np.inf], np.nan)
    for col, levels in CATEGORY_LEVELS.items():
        df[col] = df[col].astype("string").str.strip().replace("", pd.NA)
        unexpected = set(df[col].dropna().unique()) - set(levels)
        if unexpected:
            raise ValueError(f"Unexpected {col} categories: {sorted(unexpected)}")
    meta = {"header_lines_skipped": offset,
            "invalid_numeric": {k: v for k, v in invalid_numeric.items() if v},
            "sha256": hashlib.sha256(content).hexdigest()}
    return df.sort_values("date", kind="stable").reset_index(drop=True), meta


def clean_data(raw):
    df = raw.copy()
    duplicates = df[df.duplicated("date", keep=False)].copy()
    removed = int(df.duplicated("date").sum())
    df = df.drop_duplicates("date", keep="first").set_index("date")
    full_dates = pd.date_range(df.index.min(), df.index.max(), freq="MS")
    missing_months = full_dates.difference(df.index)
    df = df.reindex(full_dates).rename_axis("date").reset_index()
    calendar_fixes = {}
    for col, expected in {"year": df.date.dt.year, "month": df.date.dt.month,
                          "quarter": df.date.dt.quarter}.items():
        calendar_fixes[col] = int((df[col].isna() | df[col].ne(expected)).sum())
        df[col] = expected
    # Invalid physical values become missing, not invented zero measurements.
    invalid = {}
    for col in ["arrivals", "rainfall_mm", "rainy_days", "typhoon_count",
                "typhoon_max_wind_kt", "storm_signal_days", "pm25_ugm3", "wave_height_m"]:
        mask = df[col] < 0
        invalid[col] = int(mask.sum())
        df.loc[mask, col] = np.nan
    for col, mask in {
        "humidity_pct": ~df.humidity_pct.between(0, 100) & df.humidity_pct.notna(),
        "is_holiday_peak": ~df.is_holiday_peak.isin([0, 1]) & df.is_holiday_peak.notna(),
        "rainy_days": df.rainy_days > df.date.dt.days_in_month,
        "storm_signal_days": df.storm_signal_days > df.date.dt.days_in_month,
    }.items():
        invalid[col] = int(mask.sum())
        df.loc[mask, col] = np.nan
    q1, q3 = df.arrivals.quantile([0.25, 0.75])
    lower, upper = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    flagged = df.loc[~df.arrivals.between(lower, upper) & df.arrivals.notna(),
                     ["date", "arrivals"]]
    report = {"duplicates": duplicates, "duplicates_removed": removed,
              "missing_months": list(missing_months),
              "missing": df.isna().sum().rename("Missing values"),
              "flagged": flagged, "iqr_bounds": (lower, upper),
              "invalid": {k: v for k, v in invalid.items() if v},
              "calendar_fixes": {k: v for k, v in calendar_fixes.items() if v}}
    return df, report


def split_positions(n, development_ratio=0.8, validation_ratio=0.15):
    test_start = int(n * development_ratio)
    val_start = int(test_start * (1 - validation_ratio))
    if min(val_start, test_start - val_start, n - test_start) < 3:
        raise ValueError("Too few months for fitting, validation and test partitions.")
    return val_start, test_start


def encoded_frame(df, include_categories=False):
    out = df[CANDIDATES].copy()
    if include_categories:
        for col, levels in CATEGORY_LEVELS.items():
            # Fixed schema vocabulary; no vocabulary learned from future rows.
            for level in levels[1:]:
                name = f"{col}_{level.lower()}"
                out[name] = np.where(df[col].isna(), np.nan,
                                     (df[col] == level).fillna(False).astype(float))
    return out.astype(float)


def causal_inputs(df, include_categories=False):
    return encoded_frame(df, include_categories).ffill(limit=2)


def select_features(df, development_ratio=.8, validation_ratio=.15, include_categories=False):
    val_start, test_start = split_positions(len(df), development_ratio, validation_ratio)
    raw_X = causal_inputs(df, include_categories)
    fit_X = raw_X.iloc[:val_start]
    y = df.arrivals.iloc[:val_start]
    # Remaining gaps use fitting-period medians. Targets are never imputed.
    medians = fit_X.median()
    X = fit_X.fillna(medians)
    rows, kept = [], []
    for col in X:
        valid = X[col].notna() & y.notna()
        if valid.sum() < 3 or X.loc[valid, col].nunique() < 2 or y[valid].nunique() < 2:
            rho, p, keep = np.nan, np.nan, False
        else:
            rho, p = spearmanr(X.loc[valid, col], y[valid])
            keep = bool(abs(rho) > .10 and p < .05)
        rows.append({"feature": col, "rho": rho, "p_value": p,
                     "n": int(valid.sum()), "kept": keep})
        if keep:
            kept.append(col)
    log = []
    # Use the same fitting rows for all VIF regressions, including an intercept.
    vif_X = X[kept].loc[y.notna()].dropna()
    if kept and len(vif_X) <= len(kept) + 1:
        raise ValueError("Too few fitting observations for VIF. Use a longer series.")
    while len(kept) > 1:
        matrix = add_constant(vif_X[kept], has_constant="add").to_numpy(dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            vifs = np.array([variance_inflation_factor(matrix, i + 1)
                             for i in range(len(kept))])
        if np.isnan(vifs).any():
            raise ValueError("VIF is undefined. Check constant or invalid fitting columns.")
        for col, value in zip(kept, vifs):
            log.append({"iteration": 1 + sum(r.get("removed", False) for r in log),
                        "feature": col, "VIF": float(value), "removed": False})
        worst = int(np.argmax(vifs))
        if vifs[worst] < 5:
            break
        for r in reversed(log):
            if r["feature"] == kept[worst]:
                r["removed"] = True
                break
        kept.pop(worst)
    return {"selected": kept, "spearman": pd.DataFrame(rows),
            "vif": pd.DataFrame(log, columns=["iteration", "feature", "VIF", "removed"]),
            "medians": medians, "val_start": val_start, "test_start": test_start,
            "development_ratio": development_ratio, "validation_ratio": validation_ratio,
            "include_categories": include_categories}


def prepare_data(df, selection, lookback=12):
    features = selection["selected"]
    if not features:
        raise ValueError("No predictors passed selection. Check the data; thresholds stay unchanged.")
    a, b = selection["val_start"], selection["test_start"]
    if min(a, b - a, len(df) - b) <= lookback:
        raise ValueError("Each partition needs more months than the lookback. Adjust the split or lookback.")
    X = causal_inputs(df, selection["include_categories"])[features]
    medians = X.iloc[:a].median()
    X = X.fillna(medians)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Some selected predictors have no valid fitting values.")
    valid_fit_y = df.arrivals.iloc[:a].dropna()
    if len(valid_fit_y) < 3:
        raise ValueError("Too few known fitting targets.")
    scaler_X = MinMaxScaler().fit(X.iloc[:a])
    scaler_y = MinMaxScaler().fit(valid_fit_y.to_numpy().reshape(-1, 1))
    Xs = scaler_X.transform(X)
    ys = (df.arrivals.to_numpy(dtype=float) - scaler_y.data_min_[0]) * scaler_y.scale_[0]
    sequences = {}
    # Separate windows match the handout; no target spans a partition boundary.
    for name, start, stop in [("fit", 0, a), ("val", a, b), ("test", b, len(df))]:
        indices = [i for i in range(start + lookback, stop) if np.isfinite(ys[i])]
        if len(indices) < 2:
            raise ValueError(f"Not enough valid {name} targets after windowing.")
        sequences[name] = {
            "X": np.asarray([Xs[i-lookback:i] for i in indices], dtype=np.float32),
            "y": np.asarray([ys[i] for i in indices], dtype=np.float32).reshape(-1, 1),
            "dates": df.date.iloc[indices].to_numpy(), "indices": np.asarray(indices),
        }
    summary = pd.DataFrame([
        {"partition": name, "rows": stop-start,
         "from": df.date.iloc[start].date(), "to": df.date.iloc[stop-1].date(),
         "windows": len(sequences[name]["X"]),
         "first_target": pd.Timestamp(sequences[name]["dates"][0]).date(),
         "last_target": pd.Timestamp(sequences[name]["dates"][-1]).date()}
        for name, start, stop in [("fit", 0, a), ("val", a, b), ("test", b, len(df))]
    ])
    return {"features": features, "lookback": lookback, "scaler_X": scaler_X,
            "scaler_y": scaler_y, "medians": medians, "inputs": X,
            "sequences": sequences, "summary": summary}


def score(actual, predicted):
    a, p = np.asarray(actual).reshape(-1), np.asarray(predicted).reshape(-1)
    error = a - p
    nz = a != 0
    total = float(np.sum((a - a.mean()) ** 2))
    return {"MAE": float(np.abs(error).mean()),
            "RMSE": float(np.sqrt(np.mean(error ** 2))),
            "MAPE (%)": float(np.mean(np.abs(error[nz] / a[nz])) * 100) if nz.any() else np.nan,
            "R²": float(1 - np.sum(error**2) / total) if total > 0 and len(a) > 1 else np.nan}


def evaluate_model(model, prep, df, predict):
    seq = prep["sequences"]["test"]
    prediction = predict(model, seq["X"], prep["scaler_y"])
    indices = seq["indices"]
    table = pd.DataFrame({"date": pd.to_datetime(seq["dates"]),
                          "Actual": df.arrivals.iloc[indices].to_numpy(),
                          "LSTM": prediction,
                          "Naive": df.arrivals.shift(1).iloc[indices].to_numpy(),
                          "Seasonal naive": df.arrivals.shift(12).iloc[indices].to_numpy()})
    valid = np.isfinite(table[["Actual", "LSTM", "Naive", "Seasonal naive"]]).all(axis=1)
    common = table.loc[valid].copy()
    if len(common) < 2:
        raise ValueError("Fewer than two common target dates are available for all three methods.")
    metrics = pd.DataFrame({name: score(common.Actual, common[name])
                            for name in ["LSTM", "Naive", "Seasonal naive"]}).T
    return {"predictions": common, "metrics": metrics,
            "excluded": int((~valid).sum()), "zero_actuals": int((common.Actual == 0).sum())}


def validate_forecast_inputs(edited, features, lookback):
    if len(edited) != lookback:
        raise ValueError(f"Provide exactly {lookback} historical months.")
    dates = pd.to_datetime(edited["date"])
    expected = pd.date_range(dates.iloc[0], periods=lookback, freq="MS")
    if not np.array_equal(dates.to_numpy(), expected.to_numpy()):
        raise ValueError("Historical months must be consecutive and ordered.")
    X = edited[features].astype(float)
    if not np.isfinite(X.to_numpy()).all():
        raise ValueError("Fill every input with a finite number.")
    for col in features:
        v = X[col]
        if col in {"rainfall_mm", "rainy_days", "typhoon_count", "typhoon_max_wind_kt",
                   "storm_signal_days", "pm25_ugm3", "wave_height_m"} and (v < 0).any():
            raise ValueError(f"{col} cannot be negative.")
        if col == "humidity_pct" and not v.between(0, 100).all():
            raise ValueError("Humidity must be between 0 and 100%.")
        if col == "quarter" and not v.isin([1, 2, 3, 4]).all():
            raise ValueError("Quarter must be 1, 2, 3 or 4.")
        if (col == "is_holiday_peak" or col.startswith(("season_", "monsoon_"))) and not v.isin([0, 1]).all():
            raise ValueError(f"{col} must be 0 or 1.")
        if col in {"rainy_days", "storm_signal_days"} and (v.to_numpy() > dates.dt.days_in_month.to_numpy()).any():
            raise ValueError(f"{col} exceeds the number of days in its month.")
        if col in {"rainy_days", "storm_signal_days", "typhoon_count"} and not (v % 1 == 0).all():
            raise ValueError(f"{col} must contain whole numbers.")
    for prefix in ["season_", "monsoon_"]:
        group = [c for c in features if c.startswith(prefix)]
        if len(group) > 1 and (X[group].sum(axis=1) > 1).any():
            raise ValueError(f"Choose at most one active {prefix.rstrip('_')} indicator per month.")
    return X
