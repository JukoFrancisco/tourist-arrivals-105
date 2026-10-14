import numpy as np
import pandas as pd
import streamlit as st

from backend.modeling import predict_arrivals
from backend.pipeline import LABELS, validate_forecast_inputs
from frontend.ui import setup, require, stage_footer, time_chart, csv_download

setup("Forecast", "Use the latest history or edit a scenario to predict next month's arrivals.")
require("model", "Train")
p = st.session_state.prepared
df = st.session_state.clean_df
features, lookback = p["features"], p["lookback"]
default = p["inputs"].tail(lookback).copy()
default.insert(0, "date", df.date.tail(lookback).to_numpy())
target = pd.Timestamp(default.date.iloc[-1]) + pd.offsets.MonthBegin(1)
a, b = st.columns(2)
a.metric("Forecast month", f"{target:%B %Y}")
b.metric("Historical inputs", f"{lookback} months")
st.caption(f"History: {default.date.iloc[0]:%b %Y} – {default.date.iloc[-1]:%b %Y}. Edits are hypothetical historical readings.")
physical_features = [c for c in features if c in df.columns]
if physical_features and df[physical_features].tail(lookback).isna().any().any():
    st.caption("Some defaults use the saved forward-fill / fitting-median preprocessing.")

# A new editor key explicitly clears edits; downstream invalidation also removes them.
revision = st.session_state.get("forecast_editor_revision", 0)
if st.button("Reset inputs to latest history", icon=":material/restart_alt:"):
    st.session_state.forecast_editor_revision = revision + 1
    st.session_state.pop("forecast_report", None)
    st.rerun()
calendar = [c for c in features if c == "quarter" or c.startswith("season_") or c.startswith("monsoon_") or c == "is_holiday_peak"]
with st.form("forecast_inputs"):
    config = {"date": st.column_config.DateColumn("Month", format="MMM YYYY")}
    for col in features:
        config[col] = st.column_config.NumberColumn(LABELS.get(col, col), format="%.2f")
    edited = st.data_editor(default, key=f"forecast_editor_{revision}", disabled=["date", *calendar], hide_index=True,
                            num_rows="fixed", column_config=config, width="stretch")
    st.caption("Calendar and category indicators stay fixed. Edit climate or hazard readings to explore a scenario.")
    submitted = st.form_submit_button("Forecast next month", type="primary", icon=":material/trending_up:")

if submitted:
    try:
        inputs = validate_forecast_inputs(edited, features, lookback)
        scaled = p["scaler_X"].transform(inputs)
        outside = [features[i] for i in range(len(features))
                   if ((scaled[:, i] < 0) | (scaled[:, i] > 1)).any()]
        sequence = scaled.reshape(1, lookback, len(features))
        prediction = float(predict_arrivals(st.session_state.model, sequence, p["scaler_y"])[0])
        baseline_X = p["scaler_X"].transform(default[features]).reshape(1, lookback, len(features))
        baseline = float(predict_arrivals(st.session_state.model, baseline_X, p["scaler_y"])[0])
        changes = []
        for col in features:
            for pos in np.flatnonzero(~np.isclose(inputs[col].to_numpy(), default[col].to_numpy(), rtol=0, atol=1e-9)):
                changes.append({"month": edited.date.iloc[pos], "predictor": LABELS.get(col, col),
                                "original": float(default[col].iloc[pos]), "scenario": float(inputs[col].iloc[pos])})
        st.session_state.forecast_report = {"date": target, "prediction": prediction, "baseline": baseline,
                                            "outside": outside, "inputs": edited.copy(), "changes": changes}
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if "forecast_report" in st.session_state:
    r = st.session_state.forecast_report
    value = r["prediction"]
    st.subheader("Forecast result")
    a, b, c = st.columns(3)
    a.metric("Unchanged history", f"{r['baseline']:,.0f}")
    b.metric("Your forecast", f"{value:,.0f}")
    c.metric("Scenario change", f"{value-r['baseline']:+,.0f}", help="Difference from the same model using unchanged history. This is model sensitivity, not a causal estimate.")
    st.caption(f"Result uses the last submitted inputs. {len(r['changes'])} changed cells. Submit again after making further edits.")
    if value < 0:
        st.warning("The model produced negative arrivals. This forecast is invalid; review model fit and inputs.")
    if r["outside"]:
        st.warning("Inputs outside the fitting range: " + ", ".join(LABELS.get(x, x) for x in r["outside"]))
    chart = df[["date", "arrivals"]].tail(24).rename(columns={"arrivals": "Observed"}).copy()
    chart["Forecast"] = np.nan
    chart = pd.concat([chart, pd.DataFrame({"date": [r["date"]], "Observed": [np.nan],
                                          "Forecast": [value]})], ignore_index=True)
    if len(chart) > 1 and pd.notna(chart.Observed.iloc[-2]):
        chart.loc[chart.index[-2], "Forecast"] = chart.Observed.iloc[-2]
    time_chart(chart, ["Observed", "Forecast"])
    if r["changes"]:
        with st.expander("Scenario changes", icon=":material/edit:"):
            st.dataframe(pd.DataFrame(r["changes"]), hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        csv_download(pd.DataFrame({"date": [r["date"]], "predicted_arrivals": [value],
                                  "unchanged_prediction": [r["baseline"]], "difference": [value-r["baseline"]]}),
                     "next_month_forecast.csv", "Download forecast")
    with b:
        csv_download(r["inputs"], "forecast_inputs.csv", "Download submitted inputs")
    st.caption("Point forecast from the saved model. No uncertainty interval has been estimated.")

stage_footer("Forecast")
