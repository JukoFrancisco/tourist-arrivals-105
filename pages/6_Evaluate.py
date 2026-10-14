import streamlit as st

from backend.modeling import predict_arrivals
from backend.pipeline import evaluate_model
from frontend.ui import stage_footer, setup, require, time_chart, csv_download, save_current_run, interpretation

setup("Evaluate", "Compare the LSTM with naive and seasonal-naive forecasts on identical dates.")
require("model", "Train")

if st.button("Evaluate test set", type="primary"):
    try:
        st.session_state.evaluation = evaluate_model(
            st.session_state.model, st.session_state.prepared,
            st.session_state.clean_df, predict_arrivals)
        save_current_run()
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if "evaluation" in st.session_state:
    r = st.session_state.evaluation
    table = r["predictions"]
    st.caption(f"{len(table)} common months · {table.date.min():%b %Y} – {table.date.max():%b %Y} "
               f"· {r['excluded']} unavailable baseline months excluded")
    a, b, c = st.columns(3)
    winner = r["metrics"].MAE.idxmin()
    a.metric("Lowest MAE", winner)
    b.metric("LSTM MAE", f"{r['metrics'].loc['LSTM', 'MAE']:,.0f}", help="Average absolute miss, in arrivals. Lower is better.")
    c.metric("LSTM RMSE", f"{r['metrics'].loc['LSTM', 'RMSE']:,.0f}", help="Penalizes large misses more heavily. Lower is better.")
    st.dataframe(r["metrics"].style.format({"MAE": "{:,.0f}", "RMSE": "{:,.0f}",
                                         "MAPE (%)": "{:.2f}", "R²": "{:.3f}"}), width="stretch")
    shown = st.multiselect("Chart series", ["Actual", "LSTM", "Naive", "Seasonal naive"],
                           default=["Actual", "LSTM", "Naive", "Seasonal naive"])
    if shown:
        time_chart(table, shown)
    else:
        st.info("Select at least one series to show the chart.")
    if r["zero_actuals"]:
        st.warning(f"MAPE excludes {r['zero_actuals']} zero-arrivals months for all methods.")
    st.caption("Negative R² is possible. Very small arrivals make percentage errors unstable. "
               "A baseline winning is a valid result.")
    with st.expander("Residuals and predictions"):
        residuals = table[["date"]].copy()
        residuals["Actual − LSTM"] = table.Actual - table.LSTM
        time_chart(residuals, ["Actual − LSTM"], "Prediction error")
        st.dataframe(table, hide_index=True, width="stretch")
    a, b = st.columns(2)
    with a:
        csv_download(table, "test_predictions.csv", "Download predictions")
    with b:
        csv_download(r["metrics"], "evaluation_metrics.csv", "Download metrics", index=True)
    st.subheader("Interpretation")
    st.write(interpretation(r))

stage_footer("Evaluate")
