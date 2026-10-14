import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from backend.explainability import explain_model, waterfall_table
from backend.pipeline import LABELS
from frontend.ui import stage_footer, setup, require, csv_download, save_current_run, ACCENT, interpretation

setup("Explain", "Kernel SHAP contributions, expressed in arrivals.")
require("model", "Train")

with st.form("explain_settings"):
    a, b = st.columns(2)
    maximum = min(20, len(st.session_state.prepared["sequences"]["test"]["X"]))
    count = a.number_input("Forecasts to explain", 1, maximum, min(10, maximum))
    nsamples = b.select_slider("SHAP sampling budget", options=[128, 256, 512, 1024], value=256)
    st.caption("Uses up to 30 fitting windows as background. Larger budgets take longer.")
    submitted = st.form_submit_button("Compute explanations", type="primary")

if submitted:
    try:
        with st.spinner("Explaining test forecasts…"):
            st.session_state.explanation = explain_model(
                st.session_state.model, st.session_state.prepared,
                int(count), int(nsamples))
        save_current_run()
        st.rerun()
    except Exception as exc:
        st.error(f"SHAP computation failed: {exc}")

if "explanation" in st.session_state:
    r = st.session_state.explanation
    p = st.session_state.prepared
    names = p["features"]
    st.caption(f"{r['sample_count']} test forecasts · {r['background_count']} fitting background windows "
               f"· {r['nsamples']} samples per explanation")
    st.subheader("Global importance")
    imp = r["importance"].copy()
    imp["Predictor"] = imp.feature.map(lambda x: LABELS.get(x, x))
    chart = alt.Chart(imp).mark_bar(color=ACCENT, cornerRadiusEnd=3).encode(
        x=alt.X("importance:Q", title="Mean |SHAP| per time step (arrivals)"),
        y=alt.Y("Predictor:N", sort="-x", title=None),
        tooltip=["Predictor", alt.Tooltip("importance:Q", format=",.1f")],
    ).properties(height=max(180, 32 * len(imp)))
    st.altair_chart(chart, width="stretch")
    st.caption("Average absolute influence across the explained sample and all historical steps; "
               "this is a sample summary, not an explanation of every forecast.")

    st.subheader("One forecast")
    choice = st.selectbox("Forecast month", range(r["sample_count"]),
                          format_func=lambda i: pd.Timestamp(r["dates"][i]).strftime("%B %Y"))
    a, b = st.columns(2)
    a.metric("Base prediction", f"{r['base']:,.0f}")
    b.metric("Model prediction", f"{r['predictions'][choice]:,.0f}")
    contributions = r["values"][choice].sum(axis=0)
    labels = [LABELS.get(x, x) for x in names]
    table = waterfall_table(r["base"], contributions, labels, r["predictions"][choice])
    order = table.term.tolist()
    chart = alt.Chart(table).mark_bar(size=20).encode(
        x=alt.X("start:Q", title="Predicted arrivals", scale=alt.Scale(zero=True)),
        x2="end:Q", y=alt.Y("term:N", sort=order, title=None),
        color=alt.Color("kind:N", scale=alt.Scale(domain=["Total", "Increase", "Decrease"],
                          range=["#64748b", ACCENT, "#b5bdc7"]), legend=alt.Legend(orient="bottom")),
        tooltip=["term:N", alt.Tooltip("contribution:Q", format="+,.1f")],
    ).properties(height=max(240, 32 * len(table)))
    st.altair_chart(chart, width="stretch")
    st.caption("Each predictor sums its contributions over the lookback window. "
               f"Maximum additivity error: {r['max_additivity_error']:.4f} arrivals.")

    st.subheader("Dependence")
    a, b = st.columns(2)
    top = a.selectbox("Dependence predictor", r["importance"].feature.tolist(),
                      format_func=lambda x: LABELS.get(x, x))
    lag = b.select_slider("Historical lag (months before forecast)", options=list(range(1, p["lookback"]+1)), value=1)
    idx = names.index(top)
    # Explanations are spread across the test windows; reconstruct those exact inputs.
    test_X = p["sequences"]["test"]["X"]
    indices = np.linspace(0, len(test_X)-1, r["sample_count"], dtype=int)
    original = p["scaler_X"].inverse_transform(test_X[indices, -lag, :])
    dependence = pd.DataFrame({"value": original[:, idx],
                               "SHAP": r["values"][:, -lag, idx],
                               "forecast_month": pd.to_datetime(r["dates"])})
    chart = alt.Chart(dependence).mark_circle(size=70, color=ACCENT).encode(
        x=alt.X("value:Q", title=f"{LABELS.get(top, top)} · t−{lag}", scale=alt.Scale(zero=False)),
        y=alt.Y("SHAP:Q", title=f"Contribution at t−{lag} (arrivals)"),
        tooltip=[alt.Tooltip("forecast_month:T", format="%b %Y"), "value:Q", "SHAP:Q"],
    ).properties(height=280)
    st.altair_chart(chart, width="stretch")
    st.caption(f"The plot pairs a value and contribution from the same lag (t−{lag}). "
               "SHAP is approximate and does not establish causation.")
    csv_download(r["importance"], "shap_importance.csv")
    with st.expander("Explanation details"):
        st.dataframe(table, hide_index=True, width="stretch")
        csv_download(table, "forecast_contributions.csv", "Download contributions")
        st.caption("Kernel SHAP treats lag-feature cells as inputs. Their temporal and cross-feature "
                   "correlations can make attribution unstable; interpret small samples cautiously.")
    if "evaluation" in st.session_state:
        st.subheader("Interpretation")
        st.write(interpretation(st.session_state.evaluation, r))

stage_footer("Explain")
