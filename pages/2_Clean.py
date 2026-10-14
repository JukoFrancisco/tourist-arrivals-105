import pandas as pd
import streamlit as st

from backend.pipeline import clean_data
from frontend.ui import stage_footer, setup, require, invalidate, csv_download

setup("Clean", "Resolve duplicate dates and inspect missing values and outliers.")
require("raw_df", "Dataset")

if st.button("Run cleaning", type="primary"):
    try:
        df, report = clean_data(st.session_state.raw_df)
        invalidate("clean_df")
        st.session_state.clean_df = df
        st.session_state.clean_report = report
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if "clean_report" in st.session_state:
    r = st.session_state.clean_report
    a, b, c = st.columns(3)
    a.metric("Duplicates removed", r["duplicates_removed"])
    b.metric("Missing months inserted", len(r["missing_months"]))
    c.metric("Outliers flagged", len(r["flagged"]))
    duplicates, missing, outliers = st.tabs(["Duplicates", "Missing values", "Outliers"])
    with duplicates:
        st.caption("Keep the first occurrence in source order. Review conflicting values below.")
        st.dataframe(r["duplicates"], hide_index=True, width="stretch")
    with missing:
        st.dataframe(r["missing"][r["missing"] > 0], width="stretch")
        st.caption("Predictors: forward-fill up to two months, then fitting-period median. "
                   "Missing arrivals stay unknown and are excluded as targets.")
        if r["missing_months"]:
            st.write([str(x.date()) for x in r["missing_months"]])
        if r["invalid"]:
            st.write("Invalid physical readings marked missing:", r["invalid"])
        if r["calendar_fixes"]:
            st.write("Calendar fields derived from date:", r["calendar_fixes"])
    with outliers:
        lo, hi = r["iqr_bounds"]
        st.caption(f"IQR fences: {lo:,.0f} to {hi:,.0f} arrivals. Flagged observations are retained.")
        st.dataframe(r["flagged"], hide_index=True, width="stretch")
        st.caption("These full-series flags are descriptive only; they do not fit or modify the model.")
    csv_download(st.session_state.clean_df, "cleaned_arrivals.csv")

stage_footer("Clean")
