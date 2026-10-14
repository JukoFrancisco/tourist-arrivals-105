import streamlit as st

from backend.pipeline import prepare_data
from frontend.ui import stage_footer, setup, require, invalidate, csv_download

setup("Prepare", "Fit scalers on fitting months, then create ordered sequences.")
require("selection", "Features")

lookback = st.select_slider("History per prediction (months)", options=[6, 12, 18, 24], value=12)
st.caption("Split settings are chosen on Features, before any feature selection.")
if st.button("Prepare sequences", type="primary"):
    try:
        p = prepare_data(st.session_state.clean_df, st.session_state.selection, lookback)
        invalidate("prepared")
        st.session_state.prepared = p
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if "prepared" in st.session_state:
    p = st.session_state.prepared
    st.dataframe(p["summary"], hide_index=True, width="stretch")
    a, b, c = st.columns(3)
    for col, key, label in [(a, "fit", "Fitting windows"), (b, "val", "Validation windows"), (c, "test", "Test windows")]:
        col.metric(label, len(p["sequences"][key]["X"]))
        col.caption(str(p["sequences"][key]["X"].shape))
    st.caption(f"Saved lookback: {p['lookback']} months. Each window predicts the next month's arrivals. "
               "Partitions are windowed separately, so each loses its first lookback months as targets. "
               "Previously observed test inputs are available to later one-month-ahead forecasts.")
    with st.expander("Scaling parameters"):
        import pandas as pd
        st.dataframe(pd.DataFrame({"feature": p["features"],
                                   "fit_min": p["scaler_X"].data_min_,
                                   "fit_max": p["scaler_X"].data_max_}), hide_index=True)
        st.caption("Validation/test values may fall outside [0, 1]. They are not clipped.")
    csv_download(p["summary"], "split_summary.csv")

stage_footer("Prepare")
