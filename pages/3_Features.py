import streamlit as st

from backend.pipeline import select_features
from frontend.ui import stage_footer, setup, require, invalidate, csv_download

setup("Features", "Spearman screening, followed by iterative VIF on fitting data only.")
require("clean_df", "Clean")

with st.form("feature_settings"):
    a, b = st.columns(2)
    development = a.slider("Development share", .60, .90, .80, .05,
                           help="The remaining most recent months form the test period.")
    validation = b.slider("Validation share of development", .15, .30, .15, .05)
    categories = st.checkbox("Include season and monsoon indicators", value=False)
    st.caption("Keep |ρ| > 0.10 and p < 0.05; remove the largest VIF until all are below 5.")
    submitted = st.form_submit_button("Select features", type="primary")

if submitted:
    try:
        result = select_features(st.session_state.clean_df, development, validation, categories)
        invalidate("selection")
        st.session_state.selection = result
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if "selection" in st.session_state:
    s = st.session_state.selection
    df = st.session_state.clean_df
    st.caption(f"Selection period: {df.date.iloc[0]:%b %Y} – {df.date.iloc[s['val_start']-1]:%b %Y}")
    if not s["selected"]:
        st.error("No predictors passed. Check the dataset; selection thresholds have not been relaxed.")
    else:
        st.success("Selected: " + ", ".join(s["selected"]))
    spearman, vif = st.tabs(["Spearman", "VIF log"])
    with spearman:
        st.dataframe(s["spearman"], hide_index=True, width="stretch")
        csv_download(s["spearman"], "spearman.csv")
    with vif:
        st.dataframe(s["vif"], hide_index=True, width="stretch")
        if len(s["selected"]) == 1:
            st.caption("One predictor remains; there is no between-predictor multicollinearity to test.")
        csv_download(s["vif"], "vif.csv")
    st.caption("Numeric-only mode follows the handout's candidate list. Optional indicators use "
               "fixed category levels with Dry and Amihan as reference categories.")

stage_footer("Features")
