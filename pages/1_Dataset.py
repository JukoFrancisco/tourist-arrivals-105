import pandas as pd
import streamlit as st

from backend.pipeline import ROOT, DATA_URL, read_csv
from frontend.ui import stage_footer, setup, invalidate, time_chart

setup("Dataset", "Load the lab CSV and check monthly coverage.")

uploaded = st.file_uploader("CSV file", type=["csv"])
st.caption("Leave empty to use the included lab dataset.")
if st.button("Load dataset", type="primary"):
    try:
        source = uploaded.getvalue() if uploaded is not None else ROOT / "data" / "tourist_arrivals.csv"
        df, meta = read_csv(source)
        invalidate("raw_df")
        st.session_state.raw_df = df
        st.session_state.source_meta = meta
        st.session_state.source_name = uploaded.name if uploaded is not None else "tourist_arrivals.csv"
        st.rerun()
    except Exception as exc:
        st.error(str(exc))

if "raw_df" in st.session_state:
    df = st.session_state.raw_df
    a, b, c = st.columns(3)
    a.metric("Rows", f"{len(df):,}")
    b.metric("Columns", len(df.columns))
    c.metric("Coverage", f"{df.date.min():%Y}–{df.date.max():%Y}")
    expected = pd.date_range(df.date.min(), df.date.max(), freq="MS")
    gaps = expected.difference(df.date)
    st.caption(f"{len(gaps)} absent months · {df.date.duplicated().sum()} duplicate dates · "
               f"{int(df.isna().sum().sum())} missing cells")
    if len(gaps):
        st.warning("Absent months: " + ", ".join(x.strftime("%b %Y") for x in gaps))
    meta = st.session_state.source_meta
    if meta["header_lines_skipped"]:
        st.caption(f"Skipped {meta['header_lines_skipped']} title rows above the CSV header.")
    if meta["invalid_numeric"]:
        st.warning(f"Unreadable numeric entries were marked missing: {meta['invalid_numeric']}")
    # Deduplicated view only; the raw data remains intact until Clean.
    chart_df = df.drop_duplicates("date", keep="first").rename(columns={"arrivals": "Arrivals"})
    time_chart(chart_df, ["Arrivals"])
    preview, types = st.tabs(["Preview", "Column types"])
    with preview:
        columns = st.multiselect("Preview columns", df.columns.tolist(), default=df.columns.tolist())
        if columns:
            st.dataframe(df[columns], hide_index=True, width="stretch")
    with types:
        st.dataframe(df.dtypes.astype(str).rename("Type"), width="stretch")
else:
    st.info("Load the included CSV to begin.")
st.link_button("Dataset source", DATA_URL)

stage_footer("Dataset")
