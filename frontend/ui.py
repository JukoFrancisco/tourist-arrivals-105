import altair as alt
import pandas as pd
import streamlit as st

ACCENT = "#32a1d1"
STAGES = ["raw_df", "clean_df", "selection", "prepared", "model", "evaluation", "explanation", "forecast_report"]
DEPENDENCIES = {
    "raw_df": ["clean_df", "clean_report", "selection", "prepared", "model", "train_report", "evaluation", "explanation", "forecast_report"],
    "clean_df": ["selection", "prepared", "model", "train_report", "evaluation", "explanation", "forecast_report"],
    "selection": ["prepared", "model", "train_report", "evaluation", "explanation", "forecast_report"],
    "prepared": ["model", "train_report", "evaluation", "explanation", "forecast_report"],
    "model": ["evaluation", "explanation", "forecast_report"],
}


# Native SF fonts are used on macOS; other platforms use their system sans font.
from base64 import b64encode
from html import escape
from pathlib import Path

ASSETS = Path(__file__).resolve().parent / "assets"
PAGES = [
    ("Dataset", "raw_df", "pages/1_Dataset.py", "database", "Load and inspect monthly arrivals and climate readings."),
    ("Clean", "clean_df", "pages/2_Clean.py", "cleaning_services", "Resolve duplicate months and review data quality."),
    ("Features", "selection", "pages/3_Features.py", "filter_alt", "Choose useful predictors using fitting data only."),
    ("Prepare", "prepared", "pages/4_Prepare.py", "view_timeline", "Scale inputs and create chronological history windows."),
    ("Train", "model", "pages/5_Train.py", "neurology", "Compare eight LSTM configurations using validation loss."),
    ("Evaluate", "evaluation", "pages/6_Evaluate.py", "monitoring", "Compare held-out forecasts with two simple baselines."),
    ("Explain", "explanation", "pages/7_Explain.py", "lightbulb", "Explore which historical readings influence the model."),
    ("Forecast", "forecast_report", "pages/8_Forecast.py", "trending_up", "Predict next month or explore a hypothetical scenario."),
]
GUIDES = {
    "Dataset": ("Use the included CSV or upload a file with the same lab columns, then select Load dataset.", "Inspect the arrivals chart, duplicate dates, missing cells and date coverage. Loading another file clears dependent results."),
    "Clean": ("Select Run cleaning, then review each quality tab.", "The first row for each month is kept. Absent months are inserted, impossible readings become missing and IQR outliers are retained. Filling predictors happens later using past values and fitting-only medians."),
    "Features": ("Keep the default chronological split for your first run, then select Select features.", "Spearman measures association with arrivals; VIF checks redundant predictors. Both use only fitting months. Changing these settings clears preparation and later results."),
    "Prepare": ("Start with 12 months of history and select Prepare sequences.", "Each sequence contains historical predictors and targets the following month's arrivals. Scalers learn only from fitting months. The table shows which dates are actually predicted."),
    "Train": ("Select Train and tune and wait for all eight trials to finish.", "An epoch is one pass over fitting windows. Early stopping ends a trial when validation loss stops improving; the best weights are kept. Test scores never choose the model."),
    "Evaluate": ("Select Evaluate test set to compare three methods on the same months.", "MAE is average absolute error; RMSE penalizes larger misses. Lower is better for both. Naive uses last month's arrivals; seasonal naive uses the same month last year. A baseline may outperform the LSTM."),
    "Explain": ("Compute explanations, choose a forecast for the waterfall and explore the dependence plot.", "SHAP divides a prediction into a background value and signed contributions. Blue contributions increase the prediction; gray contributions reduce it. These describe model behavior, not causal effects."),
    "Forecast": ("Edit the historical input table if needed, then select Forecast next month.", "The app uses the latest consecutive months and the saved model. A scenario comparison shows how edits change its prediction. Historical calendar fields stay fixed. This is a point forecast, with no estimated uncertainty interval."),
}


def logo_uri():
    return "data:image/svg+xml;base64," + b64encode((ASSETS / "logo.svg").read_bytes()).decode()


def next_stage():
    return next((page for page in PAGES if page[1] not in st.session_state), None)


def setup(title, subtitle=""):
    st.set_page_config(page_title=f"{title} · Arrivals", page_icon=str(ASSETS / "logo.svg"), layout="wide")
    st.markdown("<style>" + (ASSETS / "style.css").read_text() + "</style>", unsafe_allow_html=True)
    with st.sidebar:
        st.markdown(f'<div class="brand"><img src="{logo_uri()}" alt="Arrivals logo"><div><strong>Arrivals</strong><small>Philippine tourism forecast</small></div></div>', unsafe_allow_html=True)
        st.page_link("Home.py", label="Overview", icon=":material/dashboard:")
        st.caption("WORKSPACE")
        for i, (name, key, path, icon, desc) in enumerate(PAGES, 1):
            st.page_link(path, label=f"{i:02}  {name}", icon=f":material/{icon}:")
        st.divider()
        completed = sum(k in st.session_state for k in STAGES)
        st.caption("YOUR PROGRESS")
        st.progress(completed / len(STAGES))
        st.caption(f"{completed} of {len(STAGES)} stages ready")
        upcoming = next_stage()
        if upcoming:
            st.caption(f"Next: {upcoming[0]}")
        else:
            st.caption("All stages ready. Explore your results.")
        if "prepared" in st.session_state:
            p = st.session_state.prepared
            st.caption(f"{len(p['features'])} predictors · {p['lookback']}-month history")
    if title != "Tourist arrivals":
        stage = next((i for i, page in enumerate(PAGES, 1) if page[0] == title), 0)
        st.markdown(f'<div class="eyebrow">Step {stage:02} / 08</div><div class="page-heading"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p></div>', unsafe_allow_html=True)
        with st.expander("How to use this page", icon=":material/help_outline:"):
            for line in GUIDES.get(title, ()):
                st.write(line)


def require(key, page):
    if key not in st.session_state:
        st.info(f"Complete {page} to unlock this step.", icon=":material/info:")
        target = next((p for p in PAGES if p[0] == page), None)
        if target:
            st.page_link(target[2], label=f"Go to {page}", icon=f":material/{target[3]}:")
        st.stop()


def stage_footer(title):
    i = next((i for i, p in enumerate(PAGES) if p[0] == title), None)
    if i is not None and PAGES[i][1] in st.session_state and i < len(PAGES)-1:
        if title == "Features" and not st.session_state.selection["selected"]:
            return
        following = PAGES[i+1]
        st.divider()
        st.page_link(following[2], label=f"Continue to {following[0]}", icon=":material/arrow_forward:")


def invalidate(stage):
    st.session_state.forecast_editor_revision = st.session_state.get("forecast_editor_revision", 0) + 1
    for editor_key in list(st.session_state):
        if editor_key.startswith("forecast_editor_") and editor_key != "forecast_editor_revision":
            st.session_state.pop(editor_key, None)
    for key in DEPENDENCIES.get(stage, []):
        st.session_state.pop(key, None)


def csv_download(df, name, label="Download CSV", index=False):
    st.download_button(label, df.to_csv(index=index).encode(), name, "text/csv", icon=":material/download:")


def time_chart(df, columns, y_title="Arrivals"):
    long = df[["date", *columns]].melt("date", var_name="Series", value_name="value")
    palette = [ACCENT, "#64748b", "#b7c1cb", "#0f5273"]
    chart = alt.Chart(long).mark_line(strokeWidth=2.5, point=True).encode(
        x=alt.X("date:T", title=None), y=alt.Y("value:Q", title=y_title),
        color=alt.Color("Series:N", scale=alt.Scale(domain=columns, range=palette[:len(columns)]),
                        legend=alt.Legend(orient="bottom")),
        tooltip=[alt.Tooltip("date:T", title="Month", format="%b %Y"), "Series:N",
                 alt.Tooltip("value:Q", title=y_title, format=",.1f")],
    ).properties(height=310).interactive()
    st.altair_chart(chart, width="stretch")


def save_current_run():
    from backend.storage import save_run
    try:
        save_run(st.session_state)
    except Exception as exc:
        st.warning(f"Results are available in this session, but local saving failed: {exc}")


from backend.reporting import interpretation
