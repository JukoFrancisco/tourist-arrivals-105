from html import escape

import streamlit as st

from backend.storage import has_run, restore_run
from frontend.ui import PAGES, interpretation, logo_uri, next_stage, setup
from backend.reporting import export_report

setup("Tourist arrivals")
st.markdown(f'''<div class="hero"><img class="hero-mark" src="{logo_uri()}" alt=""><div class="eyebrow">Tourism · Time series · Explainable AI</div><h1>Understand the past.<br>Forecast the next month.</h1><p>Explore Philippine tourist arrivals alongside seasonal, climate and hazard readings. Train an LSTM, compare its forecasts and understand what influences its predictions.</p></div>''', unsafe_allow_html=True)

upcoming = next_stage()
a, b, c = st.columns(3)
a.metric("Forecast horizon", "1 month")
b.metric("History per forecast", f"{st.session_state.get('prepared', {}).get('lookback', 12)} months")
c.metric("Workflow progress", f"{sum(p[1] in st.session_state for p in PAGES)} / 8")

st.subheader("Start your workspace")
left, right = st.columns([1.2, 1])
with left:
    with st.container(border=True):
        st.write("**Build your own run**")
        st.caption("Follow the eight steps below. Start with the default settings, then inspect the results.")
        if upcoming:
            st.page_link(upcoming[2], label=f"{'Start with' if upcoming[0] == 'Dataset' else 'Continue to'} {upcoming[0]}", icon=":material/arrow_forward:")
        else:
            st.page_link("pages/8_Forecast.py", label="Explore forecasts", icon=":material/trending_up:")
with right:
    with st.container(border=True):
        st.write("**Preview the completed example**")
        st.caption("Restore the bundled model, evaluation and SHAP results. This replaces the current session.")
        if has_run():
            if st.button("Restore saved run", type="primary", icon=":material/restore:", width="stretch"):
                try:
                    with st.spinner("Restoring model and results…"):
                        state = restore_run()
                    st.session_state.clear()
                    st.session_state.update(state)
                    st.rerun()
                except Exception as exc:
                    st.error(f"Could not restore this run: {exc}. Start from Dataset instead.")
        else:
            st.caption("Train a model to create a saved run.")

st.subheader("Your workflow")
for row in range(4):
    columns = st.columns(2)
    for col, i in zip(columns, range(row*2, row*2+2)):
        name, key, path, icon, description = PAGES[i]
        status = "Ready" if key in st.session_state else ("Next step" if upcoming and upcoming[0] == name else "Pending")
        with col, st.container(border=True):
            st.markdown(f'<div class="step-card"><div class="step-number">{i+1:02} / {status}</div><strong>{escape(name)}</strong><p>{escape(description)}</p></div>', unsafe_allow_html=True)
            st.page_link(path, label=f"Open {name}", icon=f":material/{icon}:")

if "evaluation" in st.session_state:
    st.subheader("Your latest results")
    result = st.session_state.evaluation
    best = result["metrics"].MAE.idxmin()
    a, b, c = st.columns(3)
    a.metric("Lowest MAE", best)
    b.metric("LSTM MAE", f"{result['metrics'].loc['LSTM', 'MAE']:,.0f}", help="Average absolute error, measured in arrivals.")
    c.metric("Evaluated months", len(result["predictions"]))
    st.write(interpretation(result, st.session_state.get("explanation")))
    st.download_button("Download run report", export_report(st.session_state), "arrivals-run-report.zip", "application/zip", icon=":material/download:")

with st.expander("How the app works", icon=":material/help_outline:"):
    st.write("**1. Explore and prepare.** Load the CSV, review quality issues, choose predictors and build monthly history windows.")
    st.write("**2. Train and compare.** An LSTM learns from ordered history. Validation chooses its settings; held-out test months compare it with last-month and last-year baselines.")
    st.write("**3. Explain and forecast.** SHAP shows how model inputs contribute to predictions. The forecast page uses the latest history and lets you test input scenarios.")
    st.caption("Seasonal and climate associations are not proof of cause and effect. A functioning app does not guarantee accurate forecasts. Use the evaluation results to judge the model.")
with st.expander("Session and saved results", icon=":material/save:"):
    st.write("Page results stay in your current browser session. Training, evaluation and explanations also update a local saved run, which you can restore after restarting. Changing an earlier stage clears its dependent results.")
    st.caption("Start fresh clears the browser workflow; it keeps the saved run and the included dataset.")
    if st.button("Start fresh", icon=":material/restart_alt:"):
        st.session_state.clear()
        st.rerun()
