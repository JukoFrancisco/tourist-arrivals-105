import pandas as pd
import streamlit as st

from backend.modeling import train_search
from frontend.ui import stage_footer, setup, require, invalidate, csv_download, save_current_run

setup("Train", "Tune a small LSTM using validation loss; the test set stays untouched.")
require("prepared", "Prepare")

with st.form("train_settings"):
    a, b = st.columns(2)
    epochs = a.number_input("Maximum epochs per trial", 10, 200, 80, 10)
    patience = b.number_input("Early-stopping patience", 3, 20, 8)
    st.caption("8 trials · units 32/64 · dropout 0.1/0.3 · batch size 16/32 · seed 42")
    submitted = st.form_submit_button("Train and tune", type="primary")

if submitted:
    bar = st.progress(0, text="Starting training…")
    try:
        with st.spinner("Training models…"):
            model, report = train_search(st.session_state.prepared, epochs=int(epochs),
                                         patience=int(patience),
                                         progress=lambda value, text: bar.progress(value, text=text))
        invalidate("model")
        st.session_state.model = model
        st.session_state.train_report = report
        save_current_run()
        st.rerun()
    except Exception as exc:
        st.error(f"Training failed: {exc}")

if "train_report" in st.session_state:
    r = st.session_state.train_report
    a, b, c = st.columns(3)
    a.metric("LSTM units", r["params"]["units"])
    b.metric("Dropout", r["params"]["dropout"])
    c.metric("Batch size", r["params"]["batch_size"])
    st.caption(f"Best validation MSE: {r['val_loss']:.6f} (scaled target)")
    hist = pd.DataFrame({"Training": r["history"]["loss"], "Validation": r["history"]["val_loss"]})
    hist.index = pd.RangeIndex(1, len(hist)+1, name="Epoch")
    st.line_chart(hist, color=["#32a1d1", "#64748b"])
    with st.expander("All trials"):
        st.dataframe(r["trials"].sort_values("best_val_loss"), hide_index=True, width="stretch")
        csv_download(r["trials"], "training_trials.csv")
    st.caption("The best early-stopped weights are retained. A local copy can be restored from Home.")

stage_footer("Train")
