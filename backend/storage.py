"""Save local app-generated runs; never deserialize user-uploaded model bundles."""
import json
import platform
from pathlib import Path

import joblib

from backend.pipeline import ROOT

RUN = ROOT / "artifacts" / "saved_run"
STATE_KEYS = ["raw_df", "source_meta", "source_name", "clean_df", "clean_report",
              "selection", "prepared", "train_report", "evaluation", "explanation"]


def save_run(state):
    if "model" not in state:
        return
    RUN.mkdir(parents=True, exist_ok=True)
    state["model"].save(RUN / "model.keras")
    joblib.dump({k: state[k] for k in STATE_KEYS if k in state}, RUN / "state.joblib")
    import importlib.metadata
    versions = {}
    for name in ["streamlit", "numpy", "pandas", "scikit-learn", "keras", "shap", "tensorflow", "tensorflow-cpu"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
    (RUN / "versions.json").write_text(json.dumps({"python": platform.python_version(), **versions}, indent=2))


def restore_run():
    from backend.modeling import tensorflow
    state = joblib.load(RUN / "state.joblib")
    state["model"] = tensorflow().keras.models.load_model(RUN / "model.keras", compile=False)
    return state


def has_run():
    return (RUN / "model.keras").exists() and (RUN / "state.joblib").exists()
