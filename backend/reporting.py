"""Portable, human-readable run exports; no deserialization is needed."""
import io
import json
import zipfile


def export_report(state):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        if "evaluation" in state:
            evaluation = state["evaluation"]
            archive.writestr("metrics.csv", evaluation["metrics"].to_csv(index=True))
            archive.writestr("predictions.csv", evaluation["predictions"].to_csv(index=False))
            archive.writestr("interpretation.md", interpretation(evaluation, state.get("explanation")))
        for key, items in {
            "selection": [("spearman", "spearman.csv"), ("vif", "vif.csv")],
            "prepared": [("summary", "split_summary.csv")],
            "train_report": [("trials", "training_trials.csv")],
            "explanation": [("importance", "shap_importance.csv")],
        }.items():
            if key in state:
                for column, filename in items:
                    archive.writestr(filename, state[key][column].to_csv(index=False))
        details = {"source": state.get("source_name"), "features": state.get("prepared", {}).get("features"),
                   "lookback": state.get("prepared", {}).get("lookback"),
                   "best_parameters": state.get("train_report", {}).get("params"),
                   "seed": state.get("train_report", {}).get("seed")}
        if "explanation" in state:
            details["shap"] = {k: state["explanation"][k] for k in ["sample_count", "background_count", "nsamples", "max_additivity_error"]}
        archive.writestr("run_details.json", json.dumps(details, indent=2))
        archive.writestr("README.txt", "Results from the current Arrivals workflow. Metrics use original arrivals units and identical held-out dates. SHAP describes model reliance, not causation. This report does not contain the trained model or raw dataset.")
    return output.getvalue()


def interpretation(evaluation, explanation=None):
    m = evaluation["metrics"]
    best = m["MAE"].idxmin()
    lines = [f"The lowest test MAE belongs to {best} ({m.loc[best, 'MAE']:,.0f} arrivals).",
             f"The LSTM's MAE is {m.loc['LSTM', 'MAE']:,.0f}, compared with "
             f"{m.loc['Seasonal naive', 'MAE']:,.0f} for seasonal naive."]
    if explanation is not None:
        top = explanation["importance"].iloc[0]["feature"]
        lines.append(f"{top} has the largest average absolute contribution in the explained test sample. "
                     "This describes the model's reliance on the predictor, not a causal effect.")
    lines.append("Scores describe rolling one-month-ahead predictions on the displayed held-out dates. "
                 "Unmeasured events, structural changes and a limited monthly sample can reduce reliability.")
    return "\n\n".join(lines)
