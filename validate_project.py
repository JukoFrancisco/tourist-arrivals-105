"""Optional end-to-end verification. Produces a real, restorable run on the supplied CSV."""
import argparse
import json

from backend.explainability import explain_model
from backend.modeling import predict_arrivals, train_search
from backend.pipeline import ROOT, clean_data, evaluate_model, prepare_data, read_csv, select_features
from backend.storage import save_run


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--shap-count", type=int, default=10)
    args = parser.parse_args()
    raw, meta = read_csv(ROOT / "data" / "tourist_arrivals.csv")
    clean, report = clean_data(raw)
    selection = select_features(clean)
    prepared = prepare_data(clean, selection)
    model, train_report = train_search(prepared, epochs=args.epochs,
                                       progress=lambda v, t: print(t, flush=True))
    evaluation = evaluate_model(model, prepared, clean, predict_arrivals)
    explanation = explain_model(model, prepared, count=args.shap_count)
    state = dict(raw_df=raw, source_meta=meta, source_name="tourist_arrivals.csv",
                 clean_df=clean, clean_report=report, selection=selection, prepared=prepared,
                 model=model, train_report=train_report, evaluation=evaluation, explanation=explanation)
    save_run(state)
    print(evaluation["metrics"].to_string())
    print("SHAP maximum additivity error:", explanation["max_additivity_error"])
    print("Saved run ready to restore from Home.")


if __name__ == "__main__":
    main()
