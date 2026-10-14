"""Kernel SHAP on the real LSTM, returning contributions in arrivals."""
import numpy as np
import pandas as pd

from backend.modeling import predict_arrivals


def explain_model(model, prep, count=10, nsamples=256, background_count=30):
    import shap
    rng = np.random.default_rng(42)
    fit = prep["sequences"]["fit"]["X"]
    test = prep["sequences"]["test"]
    n = min(count, len(test["X"]))
    # Spread explanations across the test period instead of implying its first months represent all of it.
    indices = np.linspace(0, len(test["X"]) - 1, n, dtype=int)
    sample = test["X"][indices]
    lookback, features = sample.shape[1:]
    bg_n = min(background_count, len(fit))
    background = fit[rng.choice(len(fit), bg_n, replace=False)].reshape(bg_n, -1)

    def predict_flat(flat):
        return predict_arrivals(model, np.asarray(flat).reshape(-1, lookback, features), prep["scaler_y"])

    np.random.seed(42)
    explainer = shap.KernelExplainer(predict_flat, background)
    raw = explainer.shap_values(sample.reshape(n, -1), nsamples=nsamples,
                                l1_reg="num_features(10)", silent=True)
    values = np.asarray(raw).reshape(n, lookback, features)
    base = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    predictions = predict_flat(sample.reshape(n, -1))
    residual = predictions - (base + values.sum(axis=(1, 2)))
    # Compare reconstructed outputs, not residuals to zero: relative tolerance
    # must be relative to the forecast magnitude. Float32 LSTM kernels can
    # differ slightly between single-sample and batched calls.
    if not np.allclose(base + values.sum(axis=(1, 2)), predictions, rtol=1e-5, atol=1.):
        raise ValueError(f"SHAP additivity check failed (max error {np.abs(residual).max():.3f} arrivals).")
    importance = pd.DataFrame({"feature": prep["features"],
                               "importance": np.abs(values).mean(axis=(0, 1))})
    importance = importance.sort_values("importance", ascending=False).reset_index(drop=True)
    original = prep["scaler_X"].inverse_transform(sample[:, -1, :])
    return {"values": values, "base": base, "predictions": predictions,
            "dates": test["dates"][indices], "importance": importance,
            "latest_inputs": original, "sample_count": n,
            "background_count": bg_n, "nsamples": nsamples,
            "max_additivity_error": float(np.abs(residual).max())}


def waterfall_table(base, contributions, names, prediction):
    rows = [{"term": "Base value", "start": 0., "end": base,
             "contribution": base, "kind": "Total"}]
    current = base
    for i in np.argsort(-np.abs(contributions)):
        value = float(contributions[i])
        rows.append({"term": names[i], "start": current, "end": current + value,
                     "contribution": value, "kind": "Increase" if value >= 0 else "Decrease"})
        current += value
    rows.append({"term": "Prediction", "start": 0., "end": float(prediction),
                 "contribution": float(prediction), "kind": "Total"})
    return pd.DataFrame(rows)
