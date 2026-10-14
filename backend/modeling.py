"""Small LSTM search; TensorFlow is imported only when needed."""
from __future__ import annotations

import itertools
import os

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")


def tensorflow():
    import tensorflow as tf
    return tf


def build_model(lookback, features, units, dropout):
    tf = tensorflow()
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(lookback, features)),
        tf.keras.layers.LSTM(units),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model


def _dataset(X, y, batch_size):
    tf = tensorflow()
    options = tf.data.Options()
    options.threading.private_threadpool_size = 1
    options.threading.max_intra_op_parallelism = 1
    return tf.data.Dataset.from_tensor_slices((X, y)).batch(batch_size).with_options(options)


def train_search(prep, units=(32, 64), dropouts=(.1, .3), batches=(16, 32),
                 epochs=80, patience=8, progress=None):
    tf = tensorflow()
    fit, val = prep["sequences"]["fit"], prep["sequences"]["val"]
    combinations = list(itertools.product(units, dropouts, batches))
    best_loss, best_weights, best_params, best_history = float("inf"), None, None, None
    trials = []
    for i, (u, d, b) in enumerate(combinations):
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(42)
        model = build_model(prep["lookback"], len(prep["features"]), u, d)
        if progress:
            progress(i / len(combinations), f"Trial {i+1}/{len(combinations)} · {u} units · dropout {d} · batch {b}")
        class EpochProgress(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                if progress:
                    logs = logs or {}
                    progress((i + (epoch + 1) / epochs) / len(combinations),
                             f"Trial {i+1}/{len(combinations)} · epoch {epoch+1}/{epochs} · "
                             f"validation loss {logs.get('val_loss', float('nan')):.6f}")

        history = model.fit(
            _dataset(fit["X"], fit["y"], b),
            validation_data=_dataset(val["X"], val["y"], b),
            epochs=epochs, shuffle=False, verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience, restore_best_weights=True), EpochProgress()],
        ).history
        loss = float(min(history["val_loss"]))
        params = {"units": int(u), "dropout": float(d), "batch_size": int(b)}
        trials.append({**params, "best_val_loss": loss, "epochs": len(history["loss"])})
        if np.isfinite(loss) and loss < best_loss:
            best_loss, best_params, best_weights, best_history = loss, params, model.get_weights(), history
    if best_weights is None:
        raise ValueError("All trials produced invalid losses. Check inputs and scaling.")
    best_model = build_model(prep["lookback"], len(prep["features"]), best_params["units"], best_params["dropout"])
    best_model.set_weights(best_weights)
    if progress:
        progress(1.0, "Training complete")
    return best_model, {"params": best_params, "val_loss": best_loss,
                        "history": best_history, "trials": pd.DataFrame(trials),
                        "seed": 42, "max_epochs": epochs}


def predict_arrivals(model, X, scaler_y):
    # Direct inference avoids spinning up a tf.data thread pool for each SHAP call.
    X = np.asarray(X, dtype=np.float32)
    outputs = [model(X[i:i+256], training=False).numpy() for i in range(0, len(X), 256)]
    if not outputs:
        return np.array([])
    return scaler_y.inverse_transform(np.concatenate(outputs).astype(np.float64).reshape(-1, 1)).reshape(-1)
