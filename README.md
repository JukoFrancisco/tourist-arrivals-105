# Philippine Tourist Arrivals — LSTM & XAI

A complete Streamlit lab project with an Arrivals logo, white and gray interface, blue `#32a1d1` accents and an SF-style native system font stack. No separate frontend or backend is required.

## UI and workflow update

See `UPGRADE.md` to reuse an existing environment. The final layout clears the fixed toolbar and hides Deploy. Backend logic lives in `backend/`, shared interface helpers and assets in `frontend/`, and Streamlit screens in `pages/` (see `FILE_STRUCTURE.md`). This remains a single Streamlit process. The overview provides progress and next-step links; every page has a concise usage guide. Forecasting compares edited history with unchanged history, Explain offers predictor/lag controls, and Overview exports a portable run report. Launch on Mac with `bash run_app.command`.

## Quick start — Mac

Python **3.11** is recommended. Open a terminal inside the extracted `tourist-arrivals-app` folder:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run Home.py
```

Open **http://localhost:8501**. If `python3.11` is unavailable, install Python 3.11 first, or use an existing compatible Python 3.10–3.12 interpreter in the commands. On Intel Macs, requirements select TensorFlow 2.16.2, NumPy < 2, and a compatible SHAP range. Apple Silicon should use a native ARM Python rather than mixing ARM and Rosetta installations.

Windows (Python 3.11):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run Home.py
```

Linux follows the Mac commands. TensorFlow uses CPU by default; no GPU configuration is needed.

For subsequent launches, activate `.venv` and run `python -m streamlit run Home.py`.

## Use the app

The archive includes the original course CSV and a real saved run, including a trained model and computed explanations. **Restore saved run** on Home opens completed results immediately. If package/platform differences prevent restoration, start from Dataset and train locally. The source code and CSV are sufficient to reproduce the workflow without the saved run.

For your own lab run, follow the sidebar in order:

| Page | Action | Result |
| --- | --- | --- |
| Dataset | Load dataset | Shape, types, coverage, preview, arrivals chart |
| Clean | Run cleaning | Duplicate, missing-value and IQR reports |
| Features | Select features | Training-only Spearman and iterative VIF tables |
| Prepare | Prepare sequences | Partition dates, shapes, training-only scalers |
| Train | Train and tune | Eight trials, early stopping, best model, loss curves |
| Evaluate | Evaluate test set | MAE/RMSE/MAPE/R² for LSTM and two baselines |
| Explain | Compute explanations | Global importance, waterfall, dependence plot |
| Forecast | Edit history; Forecast next month | Actual model inference and downloadable point forecast |

Defaults are an 80% development / 20% test split, with the last 15% of development held out for validation, and a 12-month lookback. This means approximately 68% fit / 12% validation / 20% test. The interface calls the first portion **fitting** to distinguish it from the complete development period.

Changing an earlier stage clears all its dependent results in the current session. Buttons commit settings; settings changed in a form are applied when you submit it. Training, evaluation and explanation results are also saved locally under `artifacts/saved_run/`. The restore button deliberately loads the last saved run, which can differ from an unfinished current session.

## Dataset

Source: the CSV linked in the course handout:

https://drive.google.com/drive/folders/1b8KgyEbT1E6EkpEEKOkiOe29JnHLFbml?usp=sharing

- Original export: **314 rows, 19 columns**.
- Coverage: **January 2000–December 2025**.
- After duplicate removal: **312 monthly rows**.
- The export has two title rows before its header. The loader detects the header; the included CSV bytes are unchanged.
- Source hash and parsing details are stored with each run.
- This is the course-provided paired dataset. The project does not independently verify its measurements or attribute them to an official tourism agency.

## Cleaning decisions

1. Keep the first duplicate month in source order and display all duplicate records for review.
2. Insert absent monthly timestamps, preserving regular spacing.
3. Derive year, month and quarter from the date.
4. Convert missing sentinels (`-999`, `-9999`), unreadable numeric entries, infinities and impossible nonnegative readings to missing. Report the changes.
5. Flag arrivals outside the 1.5-IQR fences, but **retain them**. The source includes an arrivals figure above 34 million. There is no external correction log, so it is not automatically replaced or removed. These descriptive full-series flags never affect training.
6. Forward-fill predictors for at most two months using earlier observations only. Remaining predictor gaps use medians fitted on the fitting period. No backward fill or future interpolation is used.
7. Missing arrivals remain missing. Windows with unknown targets are excluded from fitting/evaluation. Missing targets are never fabricated for scoring.

## Feature selection and leakage prevention

- Select numeric candidates using `abs(Spearman rho) > 0.10` and `p < 0.05`.
- Apply iterative VIF with an intercept, removing the largest VIF until every retained predictor has VIF < 5. Constant and empty candidates are handled.
- Optional season/monsoon indicators use a predefined category vocabulary; reference categories are Dry and Amihan. The numeric-only default matches the handout's explicit list of **13** numeric candidate columns (the handout's prose says fourteen).
- Splits are defined **before** feature selection. Correlations, VIF, fallback medians and scaling parameters use the fitting portion only, excluding both validation and test months.
- MinMax scalers transform validation and test using the same fitting parameters. Out-of-range transformed values are retained.
- Each partition is windowed separately, matching the handout. Its first `lookback` months supply history but are not prediction targets. The Prepare table reports both partition dates and actual target dates.
- The LSTM sees `(windows, lookback, selected_predictors)` and predicts the next month's arrivals. Historical arrivals are not added as predictors in this MVP, matching the lab's starter input definition.
- Tuning uses validation loss and early stopping only. Test results do not select hyperparameters.
- Test evaluation is rolling **one month ahead**, with observed historical inputs becoming available over time. It is not a fixed-origin multi-month forecast.

## Model and evaluation

Architecture: Input → LSTM → Dropout → Dense(1), Adam optimizer, MSE loss.

Eight combinations: units 32/64, dropout 0.1/0.3, batch size 16/32. Maximum 80 epochs by default, early-stopping patience 8, seed 42. Ordered `tf.data` batches are not shuffled. The best trial's restored early-stopping weights are retained; no test-driven refit is performed.

Evaluation inverse-transforms predictions to arrivals. Baselines are previous-month arrivals and same-month-last-year arrivals. All three methods use a common mask and identical scored dates. Zero actuals are excluded from MAPE, and the excluded count is disclosed. R² is reported as unavailable for a constant target; a negative R² is otherwise valid. See `results/metrics.csv` and `results/INTERPRETATION.md` for the included run.

The saved run's LSTM does **not** beat the baselines. This is a reported finding, not a hidden failure. The handout explicitly accepts a baseline winning. Strong scale distortion from retained extreme source readings, unmeasured events, temporal shifts and the model's input specification are limitations; this run alone does not isolate their individual effects.

## SHAP

- Uses KernelExplainer, with a flattened interface to the real LSTM.
- Predictions and explanations are in original arrivals units.
- Background: up to 30 randomly selected fitting windows, seed 42.
- Default: 10 forecasts spread across the test period and a 256-sample budget per explanation; both are adjustable.
- Uses the explicit sparse `num_features(10)` setting for the lag-feature attribution approximation. This can leave many lag cells with zero attribution; it is not a second model feature-selection step. Increase the budget to inspect stability.
- Global importance averages absolute SHAP over explained forecasts and time steps. It describes that sample, not all predictions.
- The waterfall accumulates signed contributions over each predictor's historical lags, starting at the expected base prediction and ending at the model output. Additivity is checked with a small floating-point tolerance.
- The dependence chart pairs original-unit values at lag 1 with contributions at the same lag. It does not pair a lag-1 reading with a sum across unrelated months.
- Temporal/cross-feature correlations and the limited explanation budget constrain interpretation. SHAP describes the model's learned reliance, not causation.

## Forecast page

The page starts with the **latest available** consecutive historical window, not an old training window. It predicts the following calendar month (January 2026 for the included data). Inputs are validated, transformed in the saved feature order, reshaped, and passed through the actual trained model. Edits produce hypothetical scenarios. Values outside the fitting range produce a warning. Negative predictions are flagged as invalid rather than silently clipped. There is no fabricated confidence interval.

The editable table uses the same saved imputation rules for missing defaults. Date labels cannot be edited. If using category indicators, do not activate multiple categories for the same original field.

## Project structure

```text
Home.py                  Landing page and local restore
pages/                   Eight Streamlit workflow pages
utils/pipeline.py        Loading, cleaning, selection, preparation, metrics
utils/modeling.py        LSTM search and inference
utils/explainability.py  Kernel SHAP and waterfall data
utils/storage.py         Local saved model and preprocessing state
utils/ui.py              Shared visual style and state invalidation
data/                    Original course CSV
results/                 Real example outputs and short interpretation
artifacts/saved_run/     Included model, preprocessing state and results
tests/                   Data integrity and leakage regression tests
validate_project.py      Optional full reproduction
```

Only load `state.joblib` produced by this project or another trusted source. The application does not offer arbitrary uploaded pickle/model loading.

## Verification

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

Optional full pipeline reproduction (trains all eight models and computes SHAP):

```bash
python validate_project.py --epochs 80 --shap-count 10
```

This updates the saved run. Hardware/library differences can change numerical results even with the same seed. The included `artifacts/saved_run/versions.json` records the validated environment; Intel Mac restoration is not guaranteed across TensorFlow versions, but the app can train a fresh compatible model locally.

## Lab submission

1. Run the complete workflow yourself and review the exported tables and interpretation.
2. Put the project in a GitHub repository; include the CSV only if course redistribution is permitted. A `.gitignore` excludes the local virtual environment and saved binary model. Text/CSV example results remain visible in the repository.
3. Record a demo of **at most 3 minutes**. Show dataset/cleaning briefly, selected features and split, best model, test metrics, the three SHAP charts, and one live forecast. Complete training and SHAP before recording.
4. Submit the repository link and demo video link through the course form.

`DEMO_CHECKLIST.md` contains a short recording outline. `LAB_REQUIREMENTS.md` maps the implementation to each lab stage. Next.js/FastAPI are intentionally outside this Streamlit MVP's scope; the handout lists them as an optional challenge.

## Troubleshooting

- **No module named …**: activate `.venv`; install requirements with `python -m pip` using the same interpreter.
- **TensorFlow has no matching distribution**: use Python 3.11 and check `uname -m`. Intel macOS requires the 2.16.2 branch selected by requirements.
- **Saved run fails to restore**: run Dataset → Forecast to create a local model with your installed versions.
- **Too few windows**: reduce lookback or increase the validation/test share. Every partition must contain more than `lookback + 1` valid months.
- **Training appears busy**: wait for the trial progress bar. Do not refresh during a long computation.
- **SHAP is slow**: start with 3–5 forecasts and 128 samples, then increase for the final interpretation.
- **Lost session state**: restore the locally saved run from Home or repeat the workflow.
- **Port 8501 is occupied**: run `python -m streamlit run Home.py --server.port 8502`.
