# Verification record

Validated in Linux using Python 3.12.14, TensorFlow CPU 2.21.0, Keras 3.15.1, Streamlit 1.65.0, NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 and SHAP 0.53.0. Additional installed versions are recorded with the saved run. Mac-specific requirements are supplied but were not executed on a physical Mac.

Completed checks:

- Parsed the actual supplied CSV, including its two leading title lines.
- Cleaned 314 rows into 312 unique monthly observations; retained extreme values and unknown targets according to the documented rules.
- Ran fitting-only feature selection, chronological preparation and all eight model trials, with up to 80 epochs and validation early stopping.
- Evaluated the real trained model and both baselines on 51 identical held-out dates.
- Generated 10 actual Kernel SHAP explanations, 30 background windows, 256 samples per forecast. Checked base + signed contributions against the predictions with a floating-point tolerance; maximum discrepancy below one arrival.
- Saved and restored the trained model, scalers and pipeline results.
- Passed **7 automated pipeline tests**, including protection against future data affecting feature selection/scaler parameters, sequence/target alignment, baseline alignment, missing targets, zero-valued metric handling, category encoding and forecast input validation.
- Used Streamlit AppTest to click the Dataset, Clean, Features and Prepare actions from a fresh workflow. Rendered Home and all eight completed pages without application exceptions. Submitted the Forecast form and confirmed the stage counter reaches 8 of 8.
- Used a Chromium browser to restore a saved run, navigate result pages and generate the next-month forecast with no displayed application exception.
- Verified genuine scenario inference: increasing the last historical wave-height reading by 0.1 m changed the saved model's point prediction from approximately 1,130,161 to 1,162,036 arrivals. This is a model-sensitivity check, not a causal claim.
- Compiled all project Python files successfully.

The included model's poor test performance is documented in `results/INTERPRETATION.md`. Successful software checks do not imply an accurate forecasting model. Numerical outputs can differ across platforms and package versions.

## UI and workflow update — October 14, 2026

- All **9 automated tests** passed, including two portable-report checks.
- Streamlit AppTest rendered Overview and all eight completed pages without exceptions, using Home as the multipage entrypoint.
- A fresh session loaded the CSV, cleaned it, selected predictors and prepared sequences through the actual page buttons.
- Submitting an unchanged forecast produced approximately 1,130,161 arrivals. Editing the final wave-height reading by 0.1 m produced approximately 1,162,036; the unchanged-history comparison remained fixed and the change log contained the edited cell.
- Resetting scenario inputs cleared the submitted forecast and restored the editor defaults.
- Changing the SHAP predictor and lag rendered without exceptions.
- A two-epoch, single-configuration training smoke check exercised the per-epoch progress callback. The bundled eight-trial model and its existing results were retained.
- Chromium browser checks restored the saved run, used the custom sidebar, submitted a forecast and found no displayed application exception. Home and Forecast layouts were inspected visually.
- Python compilation and the Mac launcher's shell syntax check passed. The launcher has not been executed on a physical Mac.

## Final layout and organization patch

- Inspected the supplied Overview screenshot and corrected insufficient clearance under Streamlit's fixed toolbar.
- Browser checks verified the hero starts below the toolbar at 1440px and 390px viewport widths and Deploy is not visible.
- Inspected final desktop and mobile Overview layouts.
- Moved model/data/persistence/report logic to `backend/`, shared layout and assets to `frontend/`, and updated page imports. Compatibility imports remain in `utils/`.
- All 9 automated tests passed after the reorganization.
- Overview and every lab page rendered without exceptions in Streamlit AppTest and the Chromium browser. The saved model restored and generated the same unchanged-history forecast (approximately 1,130,161 arrivals).
- Python compilation and launcher shell syntax checks passed. No additional libraries or separate API server were introduced.
