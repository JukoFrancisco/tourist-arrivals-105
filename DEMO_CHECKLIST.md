# Demo outline — maximum 3 minutes

Prepare a completed run before recording. Start from Home and restore it if necessary.

| Time | Show | Explain briefly |
| --- | --- | --- |
| 0:00–0:15 | Home | One-month tourist-arrivals forecasts with LSTM and SHAP |
| 0:15–0:40 | Dataset and Clean | 314 raw rows, 312 months, duplicates/missing values, outliers retained |
| 0:40–1:05 | Features and Prepare | Spearman → VIF; chronological fitting/validation/test split; 12-month history |
| 1:05–1:25 | Train | Eight combinations, validation loss, early-stopped best model |
| 1:25–1:50 | Evaluate | Compare all methods on identical dates; state that baselines beat this LSTM |
| 1:50–2:25 | Explain | Global importance, base-to-prediction waterfall, lag-matched dependence |
| 2:25–2:50 | Forecast | Edit a valid historical input, submit, show new prediction |
| 2:50–2:55 | Closing | Dataset anomalies and temporal shifts limit reliability; explanations are not causal |

Before submission:

- Review the interpretation using your own run and explain it in your own words.
- Ensure the repository contains setup instructions and required source files.
- Check that your video link can be opened by the instructor.
- Submit both links in the course form.
