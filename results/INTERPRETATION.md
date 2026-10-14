# Interpretation of the included run

This is an actual run on the supplied course CSV, not example scores. Defaults: numeric predictors, 80% development / 20% test, validation = the final 15% of development, 12-month lookback, eight LSTM trials, up to 80 epochs and patience 8.

## Data and selected predictors

The source contains 314 rows. Keeping the first duplicate record leaves 312 monthly observations from January 2000 through December 2025. Three arrivals values remain unknown and are excluded as prediction targets. In particular, the February 2001 arrivals figure of 34,240,153 is flagged by IQR but retained because no external correction record was provided. It strongly widens the target's training scale; its contribution to poor generalization is a plausible limitation, not a proven explanation of all errors.

Eight predictors remain after fitting-only Spearman screening and VIF:

- quarter
- is_holiday_peak
- rainy_days
- humidity_pct
- typhoon_max_wind_kt
- storm_signal_days
- pm25_ugm3
- wave_height_m

Quarter and holiday indicators describe the seasonal timing of travel. Rainy days, waves and hazard readings describe conditions that may coincide with travel disruptions. The observed associations do not prove those conditions caused a change in arrivals. Temperature candidates did not pass screening on the fitting period; rainfall and typhoon count were removed during redundancy checks.

## Training and evaluation

Fitting rows end in July 2017, validation rows cover August 2017–September 2020, and the test partition covers October 2020–December 2025. Separate partition windowing means the scored test targets are **October 2021–December 2025: 51 months**. There are 196 fitting windows and 26 validation windows.

Validation selected **64 LSTM units, dropout 0.3 and batch size 16**. Test results did not select the settings.

| Model | MAE | RMSE | MAPE | R² |
| --- | ---: | ---: | ---: | ---: |
| LSTM | 293,975 | 383,982 | 201.73% | -3.158 |
| Naive | 64,242 | 83,921 | 16.73% | 0.801 |
| Seasonal naive | 134,877 | 176,340 | 39.20% | 0.123 |

The previous-month naive forecast is the strongest of the three on this test period. The LSTM's negative R² shows poor generalization relative to a constant prediction of the test mean. A low scaled validation loss did not translate into a useful test forecast. The result does not justify deploying this LSTM for planning decisions.

Possible limitations include the extreme retained training observation, temporal changes between the fitting and test periods, a small monthly sample and the absence of lagged arrivals in the lab's predictor-only LSTM inputs. These are considerations for follow-up work, not reasons to alter test observations or select settings using the test score. A trustworthy correction to a documented data-entry error, more suitable modeling assumptions or walk-forward development validation could be studied separately.

## SHAP interpretation

Ten test forecasts were sampled across the scored period. Kernel SHAP used 30 fitting background windows and 256 evaluations per explanation, with a sparse top-10 lag-cell attribution setting. The largest average absolute contributions belong to **wave height**, **holiday peak** and **PM2.5**. This ranking describes the model's learned reliance on these predictors over the sampled windows. It does not imply that pollution or wave conditions directly determine tourism, and it is not evidence that the model is accurate.

Holiday indicators provide seasonal context. Wave height can reflect weather and marine conditions, while PM2.5 may act as a proxy for seasonal conditions or other correlated variables. The charts should be read alongside their small sample and the model's poor evaluation scores. Direction must be assessed from the signed waterfall or dependence points rather than the global absolute-importance bars.

The waterfall starts from the background expected prediction and accumulates each predictor's contributions over all 12 historical months. The dependence plot uses original-unit wave height from the previous month and that same lag's contribution. Floating-point differences between batched and single-sample LSTM calls give an additivity discrepancy below one arrival in this run.

## Forecast use

The Forecast page performs genuine inference using January–December 2025 predictor history to forecast January 2026. Input edits represent hypothetical scenarios. This is an instructional point forecast; its validity is limited by the evaluation above. No confidence interval is claimed. Reproduce the workflow and explain the results in your own words for submission.
