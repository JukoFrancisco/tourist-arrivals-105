# Lab requirement mapping

| Requirement | Implementation |
| --- | --- |
| One pure-Python Streamlit app | `Home.py` + eight files in `pages/` |
| CSV shape, types, monthly gaps | Dataset preview, coverage, dtypes and gap checks |
| Duplicates, missing values, IQR | Clean tabs; consistent duplicate rule; flags retained |
| Spearman followed by iterative VIF | Features; fixed thresholds; intercept in VIF; full removal log |
| Chronological split | Features defines fit/validation/test boundaries before selection |
| Training-only scalers | Prepare fits feature/target scalers on fitting rows |
| `(N, L, F)` sequences | Separate ordered windows per partition; default L=12 |
| LSTM training and tuning | Eight parameter combinations; explicit validation; early stopping |
| MAE, RMSE, MAPE, R² | Evaluate, in arrivals units |
| Naive and seasonal-naive comparison | Previous month and previous year on identical target dates |
| SHAP waterfall | Explain; actual cumulative floating bars with base and final prediction |
| Global feature importance | Explain; mean absolute influence over the stated sample |
| Dependence plot | Explain; matching raw-unit values and contributions at the selected lag (default lag 1) |
| Interactive forecast using own model | Forecast; editable latest window, same scalers/order, real inference |
| Short interpretation | Evaluate/Explain text and `results/INTERPRETATION.md` |
| GitHub + <=3-minute video | User submits links; README and demo checklist included |

## Deliberate corrections to starter code

- Learns feature selection and imputation from fitting rows only; the starter uses the whole series for selection.
- Uses a separate chronological validation set so its distribution is not used to fit preprocessing.
- Adds an intercept and empty/single-feature handling to VIF.
- Handles the supplied export's two title rows.
- Does not sample 50 SHAP backgrounds when fewer than 50 windows exist.
- Explains predictions in arrivals units and renders a real waterfall, not a contribution bar chart.
- Uses the latest window for a next-month forecast.
- Aligns baseline dates from the monthly index, including lookbacks shorter than 12 where original history is still available.
- Invalidates downstream session state after an upstream change.

The implementation keeps the handout's separate partition windowing. A test window using already-known training history could also be valid, but is not needed for this MVP. Missing targets stay unknown rather than being invented. Outlier corrections require external evidence, which the source does not supply.
