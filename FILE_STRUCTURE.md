# Project structure

This is one Streamlit process with a clear separation between presentation and application logic. There is no separate HTTP API or independently deployed frontend. That matches the lab's required Streamlit version; FastAPI and Next.js remain optional.

| Location | Responsibility |
| --- | --- |
| `Home.py` | Overview page and Streamlit entrypoint |
| `pages/` | Eight Streamlit screens, forms, charts and actions |
| `frontend/ui.py` | Shared page layout, navigation, stage guards and chart presentation |
| `frontend/assets/` | Logo and application CSS |
| `backend/pipeline.py` | CSV validation, cleaning, selection, scaling, sequences and scoring |
| `backend/modeling.py` | LSTM creation, tuning and prediction |
| `backend/explainability.py` | Kernel SHAP computations and waterfall data |
| `backend/storage.py` | Local model/run saving and restoration |
| `backend/reporting.py` | Interpretation text and portable result exports |
| `utils/` | Small compatibility imports for earlier project imports; no duplicate implementation |
| `data/` | Original course dataset and source notes |
| `artifacts/saved_run/` | Saved model, scalers, run state and library-version record |
| `results/` | Bundled example CSV results and interpretation |
| `tests/` | Data-pipeline and report verification |
| `.streamlit/config.toml` | Theme, viewer toolbar and navigation settings |

The pages call backend functions directly. Streamlit owns browser widgets and session state. Backend calculations are reusable without introducing an API, additional dependencies or a second server.

Run from the project root using `bash run_app.command` on Mac, or activate the environment and run `python -m streamlit run Home.py`.
