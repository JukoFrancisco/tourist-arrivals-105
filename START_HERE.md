# Start here

1. Extract the ZIP and open `tourist-arrivals-app` in VS Code.
2. Use Python 3.11. Open Terminal → New Terminal.
3. Run these commands on Mac:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m streamlit run Home.py
```

4. Open http://localhost:8501.
5. For a quick preview, choose **Restore saved run** on Home. All results use the actual course data and trained model. If restoration fails because of version differences, run the pages from Dataset onward.
6. For your own work, follow Dataset → Clean → Features → Prepare → Train → Evaluate → Explain → Forecast. Use the default settings first.
7. Read `results/INTERPRETATION.md`, then explain your own results. The included LSTM loses to the baselines; the handout accepts this outcome when reported honestly.
8. Follow `DEMO_CHECKLIST.md`, push your repository and submit its link with a demo video link (maximum three minutes).

`README.md` contains Windows setup, troubleshooting and methodology. `LAB_REQUIREMENTS.md` maps every requirement to its implementation.

## Updating your existing MVP

Read `UPGRADE.md`. Keep your working `.venv` and replace the source files and add both `backend/` and `frontend/`, including `frontend/assets/`. There are no new dependencies.

To launch from your project folder:

```bash
bash run_app.command
```

## macOS 26.1 / Homebrew XML-library error

If your error mentions `_XML_SetAllocTrackerActivationThreshold`, the Homebrew Python XML module may be loading an incompatible macOS expat library. The session workaround used during setup is:

```bash
brew install expat
export DYLD_LIBRARY_PATH="/opt/homebrew/opt/expat/lib"
.venv/bin/python -c "import pyexpat, platform; print('XML parser OK'); print(platform.mac_ver())"
```

Once this check succeeds, install pip if needed and then the requirements. The new launcher applies the same library path for the app process when needed; it does not install anything or edit shell profiles. It has been syntax-checked here but has not been executed on your physical Mac.
