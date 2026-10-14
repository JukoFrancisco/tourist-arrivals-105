import io
import json
import zipfile

import pandas as pd

from backend.reporting import export_report


def test_run_export_keeps_scores_and_metadata():
    metrics = pd.DataFrame({'MAE': [10., 8., 7.], 'RMSE': [12., 10., 9.]},
                           index=['LSTM', 'Naive', 'Seasonal naive'])
    state = {'source_name': 'example.csv',
             'evaluation': {'metrics': metrics, 'predictions': pd.DataFrame({'Actual': [100], 'LSTM': [90]})},
             'prepared': {'features': ['rainfall_mm'], 'lookback': 12, 'summary': pd.DataFrame({'partition': ['fit']})}}
    with zipfile.ZipFile(io.BytesIO(export_report(state))) as archive:
        exported = pd.read_csv(archive.open('metrics.csv'), index_col=0)
        pd.testing.assert_frame_equal(metrics, exported)
        details = json.loads(archive.read('run_details.json'))
        assert details['features'] == ['rainfall_mm'] and details['lookback'] == 12
        assert 'Seasonal naive' in archive.read('interpretation.md').decode()
        assert 'model.keras' not in archive.namelist()


def test_partial_run_export_is_readable():
    with zipfile.ZipFile(io.BytesIO(export_report({}))) as archive:
        assert 'README.txt' in archive.namelist()
        assert json.loads(archive.read('run_details.json'))['features'] is None
