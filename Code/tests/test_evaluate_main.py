import pytest
from unittest.mock import patch, MagicMock
from scripts.evaluation.evaluate_main import EvaluateDL, EvaluateTikhonov

@pytest.fixture
def test_patients():
    return [
        "LA_PLAW_140711_arm", "LA_RSPV_CAF_150115",
        "Simulation_01_200212_001_  5", "Simulation_01_200212_001_ 10",
        "Simulation_01_200316_001_  3", "Simulation_01_200316_001_  4",
        "Simulation_01_200316_001_  8", "Simulation_01_200428_001_004",
        "Simulation_01_200428_001_008", "Simulation_01_200428_001_010",
        "Simulation_01_210119_001_001", "Simulation_01_210208_001_002"
    ]

@patch('scripts.evaluation.evaluate_main.EvaluateDL.run')
def test_evaluate_dl(mock_run, test_patients):
    algorithm_ID = "OMAMI_VAE_Optuna_1"
    evaluator_dl = EvaluateDL(patient_data=test_patients, algorithm_ID=algorithm_ID)
    evaluator_dl.run()
    mock_run.assert_called_once()

@patch('scripts.evaluation.evaluate_main.EvaluateTikhonov.run')
def test_evaluate_tik(mock_run, test_patients):
    algorithm_ID = "OMAMI_VAE_Optuna_1"
    evaluator_tik = EvaluateTikhonov(patient_data=test_patients, algorithm_ID=algorithm_ID)
    evaluator_tik.run()
    mock_run.assert_called_once()