import sys
sys.path.append("../Code")
import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import time
import matplotlib.pyplot as plt

from scripts.evaluate_function import *
from tools_.tools_inference import *
from scripts.evaluation.evaluate_dl import EvaluateDL
from scripts.evaluation.evaluate_tik import EvaluateTikhonov


algorithm_ID = "OMAMI_VAE_Optuna_1"
evaluate_dl=True
evaluate_tik=False

test_patients = [
            "LA_PLAW_140711_arm", "LA_RSPV_CAF_150115",
            "Simulation_01_200212_001_  5", "Simulation_01_200212_001_ 10",
            "Simulation_01_200316_001_  3", "Simulation_01_200316_001_  4",
            "Simulation_01_200316_001_  8", "Simulation_01_200428_001_004",
            "Simulation_01_200428_001_008", "Simulation_01_200428_001_010",
            "Simulation_01_210119_001_001", "Simulation_01_210208_001_002"
        ]

# DL -->  284 /12 = 23.666 seconds per patient (all the pipeline)
# TIK --> 777/ 12 = 64.75 seconds per patient (all the pipeline)
if evaluate_dl:
    print("Evaluating DL")
    evaluator_dl = EvaluateDL(test_patients=test_patients,algorithm_ID=algorithm_ID)
    evaluator_dl.run()

if evaluate_tik:
    print("Evaluating Tikhonov")
    evaluator_tik = EvaluateTikhonov(test_patients=test_patients,algorithm_ID=algorithm_ID)
    evaluator_tik.run()

