import sys
sys.path.append("../Code")
import os
import json
import argparse
import matplotlib.pyplot as plt
import pandas as pd
import argparse
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)
import time
import matplotlib.pyplot as plt

from scripts.evaluate_function import *
from tools_.tools_inference import *
from scripts.evaluation.evaluate_dl import EvaluateDL
from scripts.evaluation.evaluate_tik import EvaluateTikhonov
from scripts.config import str_to_bool
from scripts.evaluation.evaluate_regions import EvaluateRegions

#Argparse
try:
    print("Parsing bash params")
    parser = argparse.ArgumentParser(description="Noise params")
    parser.add_argument("--algorithm_ID", type=str, help="experiment name", required=True)
    parser.add_argument("--ev_DL", type=str_to_bool, help="True or False", required=False)
    parser.add_argument("--ev_TIK", type=str_to_bool, help="True or False", required=False)

    args = parser.parse_args()
    algorithm_ID = args.algorithm_ID
    evaluate_dl = args.ev_DL
    evaluate_tik = args.ev_TIK

    print(algorithm_ID, evaluate_dl, evaluate_tik)

except:
    algorithm_ID = "OMAMI_no_filt_testing2_repeated"
    evaluate_dl=False
    evaluate_tik=True


test_patients = [
            "LA_PLAW_140711_arm", "LA_RSPV_CAF_150115",
            "Simulation_01_200212_001_  5", "Simulation_01_200212_001_ 10",
            "Simulation_01_200316_001_  3", "Simulation_01_200316_001_  4",
            "Simulation_01_200316_001_  8", "Simulation_01_200428_001_004",
            "Simulation_01_200428_001_008", "Simulation_01_200428_001_010",
            "Simulation_01_210119_001_001", "Simulation_01_210208_001_002"
        ]



experiment_ID_list=[["OMAMI_no_filt_testing2_repeated"], ["OMAMI_VAE_no_filt_testing_repeated"]]#,["OMAMI_repeated"], ["OMAMI_VAE_Optuna_1"], ['OMAMI_no_filt'], ['OMAMI_VAE_no_filt']]


# DL -->  284 /12 = 23.666 seconds per patient (all the pipeline)
# TIK --> 777/ 12 = 64.75 seconds per patient (all the pipeline)
if evaluate_dl:
    print("Evaluating DL")
    evaluator_dl = EvaluateDL(test_patients=test_patients,algorithm_ID=algorithm_ID, test_id='0')
    evaluator_dl.run()
if evaluate_tik:
    print("Evaluating Tikhonov")
    evaluator_tik = EvaluateTikhonov(test_patients=test_patients,algorithm_ID=algorithm_ID, test_id='0')
    evaluator_tik.run()


# Evaluate in regions
EvaluateRegionsObj=EvaluateRegions(experiment_ID_list=experiment_ID_list, test_id='0')()


