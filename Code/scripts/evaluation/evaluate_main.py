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
from tools_.stratified_split import StratifiedSplit

experiment_dir = f"/home/pdi/miriamgf/tesis/Autoencoders/code/egm_reconstruction/Code/output/experiments/experiments_VAE/"


#Argparse
if len(sys.argv) > 1:
    print("Parsing bash params")
    parser = argparse.ArgumentParser(description="Noise params")
    parser.add_argument("--algorithm_ID", type=str, help="experiment name", required=False)
    parser.add_argument("--ev_DL", type=str_to_bool, help="True or False", required=False)
    parser.add_argument("--ev_TIK", type=str_to_bool, help="True or False", required=False)
    parser.add_argument("--stratified_split", type=str_to_bool, help="True or False", required=False)

    args = parser.parse_args()
    algorithm_ID = args.algorithm_ID
    evaluate_dl = args.ev_DL
    evaluate_tik = args.ev_TIK
    stratified_split = args.stratified_split

    print(algorithm_ID, evaluate_dl, evaluate_tik)

else: 
    algorithm_ID = "OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_strat_2_class_overs"
    evaluate_dl=False
    evaluate_tik=True
    stratified_split=True

if stratified_split:
    experiment_dir=f"{experiment_dir}{algorithm_ID}"
    StratifiedSplit_obj = StratifiedSplit(classes_to_oversample=None,
                                            discard_classes=None,
                                            oversampling=None)
    print("Stratified split")
    test_patients = StratifiedSplit_obj.get_test_for_inference(experiment_dir)
else:
    print('Deterministic split')

    test_patients = [
                "LA_PLAW_140711_arm", "LA_RSPV_CAF_150115",
                "Simulation_01_200212_001_  5", "Simulation_01_200212_001_ 10",
                "Simulation_01_200316_001_  3", "Simulation_01_200316_001_  4",
                "Simulation_01_200316_001_  8", "Simulation_01_200428_001_004",
                "Simulation_01_200428_001_008", "Simulation_01_200428_001_010",
                "Simulation_01_210119_001_001", "Simulation_01_210208_001_002"
            ]



#experiment_ID_list=[["OMAMI_no_filt_testing2_repeated"], ["OMAMI_VAE_no_filt_testing_repeated"]]#,["OMAMI_repeated"], ["OMAMI_VAE_Optuna_1"], ['OMAMI_no_filt'], ['OMAMI_VAE_no_filt']]


# DL -->  284 /12 = 23.666 seconds per patient (all the pipeline)
# TIK --> 777/ 12 = 64.75 seconds per patient (all the pipeline)
if evaluate_dl:
    print("Evaluating DL")
    evaluator_dl = EvaluateDL(test_patients=test_patients,algorithm_ID=algorithm_ID, test_id='_str_test')
    evaluator_dl.run()
if evaluate_tik:
    print("Evaluating Tikhonov")
    evaluator_tik = EvaluateTikhonov(test_patients=test_patients,algorithm_ID=algorithm_ID, test_id='_str_test')
    evaluator_tik.run()


# Evaluate in regions
#EvaluateRegionsObj=EvaluateRegions(experiment_ID_list=experiment_ID_list, test_id='0')()


