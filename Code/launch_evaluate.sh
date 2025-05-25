#!/bin/bash

PYTHON_BIN="/home/profes/miriamgf/anaconda3/envs/research3/bin/python"
echo "Usando Python: $PYTHON_BIN"


# Evaluación 4
LOG4="output/logs/evaluation/OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_strat_5_class_overs.log"
echo "Ejecutando Evaluación 4"
echo "Log: $LOG4"
nohup $PYTHON_BIN -u scripts/evaluation/evaluate_main.py \
  --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated_no_filt_l2_strat_5_class_overs' \
  --ev_DL true \
  --stratified_split true \
  --ev_TIK true \
  > "$LOG4" 2>&1
