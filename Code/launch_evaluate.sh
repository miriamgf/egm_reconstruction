

# Esperar 7 horas (7 * 60 * 60 segundos)

#

#echo "output/logs/evaluation/OMAMI_VAE_no_filt.log"
#nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_VAE_no_filt.log 2>&1 &
#wait

#echo "output/logs/evaluation/OMAMI_VAE_no_filt_testing.log"
#nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_VAE_no_filt_testing_repeated' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_VAE_no_filt_testing.log 2>&1 
#wait

#echo "output/logs/evaluation/OMAMI_no_filt_testing2.log"
#nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_no_filt_testing2.log 2>&1 &
#wait

#!/bin/bash

#!/bin/bash


echo "output/logs/evaluation/OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs.log"
nohup python -u scripts/evaluation/evaluate_main.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs' --ev_DL true --ev_TIK true > output/logs/evaluation/OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs.log 2>&1
