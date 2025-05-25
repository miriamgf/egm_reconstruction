#!/bin/bash


echo "Saving logs at: output/logs/plots/OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs_3D.log"
nohup python -u scripts/visualization/3d_plot_egm_bpsm.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs' > output/logs/plots/OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs_3D.log 2>&1 &
PID=$!
wait $PID


echo "Saving logs at: output/logs/plots/OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_2_class_overs_2D.log"
nohup python -u scripts/visualization/2d_plot.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_2_class_overs' > output/logs/plots/OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_2_class_overs_2D.log 2>&1 &
PID=$!
wait $PID

echo "Saving logs at: output/logs/plots/OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs_2D.log"
nohup python -u scripts/visualization/2d_plot.py --algorithm_ID 'OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs' > output/logs/plots/OMAMI_no_filt_testing2_repeated_no_filt_l2_strat_5_class_overs_2D.log 2>&1 &
PID=$!
wait $PID