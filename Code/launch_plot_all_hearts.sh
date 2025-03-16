echo " output/logs/plots/all_hearts.log"
nohup python -u scripts/visualization/plot_all_hearts.py -- 'OMAMI_no_filt_testing2_repeated' > output/logs/plots/all_hearts.log 2>&1 
