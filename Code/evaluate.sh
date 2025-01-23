echo "Launching evaluate_dl.py"
nohup python scripts/evaluation/evaluate_dl.py --algorithm_ID 'OMAMI_VAE' > output/logs/evaluation/evaluate_dl.log 2>&1 &

echo "Launching evaluate_tik.py"
nohup python scripts/evaluation/evaluate_tik.py --algorithm_ID 'OMAMI_VAE' > output/logs/evaluation/evaluate_tik.log 2>&1 &

# Wait for both background processes to finish
wait

echo "All evaluation scripts have completed."
