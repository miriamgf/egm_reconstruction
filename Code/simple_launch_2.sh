python -u scripts/train_augmented.py \
    --algorithm 'OMAMI_VAE' --optuna false --discard_classes true \
    --filter_EGM false --split_mode 'stratified' --oversampling true \
    --data_augmentation true --perc_augmentation 25 \
    2>&1 | tee output/logs/cond_vae_2_4_guided.log