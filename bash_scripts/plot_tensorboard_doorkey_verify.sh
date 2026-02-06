#!/bin/bash
uncertainty="${1:-false}"
mkdir -p plots

uv run bt_as_reward/plotting/plot_doorkey_verify.py \
    --logdir ./results/tensorboard/ \
    --tag rollout/success_rate \
    --env_name "MiniGrid-DoorKey-16x16-v0" \
    --uncertainty "$uncertainty" \
    --out_file plots/MiniGrid-DoorKey-16x16-v0_verify_comparison_uncertainty="$uncertainty".pdf \
    --smooth 0.9 \