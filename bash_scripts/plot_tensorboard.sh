#!/bin/bash
env_name="${1:-MiniGrid-DoorKey-6x6-v0}"
uncertainty="${2:-false}"
mkdir -p plots

uv run bt_as_reward/plotting/plot_tensorboard.py \
    --logdir ./results/tensorboard/ \
    --tag rollout/success_rate \
    --env_name "$env_name" \
    --uncertainty "$uncertainty" \
    --out_file plots/"$env_name"_success_rate_uncertainty="$uncertainty".pdf \
    --smooth 0.9 \