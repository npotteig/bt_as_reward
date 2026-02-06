#!/bin/bash
# Usage: ./create_doorkey_trajs.sh

uv run bt_as_reward/scripts/create_minigrid_trajs.py \
    --env_name "MiniGrid-DoorKey-6x6-v0" \
    --num_expert_trajs 100 \
    --num_random_trajs 100 \