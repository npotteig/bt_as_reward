#!/bin/bash
# Usage: ./create_lockedroom_trajs.sh

uv run bt_as_reward/scripts/create_minigrid_trajs.py \
    --env_name "MiniGrid-LockedRoom-v0" \
    --num_expert_trajs 1000 \
    --num_random_trajs 1000 \