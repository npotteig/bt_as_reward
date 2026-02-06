#!/bin/bash
# Usage: ./create_trajs.sh

uv run bt_as_reward/scripts/create_mujoco_trajs.py \
    --env_name "FetchPickAndPlace2-v1" \
    --num_expert_trajs 100 \
    --num_random_trajs 100 \