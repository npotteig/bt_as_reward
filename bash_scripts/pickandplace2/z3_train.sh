#!/bin/bash
# Usage: ./train.sh <reward_mode>

reward_mode=$1
seed=$2
grip_fail=$3
action_mask_file="${4:-none}"
if [ -z "$reward_mode" ]; then
    echo "Usage: $0 <reward_mode>"
    exit 1
fi

uv run bt_as_reward/training/mujoco_train.py \
    --seed $seed \
    --max_mission_words 12 \
    --grip_fail $grip_fail \
    --num_timesteps 5000000 \
    --reward_mode "$reward_mode" \
    --env_name "FetchPickAndPlace2-v1" \
    --object_function_file "llm_functions/pickandplace2/z3_subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object" \
    --subtask_function_file "llm_functions/pickandplace2/z3_subtask_functions_gpt5.py" \
    --subtask_names "Grasp Block, Move Block To Target" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete" \
    --distance_threshold 0.05 \
    --action_mask_file "$action_mask_file" \
    --use_z3
    