#!/bin/bash
# Usage: ./train.sh <reward_mode>

reward_mode=$1
seed=$2
dropkey=$3
action_mask_file="${4:-none}"
dependent_subtask_file="${5:-none}"
if [ -z "$reward_mode" ]; then
    echo "Usage: $0 <reward_mode>"
    exit 1
fi

uv run bt_as_reward/training/minigrid_train.py \
    --seed $seed \
    --num_timesteps 1000000 \
    --reward_mode "$reward_mode" \
    --env_name "MiniGrid-DoorKey-16x16-v0" \
    --max_mission_words 15 \
    --drop_key $dropkey \
    --object_function_file "llm_functions/doorkey/subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object, subtask_3_object" \
    --subtask_function_file "llm_functions/doorkey/subtask_functions_gpt5.py" \
    --subtask_names "Acquire Key, Open Door, Reach Goal" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete, subtask_3_complete" \
    --distance_threshold 1.0 \
    --action_mask_file "$action_mask_file" \
    --dependent_subtask_file "$dependent_subtask_file" \
    