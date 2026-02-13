#!/bin/bash
# Usage: ./train.sh <reward_mode>

model_path=$1
reward_mode=$2
drop_key=$3
action_mask_file="${4:-none}"
if [ -z "$model_path" ] || [ -z "$reward_mode" ]; then
    echo "Usage: $0 <model_path> <reward_mode> <drop_key>"
    exit 1
fi



uv run drone/test_policy.py \
    --model_path "$model_path" \
    --reward_mode "$reward_mode" \
    --use_airsim "true" \
    --drop_key $drop_key \
    --object_function_file "llm_functions/dronesupplier/z3_subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object, subtask_3_object" \
    --subtask_function_file "llm_functions/dronesupplier/z3_subtask_functions_gpt5.py" \
    --subtask_names "Open the box, Pick up the key, Open the door" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete, subtask_3_complete" \
    --distance_threshold 1.0 \
    --action_mask_file $action_mask_file \
    