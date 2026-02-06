#!/bin/bash
# Usage: ./verify_lockedroom_object.sh <object_function_name> <subtask_function_name>
object_function_name=$1
subtask_function_name=$2
if [ -z "$object_function_name" ] || [ -z "$subtask_function_name" ]; then
    echo "Usage: $0 <object_function_name> <subtask_function_name>"
    exit 1
fi

uv run bt_as_reward/scripts/verify_llm_object.py \
    --object_function_name "$object_function_name" \
    --object_function_file "llm_functions/lockedroom/subtask_functions_gpt5.py" \
    --subtask_function_name "$subtask_function_name" \
    --subtask_function_file "llm_functions/lockedroom/subtask_functions_gpt5.py" \
    --expert_trajs "trajs/lockedroom/expert.npz" \
    --verifier MiniGrid \
    --output_file "llm_functions/lockedroom/response_$object_function_name.md" \
    --distance_threshold 1.0 \