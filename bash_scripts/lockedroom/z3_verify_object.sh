#!/bin/bash
# Usage: ./z3_verify_object.sh <object_function_name> <subtask_function_name>
object_function_name=$1
subtask_function_name=$2
use_trajs="${3:-false}"
if [ -z "$object_function_name" ] || [ -z "$subtask_function_name" ]; then
    echo "Usage: $0 <object_function_name> <subtask_function_name>"
    exit 1
fi

if [ "$use_trajs" = "true" ]; then
    uv run bt_as_reward/scripts/verify_llm_object.py \
        --object_function_name "$object_function_name" \
        --object_function_file "llm_functions/lockedroom/z3_subtask_functions_gpt5.py" \
        --subtask_function_name "$subtask_function_name" \
        --subtask_function_file "llm_functions/lockedroom/z3_subtask_functions_gpt5.py" \
        --verifier MiniGrid \
        --use_z3 \
        --env_name MiniGrid-LockedRoom-v0 \
        --output_file "llm_functions/lockedroom/z3_response_$object_function_name.md" \
        --expert_trajs "trajs/lockedroom/expert.npz" \
        --random_trajs "trajs/lockedroom/random.npz"
else
    uv run bt_as_reward/scripts/verify_llm_object.py \
        --object_function_name "$object_function_name" \
        --object_function_file "llm_functions/lockedroom/z3_subtask_functions_gpt5.py" \
        --subtask_function_name "$subtask_function_name" \
        --subtask_function_file "llm_functions/lockedroom/z3_subtask_functions_gpt5.py" \
        --verifier MiniGrid \
        --use_z3 \
        --env_name MiniGrid-LockedRoom-v0 \
        --output_file "llm_functions/lockedroom/z3_response_$object_function_name.md"
fi