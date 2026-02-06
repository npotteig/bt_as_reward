#!/bin/bash
# Usage: ./z3_verify_subtask.sh <function_name>
function_name=$1
use_trajs="${2:-false}"
if [ -z "$function_name" ]; then
    echo "Usage: $0 <function_name>"
    exit 1
fi

if [ "$use_trajs" = "true" ]; then
    uv run bt_as_reward/scripts/verify_llm_subtask.py \
        --function_name "$function_name" \
        --function_file "llm_functions/pickandplace2/z3_subtask_functions_gpt5.py" \
        --use_z3 \
        --verifier MuJoCo \
        --env_name "FetchPickAndPlace2-v1" \
        --output_file "llm_functions/pickandplace2/z3_response_$function_name.md" \
        --expert_trajs "trajs/pickandplace2/expert.npz" \
        --random_trajs "trajs/pickandplace2/random.npz"
else
    uv run bt_as_reward/scripts/verify_llm_subtask.py \
        --function_name "$function_name" \
        --function_file "llm_functions/pickandplace2/z3_subtask_functions_gpt5.py" \
        --use_z3 \
        --verifier MuJoCo \
        --env_name "FetchPickAndPlace2-v1" \
        --output_file "llm_functions/pickandplace2/z3_response_$function_name.md" 
fi