#!/bin/bash
# Usage: ./verify_subtask.sh <function_name>
function_name=$1
if [ -z "$function_name" ]; then
    echo "Usage: $0 <function_name>"
    exit 1
fi

uv run bt_as_reward/scripts/verify_llm_subtask.py \
    --function_name "$function_name" \
    --function_file "llm_functions/dronesupplier/subtask_functions_gpt5.py" \
    --expert_trajs "trajs/dronesupplier/expert.npz" \
    --random_trajs "trajs/dronesupplier/random.npz" \
    --verifier MiniGrid \
    --output_file "llm_functions/dronesupplier/response_$function_name.md" \
    --random_threshold 0.5 \