#!/bin/bash
# Usage: ./z3_verify_composition.sh

use_trajs="${1:-false}"

if [ "$use_trajs" = "true" ]; then
    uv run bt_as_reward/scripts/verify_llm_bt.py \
        --use_z3 \
        --env_name "FetchPickAndPlace-v5" \
        --subtask_function_file "llm_functions/pickandplace/z3_subtask_functions_gpt5.py" \
        --subtask_function_names "subtask_1_complete, subtask_2_complete" \
        --verifier MuJoCo \
        --output_file "llm_functions/pickandplace/z3_response_composition.md" \
        --expert_trajs "trajs/pickandplace/expert.npz" 
else
    uv run bt_as_reward/scripts/verify_llm_bt.py \
        --use_z3 \
        --env_name "FetchPickAndPlace-v5" \
        --subtask_function_file "llm_functions/pickandplace/z3_subtask_functions_gpt5.py" \
        --subtask_function_names "subtask_1_complete, subtask_2_complete" \
        --verifier MuJoCo \
        --output_file "llm_functions/pickandplace/z3_response_composition.md" 
fi
    