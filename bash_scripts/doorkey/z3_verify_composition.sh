#!/bin/bash
# Usage: ./z3_verify_composition.sh

use_trajs="${1:-false}"

if [ "$use_trajs" = "true" ]; then
    uv run bt_as_reward/scripts/verify_llm_bt.py \
        --use_z3 \
        --env_name "MiniGrid-DoorKey-6x6-v0" \
        --subtask_function_file "llm_functions/doorkey/z3_subtask_functions_gpt5.py" \
        --subtask_function_names "subtask_1_complete, subtask_2_complete, subtask_3_complete" \
        --verifier MiniGrid \
        --output_file "llm_functions/doorkey/z3_response_composition.md" \
        --expert_trajs "trajs/doorkey/expert.npz"
else
     uv run bt_as_reward/scripts/verify_llm_bt.py \
        --use_z3 \
        --env_name "MiniGrid-DoorKey-6x6-v0" \
        --subtask_function_file "llm_functions/doorkey/z3_subtask_functions_gpt5.py" \
        --subtask_function_names "subtask_1_complete, subtask_2_complete, subtask_3_complete" \
        --verifier MiniGrid \
        --output_file "llm_functions/doorkey/z3_response_composition.md" 
fi
    