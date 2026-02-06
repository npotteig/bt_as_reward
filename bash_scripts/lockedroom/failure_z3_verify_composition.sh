#!/bin/bash
# Usage: ./z3_verify_composition.sh

use_trajs="${1:-false}"

if [ "$use_trajs" = "true" ]; then
    uv run bt_as_reward/scripts/verify_llm_bt.py \
        --use_z3 \
        --env_name "MiniGrid-LockedRoom-v0" \
        --subtask_function_file "llm_functions/lockedroom/z3_subtask_functions_gpt5.py" \
        --subtask_function_names "subtask_1_complete_failure, subtask_2_complete, subtask_3_complete, subtask_4_complete" \
        --verifier MiniGrid \
        --output_file "llm_functions/lockedroom/z3_response_composition_failure.md" \
        --expert_trajs "trajs/lockedroom/expert.npz"
else
    uv run bt_as_reward/scripts/verify_llm_bt.py \
        --use_z3 \
        --env_name "MiniGrid-LockedRoom-v0" \
        --subtask_function_file "llm_functions/lockedroom/z3_subtask_functions_gpt5.py" \
        --subtask_function_names "subtask_1_complete_failure, subtask_2_complete, subtask_3_complete, subtask_4_complete" \
        --verifier MiniGrid \
        --output_file "llm_functions/lockedroom/z3_response_composition_failure.md" 
fi