#!/bin/bash
# Usage: ./create_action_masks.sh

uv run bt_as_reward/scripts/create_minigrid_action_masks.py \
    --expert_trajs "trajs/doorkey/expert.npz" \
    --object_function_file "llm_functions/doorkey/z3_subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object, subtask_3_object" \
    --subtask_function_file "llm_functions/doorkey/z3_subtask_functions_gpt5.py" \
    --subtask_names "Acquire Key, Open Door, Reach Goal" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete, subtask_3_complete" \
    --output_file "doorkey_action_masks_expert" \
    --distance_threshold 1.0
    