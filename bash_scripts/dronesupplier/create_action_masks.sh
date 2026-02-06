#!/bin/bash
# Usage: ./create_action_masks.sh

uv run bt_as_reward/scripts/create_minigrid_action_masks.py \
    --expert_trajs "trajs/dronesupplier/expert.npz" \
    --object_function_file "llm_functions/dronesupplier/subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object, subtask_3_object" \
    --subtask_function_file "llm_functions/dronesupplier/subtask_functions_gpt5.py" \
    --subtask_names "Open the box, Pick up the key, Open the door" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete, subtask_3_complete" \
    --output_file "dronesupplier_action_masks_expert" \
    --distance_threshold 1.0
    