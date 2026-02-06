#!/bin/bash
# Usage: ./create_action_masks.sh

uv run bt_as_reward/scripts/create_mujoco_action_masks.py \
    --expert_trajs "trajs/pickandplace2/expert.npz" \
    --object_function_file "llm_functions/pickandplace2/subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object" \
    --subtask_function_file "llm_functions/pickandplace2/subtask_functions_gpt5.py" \
    --subtask_names "Grasp Block, Move Block To Target" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete" \
    --output_file "pickandplace2_action_masks_expert" \
    --distance_threshold 0.05
    