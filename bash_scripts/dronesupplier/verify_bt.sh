#!/bin/bash
# Usage: ./verify_bt.sh

uv run bt_as_reward/scripts/verify_llm_bt.py \
    --expert_trajs "trajs/dronesupplier/expert.npz" \
    --random_trajs "trajs/dronesupplier/random.npz" \
    --object_function_file "llm_functions/dronesupplier/subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object, subtask_3_object" \
    --subtask_function_file "llm_functions/dronesupplier/subtask_functions_gpt5.py" \
    --subtask_names "Open the box, Pick up the key, Open the door" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete, subtask_3_complete" \
    --verifier MiniGrid \
    --max_reward 3.0 \
    --output_file "llm_functions/dronesupplier/response_bt.md" \
    --random_threshold 0.5 \
    --distance_threshold 1.0
    