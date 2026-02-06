#!/bin/bash
# Usage: ./verify_doorkey_bt.sh

uv run bt_as_reward/scripts/verify_llm_bt.py \
    --expert_trajs "trajs/doorkey/expert.npz" \
    --random_trajs "trajs/doorkey/random.npz" \
    --object_function_file "llm_functions/doorkey/subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object, subtask_3_object" \
    --subtask_function_file "llm_functions/doorkey/subtask_functions_gpt5.py" \
    --subtask_names "Acquire Key, Open Door, Reach Goal" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete, subtask_3_complete" \
    --verifier MiniGrid \
    --max_reward 3.0 \
    --output_file "llm_functions/doorkey/response_bt.md" \
    --random_threshold 0.5 \
    --distance_threshold 1.0
    