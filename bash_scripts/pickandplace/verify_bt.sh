#!/bin/bash
# Usage: ./verify_bt.sh

uv run bt_as_reward/scripts/verify_llm_bt.py \
    --expert_trajs "trajs/pickandplace/expert.npz" \
    --random_trajs "trajs/pickandplace/random.npz" \
    --object_function_file "llm_functions/pickandplace/subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object" \
    --subtask_function_file "llm_functions/pickandplace/subtask_functions_gpt5.py" \
    --subtask_names "Pick Block, Move And Hold At Target" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete" \
    --verifier MuJoCo \
    --max_reward 2.0 \
    --output_file "llm_functions/pickandplace/response_bt.md" \
    --random_threshold 0.5 \
    --distance_threshold 0.05
    