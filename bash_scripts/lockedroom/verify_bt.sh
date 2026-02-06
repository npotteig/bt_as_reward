#!/bin/bash
# Usage: ./verify_lockedroom_bt.sh

uv run bt_as_reward/scripts/verify_llm_bt.py \
    --expert_trajs "trajs/lockedroom/expert.npz" \
    --random_trajs "trajs/lockedroom/random.npz" \
    --object_function_file "llm_functions/lockedroom/subtask_functions_gpt5.py" \
    --object_function_names "subtask_1_object, subtask_2_object, subtask_3_object, subtask_4_object" \
    --subtask_function_file "llm_functions/lockedroom/subtask_functions_gpt5.py" \
    --subtask_names "Open key-room door, Pick up key, Unlock/open locked door, Reach goal" \
    --subtask_function_names "subtask_1_complete, subtask_2_complete, subtask_3_complete, subtask_4_complete" \
    --verifier MiniGrid \
    --max_reward 4.0 \
    --output_file "llm_functions/lockedroom/response_bt.md" \
    --random_threshold 0.5 \
    --distance_threshold 1.0
    