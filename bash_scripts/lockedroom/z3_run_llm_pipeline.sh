#!/bin/bash
# Usage: ./z3_run_llm_pipeline.sh

uv run python bt_as_reward/scripts/z3_full_llm_pipeline.py \
    --system_prompt_file "llm_prompts/minigrid/z3_lockedroom.md" \
    --user_prompt_file "llm_prompts/minigrid/prompts.md" \
    --chat_history_ckpt "llm_functions/lockedroom/z3_lockedroom_chat_history.json" \
    --chat_history_output_file "llm_functions/lockedroom/z3_lockedroom_chat_history.json" \
    --code_file "llm_functions/lockedroom/z3_lockedroom_code.py" \
    --verifier MiniGrid \
    --llm_action_mask_file "lockedroom_action_masks_llm" \
    --verify_env_name "MiniGrid-LockedRoom-v0" \
    --train_num_timesteps 2000000 \
    --train_env_name "MiniGrid-LockedRoom-v0" \
    --train_max_mission_words 20 \
    --train_script_file "bash_scripts/lockedroom/z3_run_llm_minigrid_train.sh"