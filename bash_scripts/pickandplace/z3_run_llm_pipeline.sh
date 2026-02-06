#!/bin/bash
# Usage: ./z3_run_llm_pipeline.sh

uv run python bt_as_reward/scripts/z3_full_llm_pipeline.py \
    --system_prompt_file "llm_prompts/mujoco/pickandplace.md" \
    --user_prompt_file "llm_prompts/mujoco/prompts.md" \
    --chat_history_ckpt "llm_functions/pickandplace/z3_pickandplace_chat_history.json" \
    --chat_history_output_file "llm_functions/pickandplace/z3_pickandplace_chat_history.json" \
    --code_file "llm_functions/pickandplace/z3_pickandplace_code.py" \
    --verifier MuJoCo \
    --llm_action_mask_file "pickandplace_action_masks_llm" \
    --verify_env_name "FetchPickAndPlace-v5" \
    --train_num_timesteps 5000000 \
    --train_env_name "FetchPickAndPlace-v5" \
    --train_max_mission_words 12 \
    --train_script_file "bash_scripts/pickandplace/z3_run_llm_mujoco_train.sh"