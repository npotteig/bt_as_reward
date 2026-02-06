#!/bin/bash
# Usage: ./verify_action_masks.sh

uv run bt_as_reward/scripts/verify_llm_action_masks.py \
    --expert_action_mask_file "action_masks/lockedroom_action_masks_expert.json" \
    --llm_action_mask_file "action_masks/lockedroom_action_masks_llm.json" \
    --output_file "llm_functions/lockedroom/response_action_masks.md"