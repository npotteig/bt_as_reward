#!/bin/bash


for seed in 1 2 3 4
do
    bash bash_scripts/doorkey/train.sh "bt_as_reward" $seed "false" "action_masks/doorkey_action_masks_llm.json" "dependent_subtasks/doorkey_dependent_subtasks.json" > bt_as_reward_mask_dependent.log
    bash bash_scripts/doorkey/train.sh "bt_as_reward" $seed "true" "action_masks/doorkey_action_masks_llm.json" "dependent_subtasks/doorkey_dependent_subtasks.json" > bt_as_reward_mask_dependent.log
done