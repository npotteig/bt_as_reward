#!/bin/bash

for seed in 1 2 3 4
do
    bash bash_scripts/pickandplace2/z3_train.sh "bt_as_reward" $seed "false" "action_masks/pickandplace2_action_masks_llm.json" > bt_as_reward_mask.log

    bash bash_scripts/pickandplace2/z3_train.sh "bt_as_reward" $seed "true" "action_masks/pickandplace2_action_masks_llm.json" > bt_as_reward_mask.log
done
