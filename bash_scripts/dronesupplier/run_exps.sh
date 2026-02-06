#!/bin/bash

for seed in 1 2 3 4
do
    bash bash_scripts/dronesupplier/train.sh "environment" $seed "false" > environment.log
    bash bash_scripts/dronesupplier/train.sh "proc_as_reward" $seed "false" > proc_as_reward.log
    bash bash_scripts/dronesupplier/train.sh "bt_as_reward" $seed "false" > bt_as_reward.log
    bash bash_scripts/dronesupplier/train.sh "bt_as_reward" $seed "false" "action_masks/dronesupplier_action_masks_llm.json" > bt_as_reward_mask.log

    bash bash_scripts/dronesupplier/train.sh "environment" $seed "true" > environment.log
    bash bash_scripts/dronesupplier/train.sh "proc_as_reward" $seed "true" > proc_as_reward.log
    bash bash_scripts/dronesupplier/train.sh "bt_as_reward" $seed "true" > bt_as_reward.log
    bash bash_scripts/dronesupplier/train.sh "bt_as_reward" $seed "true" "action_masks/dronesupplier_action_masks_llm.json" > bt_as_reward_mask.log
done