# Reward Shaping and Action Masking for Compositional Tasks using Behavior Trees and LLMs

## Setup
We test on a machine with Ubuntu 22.04 OS and a NVIDIA GTX 3070.

Requires Python >=3.11
```bash
pip install uv && uv sync
```

## Environments

- MiniGrid
- MuJoCo Fetch

## Mission Spaces

- DoorKey: *doorkey*
- LockedRoom: *lockedroom*
- DroneSupplier: *dronesupplier*
- PickAndPlace: *pickandplace*
- PickAndPlace2: *pickandplace2*

## Prompts

Prompts are located in `llm_prompts/`. System prompts are found in `llm_prompts/minigrid` and `llm_prompts/mujoco` under the name of the mission space (e.g. `llm_prompts/minigrid/z3_doorkey.md`). Prompts for each environment are found at `llm_prompts/minigrid/prompts.md` and `llm_prompts/mujoco/prompts.md`. The only difference between the two prompts files is the parameter types for the subtask function prototypes.

## Repeatability

### Reward Functions

- Task: *environment*
- Procedure: *proc_as_reward*
- RBT: *bt_as_reward*
- MRBT: *bt_as_reward* + *action_mask_file*

### Run the full prompt-test pipeline
Automatic MRBT generation pipeline. Requires access to Vanderbilt Amplify.

**Add environment variables**
```bash
export AMPLIFY_BASE_URL=https://prod-api.vanderbilt.ai
export AMPLIFY_API_KEY=<AMPLIFY_KEY>
```

**Run the full pipeline**
```bash
bash bash_scripts/doorkey/z3_run_llm_pipeline.sh
bash bash_scripts/{mission_space}/z3_run_llm_pipeline.sh
```

Generates a chat history at `llm_functions/doorkey/z3_doorkey_chat_history.json` and subtask code at `llm_functions/doorkey/z3_doorkey_code.py`. Action masks are generated at `action_masks/doorkey_action_masks_llm.json`. Finally, an executable training script is created to train an agent using the MRBT at `bash_scripts/doorkey/z3_run_llm_minigrid_train.sh`. Run the training script as follows:

```
bash bash_scripts/doorkey/z3_run_llm_minigrid_train.sh <reward_mode> <seed> <drop_key> <action_mask_file>
```

Parameter values can be found in `bash_scripts/doorkey/z3_run_exps.sh`.


### Functional Outputs

Uses pre-generated LLM Functions using the pipeline with GPT-5.

F1: Verification
F2: Evaluation for Ablation Results
F3: Evaluation of policy transfer to Microsoft AirSim

F2 and F3 may produce slightly different results than paper figures due to stochasticity in environment and policy training.


### F1: Verify Subtask Z3 Code

**Subtask Completion Verification Case**
```bash
bash bash_scripts/doorkey/z3_verify_subtask.sh subtask_1_complete
bash bash_scripts/{task_space}/z3_verify_subtask.sh subtask_{num}_complete
```

**Subtask Object Proximity Verification Case**
```bash
bash bash_scripts/doorkey/z3_verify_object.sh subtask_1_object subtask_1_complete
bash bash_scripts/{task_space}/z3_verify_object.sh subtask_{num}_object subtask_{num}_complete
```

**Non-regressive Maximal Reward Verification Case**
```bash
bash bash_scripts/doorkey/z3_verify_composition.sh
bash bash_scripts/{task_space}/z3_verify_composition.sh
```

**Generate Expert Action Masks**
```bash
bash bash_scripts/doorkey/create_action_masks.sh
bash bash_scripts/{task_space}/create_action_masks.sh
```

**Reprompt Subtask Object Proximity Failure**
```bash
bash bash_scripts/pickandplace2/z3_verify_object.sh subtask_2_object_failure subtask_2_complete
```

**Reprompt Composition Failure**
```bash
bash bash_scripts/lockedroom/failure_z3_verify_bt_composition.sh
```

**Generate Expert and Random Trajectories**
```bash
bash bash_scripts/doorkey/create_trajs.sh
bash bash_scripts/{mission_space}/create_trajs.sh
```

You can use the trajectories for testing the specs above by including an extra argument
```
bash bash_scripts/doorkey/z3_verify_subtask.sh subtask_1_complete true
```
where true indicates to use expert and random trajectories.


### F2: Ablation Study Results
Runs training runs with 4 random seeds in both deterministic and stochastic dynamics. 32 training runs in total. For MiniGrid, stochastic implies when the key is held, it has a 5% chance to drop. For MuJoCo Fetch, stochastic implies when the gripper is closed, it has a 5% chance to open. Results described in Section 6.5. F2 is associated with Figures 4 and 5 in the paper.

**Training**
```bash
bash bash_scripts/doorkey/z3_run_exps.sh
bash bash_scripts/{task_space}/z3_run_exps.sh
```

**Plotting**
```bash
bash bash_scripts/plot_tensorboard.sh MiniGrid-DoorKey-16x16-v0 false
bash bash_scripts/plot_tensorboard.sh MiniGrid-DoorKey-16x16-v0 true
bash bash_scripts/plot_tensorboard.sh {env_name} {is_stochastic}
```

### F4: Policy Transfer to Microsoft AirSim
Evaluation of policy learned in DroneSupplier deployed to Microsoft AirSim neighborhood environment. Results discussed in Section 6.6.2. Appropriate hardware and GPU needed to run this experiment. A policy must have been learned already in DroneSupplier for this experiment to work.

Map of DroneSupplier mission in `figs/`. Walls (excluding the border) are obstacles in a 25 X 25 section in the neighborhood. 

**Download AirSimNH**

Go to https://github.com/microsoft/AirSim/releases and download AirSimNH.zip for Linux or Windows.


**Run AirSimNH**

Unzip
```bash
unzip AirSimNH.zip
```

Run environment in headless mode

Linux
```bash
bash AirSimNH/LinuxNoEditor/AirSimNH.sh -nullrhi
```

Windows
```bash
bash AirSimNH/WindowsNoEditor/AirSimNH.exe -nullrhi
```

**Run Policy**

In a separate terminal window from this project's root directory run:
```bash
bash drone/test.sh results/checkpoints/DroneSupplier-v0_bt_as_reward_mask/ppo_cnn_2000000_steps.zip "bt_as_reward" false "action_masks/dronesupplier_action_masks_llm.json"
```
```bash
bash drone/test.sh results/checkpoints/DroneSupplier-v0_bt_as_reward_mask_uncertainty/ppo_cnn_2000000_steps.zip "bt_as_reward" true "action_masks/dronesupplier_action_masks_llm.json"
```