import argparse
import json
import os
import numpy as np
from tqdm import tqdm

from bt_as_reward.rewards.bt import BehaviourTreeConfig, BehaviourTreeReward
from bt_as_reward.utils import (
    mujoco_create_distance_function,
    mujoco_object_to_str,
    load_functions_from_file,
    mujoco_state_mission_to_z3
)

MUJOCO_NUM_ACTIONS = 8  # MuJoCo Discrete has 8 discrete actions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create action masks for mujoco subtasks using expert trajectories."
    )
    parser.add_argument(
        "--object_function_file",
        type=str,
        required=True,
        help="Path to the file containing the object detection function.",
    )
    parser.add_argument(
        "--object_function_names",
        type=str,
        required=True,
        help="Names of the object detection functions to verify.",
    )
    parser.add_argument(
        "--dependent_subtask_file",
        type=str,
        default=None,
        help="Path to the file containing the dependent subtasks. If None, no dependent subtasks are used.",
    )
    parser.add_argument(
        "--subtask_function_file",
        type=str,
        required=True,
        help="Path to the file containing the subtask function.",
    )
    parser.add_argument(
        "--subtask_names",
        type=str,
        required=True,
        help="Names of the subtask functions to verify.",
    )
    parser.add_argument(
        "--subtask_function_names",
        type=str,
        required=True,
        help="Names of the subtask function to verify.",
    )
    parser.add_argument(
        "--expert_trajs",
        type=str,
        required=True,
        help="Path to the expert trajectories npz file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Filename action mask dictionary.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=1.0,
        help="Distance threshold for proximity check.",
    )
    args = parser.parse_args()

    object_functions = load_functions_from_file(
        args.object_function_file, args.object_function_names.strip().split(", ")
    )
    subtask_functions = load_functions_from_file(
        args.subtask_function_file, args.subtask_function_names.strip().split(", ")
    )

    # Load expert trajectories
    expert_trajs = np.load(args.expert_trajs, allow_pickle=True)

    dependent_subtasks = None
    if args.dependent_subtask_file is not None:
        with open(args.dependent_subtask_file, "r") as f:
            dependent_subtasks = json.load(f)
        # Convert keys and values back to functions
        dependent_subtasks = {
            next(filter(lambda x: x.__name__ == k, subtask_functions)): [
                next(filter(lambda x: x.__name__ == v, subtask_functions)) for v in vs
            ]
            for k, vs in dependent_subtasks.items()
        }

    bt_config = BehaviourTreeConfig(
        subtask_names=args.subtask_names.strip().split(", "),
        subtask_functions=subtask_functions,
        object_functions=object_functions,
        object_to_str=mujoco_object_to_str,
        create_distance_function=mujoco_create_distance_function,
        dependent_subtasks=dependent_subtasks,
        distance_threshold=args.distance_threshold,
        use_memory=True,
    )

    actions_per_subtask = [
        [] for _ in range(len(object_functions) + len(subtask_functions))
    ]

    for episode in tqdm(expert_trajs["episodes"]):
        mission_str = episode["states"][0]["mission"]
        bt = BehaviourTreeReward.create_bt(mission_str=mission_str, bt_config=bt_config, state_mission_to_z3=mujoco_state_mission_to_z3)
        idx = 0
        for state, action in zip(episode["states"][:-1], episode["actions"]):
            reward, _ = bt.step_reward(state, state["mission"])
            if reward > 0:
                idx += int(reward / 0.5)
            idx = min(idx, len(actions_per_subtask) - 1)
            actions_per_subtask[idx].append(action)

    function_keys = []
    for object_fn, subtask_fn in zip(object_functions, subtask_functions):
        function_keys.extend(
            [f"goto_{object_fn.__name__}", f"interact_{object_fn.__name__}"]
        )

    action_mask_dict = {
        k: np.zeros(MUJOCO_NUM_ACTIONS, dtype=int) for k in function_keys
    }

    if not os.path.exists("action_masks/"):
        os.makedirs("action_masks/", exist_ok=True)
        print("Directory 'action_masks/' created.")

    for fn_name, actions in zip(function_keys, actions_per_subtask):
        action_mask_dict[fn_name][list(set(actions))] = 1

    with open(f"action_masks/{args.output_file}.json", "w") as f:
        json.dump({k: v.tolist() for k, v in action_mask_dict.items()}, f)
