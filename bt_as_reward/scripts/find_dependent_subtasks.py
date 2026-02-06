"""
Find future subtasks that when failure -> success cause the success -> failure of a current subtask.
"""

import argparse
import os
import json
import numpy as np

from bt_as_reward.utils import (
    load_functions_from_file,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find dependent subtasks.")
    parser.add_argument(
        "--subtask_function_file",
        type=str,
        required=True,
        help="Path to the file containing the subtask function.",
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
        help="Path to save the dependent subtasks output.",
    )
    args = parser.parse_args()

    subtask_functions = load_functions_from_file(
        args.subtask_function_file, args.subtask_function_names.strip().split(", ")
    )

    # Load expert trajectories
    expert_trajs = np.load(args.expert_trajs, allow_pickle=True)

    dependent_subtasks = {k.__name__: set() for k in subtask_functions}
    subtask_status = {k.__name__: False for k in subtask_functions}
    for episode in expert_trajs["episodes"]:
        for state in episode["states"]:
            current_status = {
                k.__name__: k(state, state["mission"]) for k in subtask_functions
            }
            for k in subtask_status:
                # if subtask switches from True to False, check which subtasks switched from False to True
                if subtask_status[k] and not current_status[k]:
                    for k2 in subtask_status:
                        if not subtask_status[k2] and current_status[k2]:
                            dependent_subtasks[k].add(k2)
            subtask_status = current_status

    if not os.path.exists("dependent_subtasks/"):
        os.makedirs("dependent_subtasks/", exist_ok=True)
        print("Directory 'dependent_subtasks/' created.")

    # Save to output file
    with open(f"dependent_subtasks/{args.output_file}.json", "w") as f:
        json.dump({k: list(v) for k, v in dependent_subtasks.items()}, f, indent=4)
