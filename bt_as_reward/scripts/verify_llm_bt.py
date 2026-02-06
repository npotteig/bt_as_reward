import argparse
import json
from bt_as_reward.verifiers.verifier import BTVerifyOutput, CompositionZ3Output
from bt_as_reward.verifiers.minigrid_verifier import MiniGridBTRewardVerifier, MiniGridCompositionZ3Verifier
from bt_as_reward.verifiers.mujoco_verifier import MuJoCoBTRewardVerifier, MuJoCoCompositionZ3Verifier
from bt_as_reward.rewards.bt import BehaviourTreeConfig
from bt_as_reward.utils import (
    minigrid_create_distance_function,
    minigrid_object_to_str,
    mujoco_create_distance_function,
    mujoco_object_to_str,
    load_functions_from_file,
)
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify composition of LLM functions."
    )
    parser.add_argument(
        "--object_function_file",
        type=str,
        required=False,
        help="Path to the file containing the object detection function.",
    )
    parser.add_argument(
        "--object_function_names",
        type=str,
        required=False,
        help="Names of the object detection functions to verify.",
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
        required=False,
        help="Names of the subtask functions to verify.",
    )
    parser.add_argument(
        "--subtask_function_names",
        type=str,
        required=True,
        help="Names of the subtask function to verify.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=False,
        help="Name of the environment to verify in.",
    )
    parser.add_argument(
        "--use_z3",
        action="store_true",
        help="Whether to use Z3 for verification.",
    )
    parser.add_argument(
        "--expert_trajs",
        type=str,
        required=False,
        default=None,
        help="Path to the expert trajectories npz file.",
    )
    parser.add_argument(
        "--random_trajs",
        type=str,
        required=False,
        default=None,
        help="Path to the random trajectories npz file.",
    )
    parser.add_argument(
        "--dependent_subtask_file",
        type=str,
        required=False,
        default=None,
        help="Path to the file containing the dependent subtasks. If None, no dependent subtasks are used.",
    )
    parser.add_argument(
        "--max_reward",
        type=float,
        required=False,
        help="Maximum reward value.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the verification output.",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="MiniGrid",
        help="Name of the verifier to use.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=1.0,
        help="Distance threshold for proximity check.",
    )
    parser.add_argument(
        "--random_threshold",
        type=float,
        default=0.5,
        help="Threshold for random trajectories verification.",
    )
    args = parser.parse_args()

    subtask_functions = load_functions_from_file(
        args.subtask_function_file, args.subtask_function_names.strip().split(", ")
    )
    
    if not args.use_z3:
        object_functions = load_functions_from_file(
            args.object_function_file, args.object_function_names.strip().split(", ")
        )

        match args.verifier:
            case "MiniGrid":
                verifier = MiniGridBTRewardVerifier
                composition_verifier = MiniGridCompositionZ3Verifier
                create_distance_function = minigrid_create_distance_function
                object_to_str = minigrid_object_to_str
            case "MuJoCo":
                verifier = MuJoCoBTRewardVerifier
                create_distance_function = mujoco_create_distance_function
                object_to_str = mujoco_object_to_str
            # Add more verifiers as needed
            case _:
                print(f"Verifier {args.verifier} not recognized.")

        # Load expert and random trajectories
        expert_trajs = np.load(args.expert_trajs, allow_pickle=True)
        random_trajs = np.load(args.random_trajs, allow_pickle=True)

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
            object_to_str=object_to_str,
            create_distance_function=create_distance_function,
            dependent_subtasks=dependent_subtasks,
            distance_threshold=args.distance_threshold,
            use_memory=False,
        )

        verify_output: BTVerifyOutput = verifier.verify(
            expert_trajs=expert_trajs,
            random_trajs=random_trajs,
            max_reward=args.max_reward,
            bt_config=bt_config,
            random_threshold=args.random_threshold,
        )

        # Save the verification output
        with open(args.output_file, "w") as f:
            f.write("# Description:\n")
            f.write("This file contains the verification results for the Behaviour Tree.\n")
            f.write(
                f"## Expert Response:\n{verify_output.expert_response if verify_output.expert_response is not None else 'Success'}\n"
            )
            f.write(
                f"## Random Response:\n{verify_output.random_response if verify_output.random_response is not None else 'Success'}\n"
            )
    else:
        match args.verifier:
            case "MiniGrid":
                verifier = MiniGridCompositionZ3Verifier
            case "MuJoCo":
                verifier = MuJoCoCompositionZ3Verifier
            # Add more verifiers as needed
            case _:
                print(f"Verifier {args.verifier} not recognized.")
        
        match args.env_name:
            case "MiniGrid-DoorKey-6x6-v0":
                mission_args = {
                    "num_mission_keys": 1,
                    "num_mission_doors": 1,
                    "num_mission_boxes": 0,
                }
            case "MiniGrid-LockedRoom-v0":
                mission_args = {
                    "num_mission_keys": 1,
                    "num_mission_doors": 2,
                    "num_mission_boxes": 0,
                }
            case "DroneSupplier-v0":
                mission_args = {
                    "num_mission_keys": 1,
                    "num_mission_doors": 1,
                    "num_mission_boxes": 1,
                }
            case _:
                mission_args = {"num_mission_goals": 1}

        if args.expert_trajs:
            expert_trajs = np.load(args.expert_trajs, allow_pickle=True)
        verify_output: CompositionZ3Output = verifier.verify(
            subtask_functions=subtask_functions,
            env_name=args.env_name,
            mission_args=mission_args,
            expert_trajs=expert_trajs if args.expert_trajs else None,
        )

        # Save the verification output
        with open(args.output_file, "w") as f:
            f.write("# Description:\n")
            f.write(
                f"This file contains the Z3 verification results for the composition of subtask functions.\n"
            )
            f.write(
                f"## Response:\n{verify_output.positive_response if verify_output.positive_response is not None else 'Success'}\n"
            )
