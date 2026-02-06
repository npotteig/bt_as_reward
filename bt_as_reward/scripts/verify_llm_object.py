import argparse
from bt_as_reward.verifiers.verifier import ObjectVerifyOutput, ObjectZ3Output
from bt_as_reward.verifiers.minigrid_verifier import MiniGridObjectVerifier, MiniGridObjectZ3Verifier
from bt_as_reward.verifiers.mujoco_verifier import MuJoCoObjectVerifier, MuJoCoObjectZ3Verifier
from bt_as_reward.utils import load_functions_from_file
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify LLM object detection functions."
    )
    parser.add_argument(
        "--object_function_file",
        type=str,
        required=True,
        help="Path to the file containing the object detection function.",
    )
    parser.add_argument(
        "--object_function_name",
        type=str,
        required=True,
        help="Name of the object detection function to verify.",
    )
    parser.add_argument(
        "--subtask_function_file",
        type=str,
        required=True,
        help="Path to the file containing the subtask function.",
    )
    parser.add_argument(
        "--subtask_function_name",
        type=str,
        required=True,
        help="Name of the subtask function to verify.",
    )
    parser.add_argument(
        "--use_z3",
        action="store_true",
        help="Whether to use Z3 for verification.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        required=False,
        help="Name of the environment to verify in.",
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
    args = parser.parse_args()

    object_function = load_functions_from_file(
        args.object_function_file, [args.object_function_name]
    )[0]
    subtask_function = load_functions_from_file(
        args.subtask_function_file, [args.subtask_function_name]
    )[0]

    if not args.use_z3:
        match args.verifier:
            case "MiniGrid":
                verifier = MiniGridObjectVerifier
            case "MuJoCo":
                verifier = MuJoCoObjectVerifier
            # Add more verifiers as needed
            case _:
                print(f"Verifier {args.verifier} not recognized.")

        # Load expert and random trajectories
        expert_trajs = np.load(args.expert_trajs, allow_pickle=True)

        verify_output: ObjectVerifyOutput = verifier.verify(
            expert_trajs=expert_trajs,
            object_function=object_function,
            subtask_function=subtask_function,
            distance_threshold=args.distance_threshold,
        )

        # Save the verification output
        with open(args.output_file, "w") as f:
            f.write("# Description:\n")
            f.write(
                f"This file contains the verification results for the object detection function {args.object_function_name}.\n"
            )
            f.write(
                f"## Response:\n{verify_output.response if verify_output.response is not None else 'Success'}\n"
            )
    else:
        match args.verifier:
            case "MiniGrid":
                verifier = MiniGridObjectZ3Verifier
            case "MuJoCo":
                verifier = MuJoCoObjectZ3Verifier
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
            case _: # MuJoCo
                mission_args = {
                    "num_mission_goals": 1
                }
        if args.expert_trajs:
            expert_trajs = np.load(args.expert_trajs, allow_pickle=True)
        if args.random_trajs:
            random_trajs = np.load(args.random_trajs, allow_pickle=True)
        verify_output: ObjectZ3Output = verifier.verify(
            env_name=args.env_name,
            object_function=object_function,
            subtask_function=subtask_function,
            mission_args=mission_args,
            random_trajs=random_trajs if args.random_trajs else None,
            expert_trajs=expert_trajs if args.expert_trajs else None,
        )

        # Save the verification output
        with open(args.output_file, "w") as f:
            f.write("# Description:\n")
            f.write(
                f"This file contains the Z3 verification results for the object detection function {args.object_function_name}.\n"
            )
            f.write(
                f"## Positive Check Response:\n{verify_output.positive_response if verify_output.positive_response is not None else 'Success'}\n"
            )
            f.write(
                f"## Negative Check Response:\n{verify_output.negative_response if verify_output.negative_response is not None else 'Success'}\n"
            )
