import argparse
import importlib.util
from bt_as_reward.verifiers.verifier import VerifyOutput
from bt_as_reward.verifiers.doorkey import DoorKeyVerifier
import numpy as np


def load_function_from_file(file_path, function_name):
    """
    Dynamically load a function from a specified file path.

    :param file_path: Path to the Python file containing the function.
    :param function_name: Name of the function to load.
    :return: The loaded function or None if not found.
    """
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, function_name, None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verify LLM subtask functions.")
    parser.add_argument(
        "--function_file",
        type=str,
        required=True,
        help="Path to the file containing the subtask function.",
    )
    parser.add_argument(
        "--function_name",
        type=str,
        required=True,
        help="Name of the subtask function to verify.",
    )
    parser.add_argument(
        "--expert_trajs",
        type=str,
        required=True,
        help="Path to the expert trajectories npz file.",
    )
    parser.add_argument(
        "--random_trajs",
        type=str,
        required=True,
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
        default="DoorKeyVerifier",
        help="Name of the verifier class to use.",
    )
    parser.add_argument(
        "--random_threshold",
        type=float,
        default=0.5,
        help="Threshold for random trajectories verification.",
    )
    args = parser.parse_args()

    subtask_function = load_function_from_file(args.function_file, args.function_name)
    if subtask_function is None:
        print(f"Function {args.function_name} not found in {args.function_file}.")
        exit(1)

    match args.verifier:
        case "DoorKeyVerifier":
            verifier = DoorKeyVerifier()
        # Add more verifiers as needed
        case _:
            print(f"Verifier {args.verifier} not recognized.")

    # Load expert and random trajectories
    expert_trajs = np.load(args.expert_trajs, allow_pickle=True)
    random_trajs = np.load(args.random_trajs, allow_pickle=True)

    verify_output: VerifyOutput = verifier.verify(
        subtask_function=subtask_function,
        expert_trajs=expert_trajs,
        random_trajs=random_trajs,
        random_threshold=args.random_threshold,
    )

    # Save the verification output
    with open(args.output_file, "w") as f:
        f.write("# Description:\n")
        f.write(
            f"This file contains the verification results for the subtask function {args.function_name}.\n"
        )
        f.write(
            "Use only one prompt from the file when re-prompting, selecting it based on the given priority order: expert non-reactive, then random, then expert reactive.\n"
        )
        f.write(
            f"## Expert Non-Reactive Response:\n{verify_output.expert_nonreactive_response}\n"
        )
        f.write(f"## Random Response:\n{verify_output.random_response}\n")
        f.write(
            f"## Expert Reactive Response:\n{verify_output.expert_reactive_response}\n"
        )
