import argparse
import json
import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify LLM Action Mask Format is Consistent with Expert Action Mask."
    )
    parser.add_argument(
        "--expert_action_mask_file",
        type=str,
        required=True,
        help="Path to the expert action mask JSON file.",
    )
    parser.add_argument(
        "--llm_action_mask_file",
        type=str,
        required=True,
        help="Path to the LLM action mask JSON file.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to save the verification output.",
    )
    args = parser.parse_args()

    with open(args.expert_action_mask_file, "r") as f:
        expert_action_masks = json.load(f)
    with open(args.llm_action_mask_file, "r") as f:
        llm_action_masks = json.load(f)

    response = ""

    if set(expert_action_masks.keys()) != set(llm_action_masks.keys()):
        response += "Mismatch in object keys between expert and LLM action masks.\n"
    else:
        for obj in expert_action_masks:
            expert_masks = expert_action_masks[obj]
            llm_masks = llm_action_masks[obj]
            if len(expert_masks) != len(llm_masks):
                response += f"Mismatch in length of action mask for key '{obj}'.\n"
            if np.sum(expert_masks) > np.sum(llm_masks):
                response += f"LLM action mask for key '{obj}' has fewer positive entries than expert action mask.\n"

    with open(args.output_file, "w") as f:
        f.write("# Description:\n")
        f.write("This file contains the verification results for the action masks.\n")
        if response == "":
            f.write(
                "## Response:\nAll action masks are consistent with expert action masks."
            )
        else:
            f.write(f"## Response:\n{response}")
