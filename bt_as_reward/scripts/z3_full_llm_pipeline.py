import argparse
import os
import re
import json
import dotenv
dotenv.load_dotenv()
import numpy as np
from tqdm import tqdm
import jinja2
import traceback

import bt_as_reward.utils as utils
from bt_as_reward.verifiers.verifier import SubtaskZ3Output
from bt_as_reward.verifiers.minigrid_verifier import MiniGridSubtaskZ3Verifier
from bt_as_reward.verifiers.mujoco_verifier import MuJoCoBTRewardVerifier, MuJoCoObjectVerifier, MuJoCoSubtaskVerifier, MuJoCoSubtaskZ3Verifier
from bt_as_reward.verifiers.verifier import ObjectZ3Output
from bt_as_reward.verifiers.minigrid_verifier import MiniGridObjectZ3Verifier
from bt_as_reward.verifiers.mujoco_verifier import MuJoCoObjectZ3Verifier
from bt_as_reward.verifiers.verifier import CompositionZ3Output
from bt_as_reward.verifiers.minigrid_verifier import MiniGridCompositionZ3Verifier
from bt_as_reward.verifiers.mujoco_verifier import MuJoCoCompositionZ3Verifier
from bt_as_reward.rewards.bt import BehaviourTreeConfig, BehaviourTreeReward
from bt_as_reward.utils import load_functions_from_file, minigrid_state_mission_to_z3, mujoco_state_mission_to_z3

MODEL="gpt-5"
REASONING_LEVEL="medium"
MAX_RETRIES=5

MINIGRID_NUM_ACTIONS = 7  # Minigrid has 7 discrete actions
MUJOCO_NUM_ACTIONS = 8  # MuJoCo Discrete has 8 discrete actions

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full LLM Pipeline Execution Script"
    )
    parser.add_argument(
        "--system_prompt_file",
        type=str,
        required=True,
        help="Path to the system prompt file.",
    )
    parser.add_argument(
        "--user_prompt_file",
        type=str,
        required=True,
        help="Path to the user prompt file.",
    )
    parser.add_argument(
        "--chat_history_ckpt",
        type=str,
        default=None,
        help="Path to a checkpoint file to resume chat history from.",
    )
    parser.add_argument(
        "--chat_history_output_file",
        type=str,
        required=True,
        help="Path to save the verification output.",
    )
    parser.add_argument(
        "--code_file",
        type=str,
        required=True,
        help="Path to save the generated code.",
    )
    parser.add_argument(
        "--verifier",
        type=str,
        default="MiniGrid",
        help="Name of the verifier to use.",
    )
    parser.add_argument(
        "--llm_action_mask_file",
        type=str,
        required=True,
        help="Filename llm action mask dictionary.",
    )
    parser.add_argument(
        "--verify_env_name",
        type=str,
        required=True,
        help="Name of the environment for Z3 verification.",
    )
    parser.add_argument(
        "--expert_trajs",
        type=str,
        default=None,
        required=False,
        help="Path to the expert trajectories npz file.",
    )
    parser.add_argument(
        "--random_trajs",
        type=str,
        default=None,
        required=False,
        help="Path to the random trajectories npz file.",
    )
    parser.add_argument(
        "--train_num_timesteps",
        type=int,
        default=1000000,
        help="Number of timesteps for training script.",
    )
    parser.add_argument(
        "--train_env_name",
        type=str,
        default="MiniGrid-DoorKey-16x16-v0",
        help="Environment name for training script.",
    )
    parser.add_argument(
        "--train_max_mission_words",
        type=int,
        default=15,
        help="Maximum number of words in mission for training script.",
    )
    parser.add_argument(
        "--train_script_file",
        type=str,
        required=True,
        help="Path to save the generated training script.",
    )
    args = parser.parse_args()
    
    with open(args.system_prompt_file, 'r') as f:
        system_prompt = {
            "role": "system",
            "content": f.read()
        }
    
    with open(args.user_prompt_file, 'r') as f:
        user_prompts = f.read()
    
    user_prompts = re.split(r"#### \w+ \d+", user_prompts)
    user_prompts = [prompt.strip() for prompt in user_prompts if prompt.strip()]
    
    if args.chat_history_ckpt and os.path.exists(args.chat_history_ckpt):
        with open(args.chat_history_ckpt, 'r') as f:
            chat_history = json.load(f)
    else:   
        chat_history_meta = {"stage": 1}
        chat_history = [chat_history_meta, system_prompt]
        
    if args.expert_trajs:
        expert_trajs = np.load(args.expert_trajs, allow_pickle=True)
    if args.random_trajs:
        random_trajs = np.load(args.random_trajs, allow_pickle=True)
        
        
    match args.verifier:
        case "MiniGrid":
            completion_verifier = MiniGridSubtaskZ3Verifier
            object_verifier = MiniGridObjectZ3Verifier
            composition_verifier = MiniGridCompositionZ3Verifier
            state_mission_to_z3 = minigrid_state_mission_to_z3
        case "MuJoCo":
            completion_verifier = MuJoCoSubtaskZ3Verifier
            object_verifier = MuJoCoObjectZ3Verifier
            composition_verifier = MuJoCoCompositionZ3Verifier
            state_mission_to_z3 = mujoco_state_mission_to_z3
        # Add more verifiers as needed
        case _:
            print(f"Verifier {args.verifier} not recognized.")
    
    match args.verify_env_name:
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
    
    # Stage 1: Generate Subtasks
    if chat_history[0]["stage"] <= 1:
        chat_history.append({
            "role": "user",
            "content": user_prompts[0]
        })
        user_accepted = False
        while not user_accepted:
            response = utils.amplify_make_request(
                base_url=os.getenv("AMPLIFY_BASE_URL"),
                api_key=os.getenv("AMPLIFY_API_KEY"),
                messages=chat_history[1:],
                ai_model=MODEL,
                reasoningLevel=REASONING_LEVEL
            )
            print("LLM Response:")
            print(response)
            # Check if json markdown block can be extracted
            md_code_match = re.search(r"```json(.*?)```", response, re.DOTALL)
            if md_code_match:
                json_str = md_code_match.group(1).strip()
                try:
                    response_json = json.loads(json_str)
                    print("Extracted JSON:")
                    print(json.dumps(response_json, indent=4))
                    user_input = input("Do you accept this response? (y/n): ")
                    if user_input.lower() == 'y':
                        user_accepted = True
                except json.JSONDecodeError:
                    chat_history.extend([{
                        "role": "assistant",
                        "content": response
                    },
                    {
                        "role": "user",
                        "content": "Failed to decode JSON from the response. Please review the response above."
                    }])
                    print("Failed to decode JSON from the response. Please review the response above.")
        if response:
            chat_history.append({
                "role": "assistant",
                "content": response
            })
        chat_history[0]["stage"] += 1  # Update stage to 2
        chat_history[0]["subtask_idx"] = len(chat_history) - 1  # Index of last response
    
        # save chat history to a file
        with open(args.chat_history_output_file, 'w') as f:
            json.dump(chat_history, f, indent=4)
      
    subtasks = re.search(r"```json(.*?)```", chat_history[chat_history[0]["subtask_idx"]]["content"], re.DOTALL)
    subtasks = json.loads(subtasks.group(1).strip())  
    print(f"Current Stage: {chat_history[0]['stage']}")   
    # Stage 2+: Develop Code for Subtasks
    if chat_history[0]["stage"] <= (len(subtasks)*2) + 1:
        
        for i, subtask in enumerate(subtasks):
            if ((i + 1) * 2) + 1 < chat_history[0]["stage"]:
                continue  # Skip already completed subtasks
            
            print(f"Generating code for subtask {i+1}/{len(subtasks)}: {subtask}")
            
            if chat_history[0]["stage"] == (i + 1) * 2:
                accept_response = False
                subtask_complete_code_string = ""
                tries = 0
                chat_history.append({"role": "user", "content": user_prompts[1].replace("1", str(i+1))})
                print(f"Generating Subtask {i+1} Completion Z3 Code...")
                while not accept_response and tries < MAX_RETRIES:
                    response = utils.amplify_make_request(
                        base_url=os.getenv("AMPLIFY_BASE_URL"),
                        api_key=os.getenv("AMPLIFY_API_KEY"),
                        messages=chat_history[1:],
                        ai_model=MODEL,
                        reasoningLevel=REASONING_LEVEL
                    )
                    chat_history.append({
                        "role": "assistant",
                        "content": response
                    })
                    # Check if python code can be extracted
                    md_code_match = re.search(r"```Python(.*?)```", response, re.DOTALL)
                    if md_code_match:
                        subtask_complete_code_string = md_code_match.group(1).strip()
                    else:
                        accept_response = False
                        chat_history.append({
                            "role": "user",
                            "content": "Failed to extract python code from the response. Please review the response above."
                        })
                        print(f"Failed to extract python code from the response. Retrying... ({tries+1}/{MAX_RETRIES})")
                        tries += 1
                        continue
                    
                    # Load function
                    subtask_complete_function = utils.load_function_from_str(
                        subtask_complete_code_string,
                        f"subtask_{i+1}_complete"
                    )
                    if subtask_complete_function is None:
                        accept_response = False
                        chat_history.append({
                            "role": "user",
                            "content": "Failed to load the subtask completion function from the code. Please review the response and check the function signature."
                        })
                        print(f"Failed to load the subtask completion function from the code. Retrying... ({tries+1}/{MAX_RETRIES})")
                        tries += 1
                        continue
                    
                    try:
                        verify_output: SubtaskZ3Output = completion_verifier.verify(
                            subtask_function=subtask_complete_function,
                            env_name=args.verify_env_name,
                            mission_args=mission_args,
                            random_trajs=random_trajs if args.random_trajs else None,
                            expert_trajs=expert_trajs if args.expert_trajs else None,
                        )
                    except Exception as e:
                        tb_str = traceback.format_exc()
                        accept_response = False
                        chat_history.append({
                            "role": "user",
                            "content": f"Error during verification: {str(e)}\nTraceback:\n{tb_str}\nPlease review the error and fix the code accordingly."
                        })
                        print(f"Error during verification: {str(e)}. Retrying... ({tries+1}/{MAX_RETRIES})")
                        tries += 1
                        continue
                    
                    if verify_output.positive_response is not None:
                        accept_response = False
                        chat_history.append({
                            "role": "user",
                            "content": verify_output.positive_response
                        })
                        print(f"Completion Verifier returned failure for Correctness. Retrying... ({tries+1}/{MAX_RETRIES})")
                        tries += 1
                        continue
                    
                    if verify_output.negative_response is not None:
                        accept_response = False
                        chat_history.append({
                            "role": "user",
                            "content": verify_output.negative_response
                        })
                        print(f"Completion Verifier returned failure for Non-Triviality. Retrying... ({tries+1}/{MAX_RETRIES})")
                        tries += 1
                        continue
                    
                    accept_response = True
                    # Append generated code to code file
                    with open(args.code_file, 'a') as code_f:
                        code_f.write(subtask_complete_code_string + "\n\n")
                    
                    chat_history[0]["stage"] += 1  # Update stage
    
                    # save chat history to a file
                    with open(args.chat_history_output_file, 'w') as f:
                        json.dump(chat_history, f, indent=4)
                
                if tries == MAX_RETRIES:
                    raise RuntimeError(f"Failed to generate valid code for subtask {i+1} after {MAX_RETRIES} attempts.")
            
            # Object Detection Function
            accept_response = False
            subtask_object_code_string = ""
            tries = 0
            chat_history.append({"role": "user", "content": re.sub(r'(?<!-)1', str(i+1), user_prompts[2])})
            print(f"Generating Subtask {i+1} Object Proximity Z3 Code...")
            while not accept_response and tries < MAX_RETRIES:
                response = utils.amplify_make_request(
                        base_url=os.getenv("AMPLIFY_BASE_URL"),
                        api_key=os.getenv("AMPLIFY_API_KEY"),
                        messages=chat_history[1:],
                        ai_model=MODEL,
                        reasoningLevel=REASONING_LEVEL
                )
                chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                # Check if python code can be extracted
                md_code_match = re.search(r"```Python(.*?)```", response, re.DOTALL)
                if md_code_match:
                    subtask_object_code_string = md_code_match.group(1).strip()
                else:
                    accept_response = False
                    chat_history.append({
                        "role": "user",
                        "content": "Failed to extract python code from the response. Please review the response above."
                    })
                    print(f"Failed to extract python code from the response. Retrying... ({tries+1}/{MAX_RETRIES})")
                    tries += 1
                    continue
                
                # Load function
                subtask_object_function = utils.load_function_from_str(
                    subtask_object_code_string,
                    f"subtask_{i+1}_object"
                )
                if subtask_object_function is None:
                    accept_response = False
                    chat_history.append({
                        "role": "user",
                        "content": "Failed to load the subtask object function from the code. Please review the response and check the function signature."
                    })
                    print(f"Failed to load the subtask object function from the code. Retrying... ({tries+1}/{MAX_RETRIES})")
                    tries += 1
                    continue
                
                subtask_complete_function = load_functions_from_file(
                    args.code_file, [f"subtask_{i+1}_complete"]
                )[0]

                try:
                    verify_output: ObjectZ3Output = object_verifier.verify(
                        env_name=args.verify_env_name,
                        mission_args=mission_args,
                        object_function=subtask_object_function,
                        subtask_function=subtask_complete_function,
                        random_trajs=random_trajs if args.random_trajs else None,
                        expert_trajs=expert_trajs if args.expert_trajs else None,
                    )
                except Exception as e:
                    tb_str = traceback.format_exc()
                    accept_response = False
                    chat_history.append({
                        "role": "user",
                        "content": f"Error during verification: {str(e)}\nTraceback:\n{tb_str}\nPlease review the error and fix the code accordingly."
                    })
                    print(f"Error during verification: {str(e)}. Retrying... ({tries+1}/{MAX_RETRIES})")
                    tries += 1
                    continue
                
                if verify_output.positive_response is not None:
                    accept_response = False
                    chat_history.append({
                        "role": "user",
                        "content": verify_output.positive_response
                    })
                    print(f"Object Proximity Verifier returned failure for Correctness. Retrying... ({tries+1}/{MAX_RETRIES})")
                    tries += 1
                    continue
                    
                if verify_output.negative_response is not None:
                    accept_response = False
                    chat_history.append({
                        "role": "user",
                        "content": verify_output.negative_response
                    })
                    print(f"Object Proximity Verifier returned failure for Non-Triviality. Retrying... ({tries+1}/{MAX_RETRIES})")
                    tries += 1
                    continue
                
                accept_response = True
                # Append generated code to code file
                with open(args.code_file, 'a') as code_f:
                    code_f.write(subtask_object_code_string + "\n\n")
                
                chat_history[0]["stage"] += 1  # Update stage

                # save chat history to a file
                with open(args.chat_history_output_file, 'w') as f:
                    json.dump(chat_history, f, indent=4)
                
                
    # Stage Composition: Compose Functions into BT Reward
    print("Composing Subtask Functions to check Non-regressive Maximal Reward...")
    if chat_history[0]["stage"] == (len(subtasks) * 2) + 2:
        accept_response = False
        tries = 0
        while not accept_response and tries < MAX_RETRIES:
            subtask_functions = load_functions_from_file(
                args.code_file, [f"subtask_{i+1}_complete" for i in range(len(subtasks))]
            )
            
            try:
                verify_output: CompositionZ3Output = composition_verifier.verify(
                    env_name=args.verify_env_name,
                    mission_args=mission_args,
                    subtask_functions=subtask_functions,
                    expert_trajs=expert_trajs if args.expert_trajs else None,
                )
            except Exception as e:
                tb_str = traceback.format_exc()
                accept_response = False
                chat_history.append({
                    "role": "user",
                    "content": f"Error during verification: {str(e)}\nTraceback:\n{tb_str}\nPlease review the error and fix the code accordingly."
                })
                print(f"Error during verification: {str(e)}. Retrying... ({tries+1}/{MAX_RETRIES})")
                tries += 1
                continue
            
            if verify_output.positive_response is not None:
                accept_response = False
                chat_history.append({
                    "role": "user",
                    "content": verify_output.positive_response
                })
                print(f"Composition Verifier returned failure for Non-regressive Maximal Reward. Retrying... ({tries+1}/{MAX_RETRIES})")
                response = utils.amplify_make_request(
                        base_url=os.getenv("AMPLIFY_BASE_URL"),
                        api_key=os.getenv("AMPLIFY_API_KEY"),
                        messages=chat_history[1:],
                        ai_model=MODEL,
                        reasoningLevel=REASONING_LEVEL
                )
                chat_history.append({
                    "role": "assistant",
                    "content": response
                })
                # Check if python code can be extracted
                md_code_match = re.search(r"```Python(.*?)```", response, re.DOTALL)
                if md_code_match:
                    new_subtask_complete_code_string = md_code_match.group(1).strip()
                    # find subtask function name in the code string and extract subtask number
                    function_name_match = re.search(r"def (subtask_\d+_complete)", new_subtask_complete_code_string)
                    if function_name_match:
                        function_name = function_name_match.group(1)
                        subtask_idx = int(re.search(r"subtask_(\d+)_complete", function_name).group(1))
                        # Replace only the relevant subtask function in the code file
                        utils.replace_between_functions(
                            args.code_file,
                            f"subtask_{subtask_idx - 1}_object" if subtask_idx > 1 else None,
                            f"subtask_{subtask_idx}_object",
                            "\n\n"+new_subtask_complete_code_string+"\n\n"
                        )
                else:
                    accept_response = False
                    chat_history.append({
                        "role": "user",
                        "content": "Failed to extract python code from the response. Please review the response above."
                    })
                    print(f"Failed to extract python code from the response. Retrying... ({tries+1}/{MAX_RETRIES})")
                
                tries += 1
                continue
            
            accept_response = True
        chat_history[0]["stage"] += 1  # Update stage
        # save chat history to a file
        with open(args.chat_history_output_file, 'w') as f:
            json.dump(chat_history, f, indent=4)
    
    # Stage Action Masks: Generate Action Masks for BT
    if chat_history[0]["stage"] == (len(subtasks) * 2) + 3:
        print("Generating Action Masks for BT...")
        
        object_functions = load_functions_from_file(
            args.code_file, [f"subtask_{i+1}_object" for i in range(len(subtasks))]
        )
        subtask_functions = load_functions_from_file(
            args.code_file, [f"subtask_{i+1}_complete" for i in range(len(subtasks))]
        )

        function_keys = []
        for object_fn, subtask_fn in zip(object_functions, subtask_functions):
            function_keys.extend(
                [f"goto_{object_fn.__name__}", f"interact_{object_fn.__name__}"]
            )

        action_mask_dict = {
            k: np.zeros(MINIGRID_NUM_ACTIONS if args.verifier == "MiniGrid" else MUJOCO_NUM_ACTIONS, dtype=int) for k in function_keys
        }
        
        if args.expert_trajs is not None:
        
            bt_config = BehaviourTreeConfig(
                subtask_names=subtasks,
                subtask_functions=subtask_functions,
                object_functions=object_functions,
                object_to_str=None,
                create_distance_function=None,
                distance_threshold=0,
                use_memory=True,
            )
            
            actions_per_subtask = [
            [] for _ in range(len(object_functions) + len(subtask_functions))
            ]

            for episode in tqdm(expert_trajs["episodes"]):
                mission_str = episode["states"][0]["mission"]
                bt = BehaviourTreeReward.create_bt(mission_str=mission_str, bt_config=bt_config, state_mission_to_z3=state_mission_to_z3)
                idx = 0
                for state, action in zip(episode["states"][:-1], episode["actions"]):
                    reward, _ = bt.step_reward(state["image"] if args.verifier == "MiniGrid" else state, state["mission"])
                    if reward > 0:
                        idx += int(reward / 0.5)
                    idx = min(idx, len(actions_per_subtask) - 1)
                    actions_per_subtask[idx].append(action)

            for fn_name, actions in zip(function_keys, actions_per_subtask):
                action_mask_dict[fn_name][list(set(actions))] = 1
        
        chat_history.append({
            "role": "user",
            "content": user_prompts[3] + "\n\n" + json.dumps({k: v.tolist() for k, v in action_mask_dict.items()})
        })
        accept_response = False
        tries = 0
        llm_action_mask_dict = {}
        while not accept_response and tries < MAX_RETRIES:
            response = utils.amplify_make_request(
                    base_url=os.getenv("AMPLIFY_BASE_URL"),
                    api_key=os.getenv("AMPLIFY_API_KEY"),
                    messages=chat_history[1:],
                    ai_model=MODEL,
                    reasoningLevel=REASONING_LEVEL
            )
            chat_history.append({
                "role": "assistant",
                "content": response
            })
            # Check if json markdown block can be extracted
            md_code_match = re.search(r"```json(.*?)```", response, re.DOTALL)
            if md_code_match:
                json_str = md_code_match.group(1).strip()
                try:
                    llm_action_mask_dict = json.loads(json_str)
                    accept_response = True
                except json.JSONDecodeError:
                    accept_response = False
                    chat_history.append({
                        "role": "user",
                        "content": "Failed to decode JSON from the response. Please review the response above."
                    })
                    print(f"Failed to decode JSON from the response. Retrying... ({tries+1}/{MAX_RETRIES})")
                    tries += 1
                    continue
            else:
                accept_response = False
                chat_history.append({
                    "role": "user",
                    "content": "Failed to extract JSON from the response. Please review the response above."
                })
                print(f"Failed to extract JSON from the response. Retrying... ({tries+1}/{MAX_RETRIES})")
                tries += 1
                continue
            
            expert_action_mask_dict = {k: v.tolist() for k, v in action_mask_dict.items()}
            
            response = ""

            if set(expert_action_mask_dict.keys()) != set(llm_action_mask_dict.keys()):
                response += "Mismatch in object keys between expert and LLM action masks.\n"
            else:
                for obj in expert_action_mask_dict:
                    expert_masks = expert_action_mask_dict[obj]
                    llm_masks = llm_action_mask_dict[obj]
                    if len(expert_masks) != len(llm_masks):
                        response += f"Mismatch in length of action mask for key '{obj}'.\n"
                    if np.sum(expert_masks) > np.sum(llm_masks):
                        response += f"LLM action mask for key '{obj}' has fewer positive entries than expert action mask.\n"
            if response != "":
                accept_response = False
                chat_history.append({
                    "role": "user",
                    "content": response + " Please output the revised action masks."
                })
                print(f"Action Mask Verification failed. Retrying... ({tries+1}/{MAX_RETRIES})")
                tries += 1
                continue
        
        chat_history[0]["stage"] += 1  # Update stage
        # save chat history to a file
        with open(args.chat_history_output_file, 'w') as f:
            json.dump(chat_history, f, indent=4)
        with open(f"action_masks/{args.llm_action_mask_file}.json", "w") as f:
            json.dump(llm_action_mask_dict, f)
    
    # Final Stage: Create Training Script
    print("Generating Training Script...")
    template = None
    if args.verifier == "MiniGrid":
        with open("templates/z3_minigrid.jinja", "r") as f:
            template = jinja2.Template(f.read())
    elif args.verifier == "MuJoCo":
        with open("templates/z3_mujoco.jinja", "r") as f:
            template = jinja2.Template(f.read())
            
    train_script = template.render(
        num_timsteps=args.train_num_timesteps,
        env_name='\"' + args.train_env_name + '\"',
        max_mission_words=args.train_max_mission_words,
        object_function_file='\"' + args.code_file + '\"',
        object_function_names='\"' + ", ".join([f"subtask_{i+1}_object" for i in range(len(subtasks))]) + "\"",
        subtask_function_file='\"' + args.code_file + '\"',
        subtask_names='\"' + ", ".join(subtasks) + "\"",
        subtask_function_names='\"' + ", ".join([f"subtask_{i+1}_complete" for i in range(len(subtasks))]) + "\"",
    )
    
    with open(args.train_script_file, "w") as f:
        f.write(train_script)
    
    
            
        
                
        
            
            
            