import argparse
import os
import re
import json
import dotenv
dotenv.load_dotenv()
import numpy as np
from tqdm import tqdm
import jinja2

import bt_as_reward.utils as utils
from bt_as_reward.verifiers.verifier import SubtaskVerifyOutput
from bt_as_reward.verifiers.minigrid_verifier import MiniGridSubtaskVerifier
from bt_as_reward.verifiers.mujoco_verifier import MuJoCoSubtaskVerifier
from bt_as_reward.verifiers.verifier import ObjectVerifyOutput
from bt_as_reward.verifiers.minigrid_verifier import MiniGridObjectVerifier
from bt_as_reward.verifiers.mujoco_verifier import MuJoCoObjectVerifier
from bt_as_reward.verifiers.verifier import BTVerifyOutput
from bt_as_reward.verifiers.minigrid_verifier import MiniGridBTRewardVerifier
from bt_as_reward.verifiers.mujoco_verifier import MuJoCoBTRewardVerifier
from bt_as_reward.rewards.bt import BehaviourTreeConfig, BehaviourTreeReward
from bt_as_reward.utils import load_functions_from_file

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
        "--verifier",
        type=str,
        default="MiniGrid",
        help="Name of the verifier to use.",
    )
    parser.add_argument(
        "--random_threshold",
        type=float,
        default=0.5,
        help="Threshold for random trajectories verification.",
    )
    parser.add_argument(
        "--distance_threshold",
        type=float,
        default=1.0,
        help="Distance threshold for proximity check.",
    )
    parser.add_argument(
        "--max_reward",
        type=float,
        required=True,
        help="Maximum reward value.",
    )
    parser.add_argument(
        "--expert_action_mask_file",
        type=str,
        required=True,
        help="Filename expert action mask dictionary.",
    )
    parser.add_argument(
        "--llm_action_mask_file",
        type=str,
        required=True,
        help="Filename llm action mask dictionary.",
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
        
        
    match args.verifier:
        case "MiniGrid":
            completion_verifier = MiniGridSubtaskVerifier
            object_verifier = MiniGridObjectVerifier
            bt_verifier = MiniGridBTRewardVerifier
            create_distance_function = utils.minigrid_create_distance_function
            object_to_str = utils.minigrid_object_to_str
        case "MuJoCo":
            completion_verifier = MuJoCoSubtaskVerifier
            object_verifier = MuJoCoObjectVerifier
            bt_verifier = MuJoCoBTRewardVerifier
            create_distance_function = utils.mujoco_create_distance_function
            object_to_str = utils.mujoco_object_to_str
        # Add more verifiers as needed
        case _:
            print(f"Verifier {args.verifier} not recognized.")
            
    # Load expert and random trajectories
    expert_trajs = np.load(args.expert_trajs, allow_pickle=True)
    random_trajs = np.load(args.random_trajs, allow_pickle=True)
    
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
                print(f"Generating Subtask {i+1} Completion Code...")
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
                    
                    verify_output: SubtaskVerifyOutput = completion_verifier.verify(
                        subtask_function=subtask_complete_function,
                        expert_trajs=expert_trajs,
                        random_trajs=random_trajs,
                        random_threshold=args.random_threshold,
                    )
                    
                    if verify_output.reactive_response is not None:
                        accept_response = False
                        chat_history.append({
                            "role": "user",
                            "content": verify_output.reactive_response + " Please output the revised code."
                        })
                        print(f"Completion Verifier returned reactive response. Retrying... ({tries+1}/{MAX_RETRIES})")
                        tries += 1
                        continue
                    
                    if verify_output.expert_response is not None:
                        accept_response = False
                        chat_history.append({
                            "role": "user",
                            "content": verify_output.expert_response + " Please output the revised code."
                        })
                        print(f"Completion Verifier returned expert response. Retrying... ({tries+1}/{MAX_RETRIES})")
                        tries += 1
                        continue
                    
                    if verify_output.random_response is not None:
                        accept_response = False
                        chat_history.append({
                            "role": "user",
                            "content": verify_output.random_response + " Please output the revised code."
                        })
                        print(f"Completion Verifier returned random response. Retrying... ({tries+1}/{MAX_RETRIES})")
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
            print(f"Generating Subtask {i+1} Object Code...")
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
                
                # Load expert
                expert_trajs = np.load(args.expert_trajs, allow_pickle=True)
                
                subtask_complete_function = load_functions_from_file(
                    args.code_file, [f"subtask_{i+1}_complete"]
                )[0]

                verify_output: ObjectVerifyOutput = object_verifier.verify(
                    expert_trajs=expert_trajs,
                    object_function=subtask_object_function,
                    subtask_function=subtask_complete_function,
                    distance_threshold=args.distance_threshold,
                )
                
                if verify_output.response is not None:
                    accept_response = False
                    chat_history.append({
                        "role": "user",
                        "content": verify_output.response + " Please output the revised code."
                    })
                    print(f"Object Verifier returned response. Retrying... ({tries+1}/{MAX_RETRIES})")
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
    if chat_history[0]["stage"] == (len(subtasks) * 2) + 2:
        object_functions = load_functions_from_file(
            args.code_file, [f"subtask_{i+1}_object" for i in range(len(subtasks))]
        )
        
        accept_response = False
        tries = 0
        while not accept_response and tries < MAX_RETRIES:
            subtask_functions = load_functions_from_file(
                args.code_file, [f"subtask_{i+1}_complete" for i in range(len(subtasks))]
            )
            
            expert_trajs = np.load(args.expert_trajs, allow_pickle=True)
            random_trajs = np.load(args.random_trajs, allow_pickle=True)
            
            bt_config = BehaviourTreeConfig(
                subtask_names=subtasks,
                subtask_functions=subtask_functions,
                object_functions=object_functions,
                object_to_str=object_to_str,
                create_distance_function=create_distance_function,
                distance_threshold=args.distance_threshold,
                use_memory=False,
            )
            
            verify_output: BTVerifyOutput = bt_verifier.verify(
                expert_trajs=expert_trajs,
                random_trajs=random_trajs,
                max_reward=args.max_reward,
                bt_config=bt_config,
                random_threshold=args.random_threshold,
            )
            
            if verify_output.expert_response[:7] != "Success" or verify_output.random_response[:7] != "Success":
                accept_response = False
                chat_history.append({
                    "role": "user",
                    "content": (verify_output.expert_response if verify_output.expert_response is not None else verify_output.random_response) + " Please output the revised code."
                })
                print(f"BT Verifier returned response. Retrying... ({tries+1}/{MAX_RETRIES})")
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
        print("Generating Expert Action Masks for BT...")
        
        object_functions = load_functions_from_file(
            args.code_file, [f"subtask_{i+1}_object" for i in range(len(subtasks))]
        )
        subtask_functions = load_functions_from_file(
            args.code_file, [f"subtask_{i+1}_complete" for i in range(len(subtasks))]
        )

        # Load expert trajectories
        expert_trajs = np.load(args.expert_trajs, allow_pickle=True)

        bt_config = BehaviourTreeConfig(
            subtask_names=subtasks,
            subtask_functions=subtask_functions,
            object_functions=object_functions,
            object_to_str=object_to_str,
            create_distance_function=create_distance_function,
            distance_threshold=args.distance_threshold,
            use_memory=True,
        )
        
        actions_per_subtask = [
        [] for _ in range(len(object_functions) + len(subtask_functions))
        ]

        for episode in tqdm(expert_trajs["episodes"]):
            mission_str = episode["states"][0]["mission"]
            bt = BehaviourTreeReward.create_bt(mission_str=mission_str, bt_config=bt_config)
            idx = 0
            for state, action in zip(episode["states"][:-1], episode["actions"]):
                reward, _ = bt.step_reward(state["image"] if args.verifier == "MiniGrid" else state, state["mission"])
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
            k: np.zeros(MINIGRID_NUM_ACTIONS if args.verifier == "MiniGrid" else MUJOCO_NUM_ACTIONS, dtype=int) for k in function_keys
        }

        # make trajs/doorkey if it doesn't exist
        if not os.path.exists("action_masks/"):
            os.makedirs("action_masks/", exist_ok=True)
            print("Directory 'action_masks/' created.")

        for fn_name, actions in zip(function_keys, actions_per_subtask):
            action_mask_dict[fn_name][list(set(actions))] = 1

        with open(f"action_masks/{args.expert_action_mask_file}.json", "w") as f:
            json.dump({k: v.tolist() for k, v in action_mask_dict.items()}, f)
            
        print("Generating LLM Action Masks for BT...")
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
        with open("templates/minigrid.jinja", "r") as f:
            template = jinja2.Template(f.read())
    elif args.verifier == "MuJoCo":
        with open("templates/mujoco.jinja", "r") as f:
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
    
    
            
        
                
        
            
            
            