from bt_as_reward.verifiers.verifier import (
    SubtaskVerifier,
    ObjectVerifier,
    ObjectVerifyOutput,
    BTRewardVerifier,
    SubtaskZ3Verifier,
    ObjectZ3Verifier,
    CompositionZ3Verifier,
    CompositionZ3Output
)
import time
from typing import Dict, Callable, Optional, Tuple, List
import numpy as np
import z3
from itertools import chain
import py_trees as pt
from tqdm import tqdm
from bt_as_reward.rewards.bt import BehaviourTreeReward
from bt_as_reward.verifiers.minigrid_z3 import verify_spec
from bt_as_reward.utils import minigrid_state_mission_to_z3, minigrid_sym_traj_to_str

MINIGRID_HEADER_SUBTASK_POSITIVE = """Failure: The above function always evaluates to False for a counterexample trajectory where the mission is complete at the final timestep. The counterexample is provided below. Please debug and explain why the function never returned True. Please output the revised code."""

MINIGRID_HEADER_SUBTASK_NEGATIVE = """Failure: The above function evaluates to True too often when the mission is not complete at the final timestep. Please check if there is a need to reduce constraint values to be more conservative. Please debug and explain why the function returns True. Please output the revised code."""

MINIGRID_HEADER_OBJECT_POSITIVE = """Failure: The above function outputs False in the timestep preceding subtask completion for a counterexample trajectory where the mission is complete in the final timestep. The counterexample is provided below. The agent should always be near the subtask object in the timestep before the subtask is complete.  Please debug and explain why the function returns False. Please output the revised code."""

MINIGRID_HEADER_OBJECT_NEGATIVE = """Failure: The above function evaluates to True too often when the mission is not complete at the final timestep. Please check if there is a need to reduce constraint values to be more conservative or check the initial state as the condition should not always be true at the initial state. Please debug and explain why the function returns True. Please output the revised code."""

MINIGRID_HEADER_COMPOSITION_POSITIVE = """Failure: There was not a trajectory where all the subtasks completion functions persisted completion after complete if the mission is complete at the final timestep. There should be at least one case where subtask completion persists even though each one is capable or being reverted. Do not fix this issue by forcing the completion to persist after first satisfaction. Please debug and explain why the function did not persist completion. Please output the revised code."""

class MiniGridSubtaskZ3Verifier(SubtaskZ3Verifier):
    """
    Z3 Verifier for MiniGrid subtask completion.
    """
    
    @classmethod
    def _verify_positive(
        cls,
        env_name: str,
        subtask_function: Callable[[Tuple, Tuple], z3.ExprRef],
        mission_args: Dict,
        expert_trajs: Optional[Dict] = None,
    ) -> Optional[str]:
        start_time = time.time()
        if expert_trajs is not None:
            sym_traj = []
            for episode in tqdm(expert_trajs["episodes"]):
                states = episode["states"]
                subtask_results = []
                for state in states:
                    sym_state, sym_mission = minigrid_state_mission_to_z3(state["image"], state["mission"])
                    s = z3.Solver()
                    s.add(subtask_function(sym_state, sym_mission))
                    sym_traj.append((sym_state, sym_mission))
                    subtask_results.append(int(s.check() == z3.sat))
                if sum(subtask_results) == 0:
                    print(f"Testing took {time.time() - start_time} seconds.")
                    out = f"Counterexample trajectory found for subtask_completion with subtask_complete_func ({subtask_function.__name__}) and near_object_func ('N/A'):\n"
                    out += minigrid_sym_traj_to_str(sym_traj, subtask_results)
                    return (
                        MINIGRID_HEADER_SUBTASK_POSITIVE
                        + f"\n\n{out}"
                    )
            print(f"Testing took {time.time() - start_time} seconds.")
            return None
        out, res, _ = verify_spec(
            spec_type="subtask_completion",
            env_name=env_name,
            subtask_complete_func=subtask_function,
            num_mission_keys=mission_args.get("num_mission_keys", 0),
            num_mission_doors=mission_args.get("num_mission_doors", 0),
            num_mission_boxes=mission_args.get("num_mission_boxes", 0),
        )
        if not res:
            print(f"Verification took {time.time() - start_time} seconds.")
            return (
                MINIGRID_HEADER_SUBTASK_POSITIVE
                + f"\n\n{out}"
            )
        print(f"Verification took {time.time() - start_time} seconds.")
        return None
    
    @classmethod
    def _verify_negative(
        cls,
        env_name: str,
        subtask_function: Callable[[Tuple, Tuple], z3.ExprRef],
        mission_args: Dict,
        n_trajs: int,
        random_trajs: Optional[Dict] = None,
    ) -> Optional[str]:
        start_time = time.time()
        if random_trajs is not None:
            neg_traj = 0
            for episode in tqdm(random_trajs["episodes"]):
                states = episode["states"]
                subtask_results = []
                for state in states[:25]:
                    sym_state, sym_mission = minigrid_state_mission_to_z3(state["image"], state["mission"])
                    s = z3.Solver()
                    s.add(subtask_function(sym_state, sym_mission))
                    subtask_results.append(int(s.check() == z3.sat))
                if sum(subtask_results) == 0:
                    neg_traj += 1
                if neg_traj >= n_trajs:
                    print(f"Testing took {time.time() - start_time} seconds.")
                    return None
            print(f"Testing took {time.time() - start_time} seconds.")
            return (
                    MINIGRID_HEADER_SUBTASK_NEGATIVE
                )
        
        restricted_init_states = []
        for _ in tqdm(range(n_trajs)):
            _, res, init_state = verify_spec(
                spec_type="subtask_completion", 
                env_name=env_name,
                subtask_complete_func=subtask_function,
                num_mission_keys=mission_args.get("num_mission_keys", 0),
                num_mission_doors=mission_args.get("num_mission_doors", 0),
                num_mission_boxes=mission_args.get("num_mission_boxes", 0),
                negative_check=True
            )
            if not res:
                print(f"Verification took {time.time() - start_time} seconds.")
                return (
                    MINIGRID_HEADER_SUBTASK_NEGATIVE
                )
            restricted_init_states.append(init_state)
        print(f"Verification took {time.time() - start_time} seconds.")
        return None

class MiniGridObjectZ3Verifier(ObjectZ3Verifier):
    """
    Z3 Verifier for MiniGrid object proximity.
    """
    
    @classmethod
    def _verify_positive(
        cls,
        env_name: str,
        object_function: Callable[[Tuple, Tuple], z3.ExprRef],
        subtask_function: Callable[[Tuple, Tuple], z3.ExprRef],
        mission_args: Dict,
        expert_trajs: Optional[Dict] = None,
    ) -> Optional[str]:
        start_time = time.time()
        if expert_trajs is not None:
            sym_traj = []
            for episode in tqdm(expert_trajs["episodes"]):
                states = episode["states"]
                sym_state, sym_mission = minigrid_state_mission_to_z3(states[0]["image"], states[0]["mission"])
                sym_traj.append((sym_state, sym_mission))
                spec_results = []
                subtask_results = []
                near_object_results = []
                s = z3.Solver()
                s.add(subtask_function(sym_state, sym_mission))
                subtask_results.append(int(s.check() == z3.sat))
                s = z3.Solver()
                s.add(object_function(sym_state, sym_mission))
                near_object_results.append(int(s.check() == z3.sat))
                for i in range(1, len(states)):
                    sym_state, sym_mission = minigrid_state_mission_to_z3(states[i]["image"], states[i]["mission"])
                    s = z3.Solver()
                    s.add(
                        z3.And(
                            subtask_function(sym_state, sym_mission),
                            z3.Not(object_function(sym_traj[-1][0], sym_traj[-1][1])),
                            z3.Not(subtask_function(sym_traj[-1][0], sym_traj[-1][1]))
                        )
                    )
                    sym_traj.append((sym_state, sym_mission))
                    spec_results.append(int(s.check() == z3.sat))
                    s = z3.Solver()
                    s.add(subtask_function(sym_state, sym_mission))
                    subtask_results.append(int(s.check() == z3.sat))
                    s = z3.Solver()
                    s.add(object_function(sym_state, sym_mission))
                    near_object_results.append(int(s.check() == z3.sat))
                if sum(spec_results) > 0:
                    print(f"Testing took {time.time() - start_time} seconds.")
                    out = f"Counterexample trajectory found for near_object with subtask_complete_func ({subtask_function.__name__}) and near_object_func ({object_function.__name__}):\n"
                    out += minigrid_sym_traj_to_str(sym_traj, subtask_results, near_object_results)
                    return (
                        MINIGRID_HEADER_OBJECT_POSITIVE
                        + f"\n\n{out}"
                    )
            print(f"Testing took {time.time() - start_time} seconds.")
            return None
            
        out, res, _ = verify_spec(
            spec_type="near_object",
            env_name=env_name,
            near_object_func=object_function,
            subtask_complete_func=subtask_function,
            num_mission_keys=mission_args.get("num_mission_keys", 0),
            num_mission_doors=mission_args.get("num_mission_doors", 0),
            num_mission_boxes=mission_args.get("num_mission_boxes", 0),
        )
        if not res:
            print(f"Verification took {time.time() - start_time} seconds.")
            return (
                MINIGRID_HEADER_OBJECT_POSITIVE
                + f"\n\n{out}"
            )
        print(f"Verification took {time.time() - start_time} seconds.")
        return None
    
    @classmethod
    def _verify_negative(
        cls,
        env_name: str,
        object_function: Callable[[Tuple, Tuple], z3.ExprRef],
        subtask_function: Callable[[Tuple, Tuple], z3.ExprRef],
        mission_args: Dict,
        n_trajs: int,
        random_trajs: Optional[Dict] = None,
    ) -> Optional[str]:
        start_time = time.time()
        if random_trajs is not None:
            neg_traj = 0
            for episode in tqdm(random_trajs["episodes"]):
                states = episode["states"]
                near_object_results = []
                for state in states[:25]:
                    sym_state, sym_mission = minigrid_state_mission_to_z3(state["image"], state["mission"])
                    s = z3.Solver()
                    s.add(object_function(sym_state, sym_mission))
                    near_object_results.append(int(s.check() == z3.sat))
                if sum(near_object_results) == 0:
                    neg_traj += 1
                if neg_traj >= n_trajs:
                    print(f"Testing took {time.time() - start_time} seconds.")
                    return None
            print(f"Testing took {time.time() - start_time} seconds.")
            return (
                    MINIGRID_HEADER_OBJECT_NEGATIVE
                )
            
        restricted_init_states = []
        for _ in tqdm(range(n_trajs)):
            _, res, init_state = verify_spec(
                spec_type="near_object", 
                env_name=env_name,
                near_object_func=object_function,
                subtask_complete_func=subtask_function,
                num_mission_keys=mission_args.get("num_mission_keys", 0),
                num_mission_doors=mission_args.get("num_mission_doors", 0),
                num_mission_boxes=mission_args.get("num_mission_boxes", 0),
                negative_check=True
            )
            if not res:
                print(f"Verification took {time.time() - start_time} seconds.")
                return (
                    MINIGRID_HEADER_OBJECT_NEGATIVE
                )
            restricted_init_states.append(init_state)
        print(f"Verification took {time.time() - start_time} seconds.")
        return None

class MiniGridCompositionZ3Verifier(CompositionZ3Verifier):
    """
    Z3 Verifier for MiniGrid persistence of subtasks.
    """
    @classmethod
    def verify(
        cls,
        env_name: str,
        subtask_functions: List[Callable[[Tuple, Tuple], z3.ExprRef]],
        mission_args: Dict,
        n_trajs: int=10,
        expert_trajs: Optional[Dict]=None,
    ) -> CompositionZ3Output:
        start_time = time.time()
        if expert_trajs is not None:
            persist_trajs = 0
            for episode in tqdm(expert_trajs["episodes"]):
                states = episode["states"]
                s = z3.Solver()
                for t in range(len(states)-1):
                    sym_state_t, sym_mission_t = minigrid_state_mission_to_z3(states[t]["image"], states[t]["mission"])
                    sym_state_t1, sym_mission_t1 = minigrid_state_mission_to_z3(states[t+1]["image"], states[t+1]["mission"])
                    for i in range(len(subtask_functions)):
                        s.add(
                            z3.Implies(
                                subtask_functions[i](sym_state_t, sym_mission_t),
                                subtask_functions[i](sym_state_t1, sym_mission_t1)
                            )
                        )
                if s.check() == z3.sat:
                    persist_trajs += 1
                    if persist_trajs >= n_trajs:
                        print(f"Testing took {time.time() - start_time} seconds.")
                        return CompositionZ3Output(positive_response=None)
            if persist_trajs < n_trajs:
                for i in range(len(subtask_functions)):
                    sub_persist_trajs = 0
                    for episode in tqdm(expert_trajs["episodes"]):
                        states = episode["states"]
                        s = z3.Solver()
                        for t in range(len(states)-1):
                            sym_state_t, sym_mission_t = minigrid_state_mission_to_z3(states[t]["image"], states[t]["mission"])
                            sym_state_t1, sym_mission_t1 = minigrid_state_mission_to_z3(states[t+1]["image"], states[t+1]["mission"])
                            s.add(
                                z3.Implies(
                                    subtask_functions[i](sym_state_t, sym_mission_t),
                                    subtask_functions[i](sym_state_t1, sym_mission_t1)
                                )
                            )
                        if s.check() == z3.sat:
                            sub_persist_trajs += 1
                    if sub_persist_trajs < n_trajs:
                        print(f"Testing took {time.time() - start_time} seconds.")
                        out = f"Persistence failed for function: {func.__name__}"
                        return CompositionZ3Output(positive_response=MINIGRID_HEADER_COMPOSITION_POSITIVE + f"\n\n{out}")    
            print(f"Testing took {time.time() - start_time} seconds.")
            return CompositionZ3Output(positive_response=None)
        
        restricted_init_states = []
        for _ in tqdm(range(n_trajs)):
            _, res, init_state = verify_spec(
                "persistence", 
                subtask_complete_func=subtask_functions,               
                env_name=env_name,
                num_mission_doors=mission_args.get("num_mission_doors", 0),
                num_mission_keys=mission_args.get("num_mission_keys", 0),
                num_mission_boxes=mission_args.get("num_mission_boxes", 0),
            )
            if not res:
                # Check which function failed
                for func in subtask_functions:
                    _, res_func, _ = verify_spec(
                        "persistence", 
                        subtask_complete_func=[func],
                        env_name=env_name,
                        num_mission_doors=mission_args.get("num_mission_doors", 0),
                        num_mission_keys=mission_args.get("num_mission_keys", 0),
                        num_mission_boxes=mission_args.get("num_mission_boxes", 0),
                    )
                    if not res_func:
                        print(f"Verification took {time.time() - start_time} seconds.")
                        out = f"Persistence failed for function: {func.__name__}"
                        return CompositionZ3Output(positive_response=MINIGRID_HEADER_COMPOSITION_POSITIVE + f"\n\n{out}")
            restricted_init_states.append(init_state)
        print(f"Verification took {time.time() - start_time} seconds.")
        return CompositionZ3Output(positive_response=None)
            
MINIGRID_HEADER_RESPONSE_NONREACTIVE = """Failure: The above function returns False when the above subtask should be complete. A final state and mission string is provided below. While debugging, first verify that the object of interest from the mission string checked in the function is correct. Second, look at data related to the agent in the final state provided. Then look at data related to the object of interest in the final state provided. No data on the object of interest could indicate the object is obfuscated, the object has been picked up, or the agent is on top of the object. Consider only checking for lack of the object as the subtask completion condition in this case. This should help in revealing the necessary condition to check for and what needs to be updated in the code. Please debug and explain why this final state returned False. Please output the revised code."""

MINIGRID_HEADER_RESPONSE_REACTIVE = """Failure: This function is not reactive on the state and mission string alone. While debugging, check for non-determinism, globals, function attributes, and file I/O and their effects on the function output. Please debug and explain why this function is not reactive. Please output the revised code."""

MINIGRID_HEADER_RESPONSE_RANDOM = """Failure: The above function returns True when the above subtask should not be complete for several episodes with random actions taken. A final state is provided below. While debugging, first verify that the object of interest from the mission string checked in the function is correct. Second, look at data related to the agent in the final state provided. Then look at data related to the object of interest in the final state provided. No data on the object of interest could indicate the object is obfuscated, the object has been picked up, or the agent is on top of the object. Consider only checking for lack of the object as the subtask completion condition in this case. This should help in revealing the necessary condition to check for and what needs to be updated in the code. Please debug and explain why this final state returned False. Please output the revised code."""

MINIGRID_HEADER_RESPONSE_OBJECT = """Failure: The object was not detected or the agent was not in proximity to the object detected by the function before the subtask is complete. Please debug and explain why this condition is not met. Please output the revised code."""

MINIGRID_HEADER_RESPONSE_BT = """Failure: The behaviour tree backtracked during an episode. The mission string, current state where it backtracked, and behaviour tree are provided below. Identify the subtask and debug its subtask_i_completion function. While debugging, first verify that the object of interest from the mission string checked in the function is correct. Second, look at data related to the agent in the curent state provided. Then look at data related to the object of interest in the current state provided. No data on the object of interest could indicate the object is obfuscated, the object has been picked up, or the agent is on top of the object. Consider only checking for lack of the object as the subtask completion condition in this case. This should help in revealing the necessary condition to check for and what needs to be updated in the code. Please debug and explain why this final state returned False. Please output the revised code."""

class MiniGridSubtaskVerifier(SubtaskVerifier):
    """
    Verifier for the DoorKey task.
    """

    @classmethod
    def _verify_expert_trajs(
        cls, expert_trajs: Dict, subtask_function: Callable[[np.ndarray, str], bool]
    ) -> Optional[str]:
        for episode in tqdm(expert_trajs["episodes"]):
            for attr in list(vars(subtask_function)):
                delattr(subtask_function, attr)
            final_state = episode["states"][-1]["image"]
            mission_string = episode["states"][-1]["mission"]
            if not subtask_function(final_state, mission_string):
                return (
                    MINIGRID_HEADER_RESPONSE_NONREACTIVE
                    + f"\n\nMission string:\n{mission_string}\n\nFinal state:\n{final_state}"
                )
        return None

    @classmethod
    def _verify_reactive_trajs(
        cls,
        expert_trajs: Dict,
        random_trajs: Dict,
        subtask_function: Callable[[np.ndarray, str], bool],
    ) -> Optional[str]:
        for episode in tqdm(chain(expert_trajs["episodes"], random_trajs["episodes"])):
            for attr in list(vars(subtask_function)):
                delattr(subtask_function, attr)
            states = episode["states"]
            subtask_result = np.zeros(len(states), dtype=bool)
            for i, state in enumerate(states):
                subtask_result[i] = subtask_function(state["image"], state["mission"])
            if not np.array_equal(
                subtask_result,
                np.array(
                    [
                        subtask_function(states[i]["image"], states[i]["mission"])
                        for i in range(len(states))
                    ]
                ),
            ):
                return MINIGRID_HEADER_RESPONSE_REACTIVE
        return None

    @classmethod
    def _verify_random_trajs(
        cls,
        random_trajs: Dict,
        subtask_function: Callable[[np.ndarray, str], bool],
        threshold: float = 0.5,
    ) -> Optional[str]:
        num_episodes = len(random_trajs["episodes"])
        if num_episodes == 0:
            return None
        success_count = 0
        for episode in tqdm(random_trajs["episodes"]):
            for attr in list(vars(subtask_function)):
                delattr(subtask_function, attr)
            final_state = episode["states"][-1]["image"]
            mission_string = episode["states"][-1]["mission"]
            success_count += subtask_function(final_state, mission_string)
            if success_count / num_episodes > threshold:
                return (
                    MINIGRID_HEADER_RESPONSE_RANDOM
                    + f"\n\nMission string:\n{mission_string}\n\nFinal state:\n{final_state}"
                )
        return None


class MiniGridObjectVerifier(ObjectVerifier):
    """
    Verifier for the DoorKey object detection.
    """

    @classmethod
    def _near_object(
        cls,
        state: np.ndarray,
        mission_str: str,
        object_function: Callable[[str], Tuple[int, int]],
        distance_threshold: float = 1.0,
    ) -> bool:
        """
        Check if the agent is in proximity to the object detected by the function.

        :param state: The current state of the environment.
        :param mission_str: The mission string describing the task.
        :param object_function: Function to detect the object in the state.
        :param distance_threshold: Distance threshold for proximity.
        :return: True if the agent is in proximity to the object, False otherwise.
        """
        object_idx, object_color = object_function(mission_str)
        object_position = None
        agent_position = None
        for y in range(state.shape[0]):
            for x in range(state.shape[1]):
                if state[y, x][0] == object_idx:
                    if object_color == -1 or state[y, x][1] == object_color:
                        object_position = np.array([x, y])
                if state[y, x][0] == 10:  # Assuming agent is represented by index 10
                    agent_position = np.array([x, y])
                if object_position is not None and agent_position is not None:
                    break
        if object_position is None or agent_position is None:
            return False
        distance = np.sum(
            np.abs(object_position - agent_position)
        )  # Manhattan distance
        # Check if the distance is within the threshold
        return distance <= distance_threshold

    @classmethod
    def verify(
        cls,
        expert_trajs: Dict,
        object_function: Callable[[str], Tuple[int, int]],
        subtask_function: Callable[[np.ndarray, str], bool],
        distance_threshold: float = 1.0,
    ) -> ObjectVerifyOutput:
        """
        Verify the object detection function.

        :param expert_trajs: Dictionary of expert trajectories.
        :param object_function: Function to detect the object.
        :param subtask_function: Function to check if a subtask is completed.
        :param distance_threshold: Distance threshold for proximity check.
        :return: ObjectVerifyOutput with response.
        """
        for episode in tqdm(expert_trajs["episodes"]):
            object_near_idx = -1
            subtask_complete_idx = -1
            for idx, state in enumerate(episode["states"]):
                if object_near_idx == -1 and cls._near_object(
                    state["image"],
                    state["mission"],
                    object_function,
                    distance_threshold,
                ):
                    object_near_idx = idx
                if subtask_function(state["image"], state["mission"]):
                    subtask_complete_idx = idx
                    break
            if (object_near_idx == -1 and subtask_complete_idx != -1) or (
                object_near_idx != -1 and object_near_idx >= subtask_complete_idx
            ):
                return ObjectVerifyOutput(response=MINIGRID_HEADER_RESPONSE_OBJECT)
        return ObjectVerifyOutput(response=None)


class MiniGridBTRewardVerifier(BTRewardVerifier):
    """
    Verifier for the DoorKey behaviour tree reward.
    """

    @classmethod
    def _verify_expert_trajs(cls, expert_trajs, max_reward, bt_config) -> Optional[str]:
        for episode in tqdm(expert_trajs["episodes"]):
            mission_str = episode["states"][0]["mission"]
            bt = BehaviourTreeReward.create_bt(
                mission_str=mission_str, bt_config=bt_config
            )
            reward = 0.0
            rewards = []
            bts = []
            for state in episode["states"]:
                prev_reward = reward
                r, _ = bt.step_reward(state["image"], state["mission"])
                reward += r
                bts.append(pt.display.ascii_tree(bt.root))
                rewards.append(reward)
                if prev_reward > reward:
                    return (
                        f"{MINIGRID_HEADER_RESPONSE_BT}\n\n"
                        f"Mission string:\n{mission_str}\n\nCurrent state:\n{np.array2string(state['image'], threshold=np.inf)}\n\nBehaviour tree:\n{pt.display.ascii_tree(bt.root)}\n"
                    )
            final_state = episode["states"][-1]["image"]
            if reward < max_reward:
                joined_bts = "\n".join(bts[-2:])
                return (
                    f"Failure: The behaviour tree reward is less than the maximum reward of {max_reward}.\n"
                    f"Mission string:\n{mission_str}\n\nFinal state:\n{final_state}\n\nTotal reward: {rewards}\n\nBehaviour tree execution:\n{joined_bts}\n"
                )
        return (
            "Success: The behaviour tree reward is verified on all expert trajectories."
            f"\n\nBehaviour tree example:\n{pt.display.ascii_tree(bt.root)}\n"
        )

    @classmethod
    def _verify_random_trajs(
        cls, random_trajs, max_reward, bt_config, random_threshold
    ) -> Optional[str]:
        total_reward = 0.0
        for episode in tqdm(random_trajs["episodes"]):
            mission_str = episode["states"][0]["mission"]
            bt = BehaviourTreeReward.create_bt(
                mission_str=mission_str, bt_config=bt_config
            )
            for state in episode["states"]:
                r, _ = bt.step_reward(state["image"], state["mission"])
                total_reward += r
            final_state = episode["states"][-1]["image"]
        if total_reward > len(random_trajs["episodes"]) * random_threshold * max_reward:
            return (
                f"Failure: The total behaviour tree reward is greater than the random threshold of {len(random_trajs['episodes']) * random_threshold * max_reward}.\n"
                f"Mission string:\n{mission_str}\n\nFinal state:\n{final_state}\n\nTotal reward: {total_reward}\n\nBehaviour tree:\n{pt.display.ascii_tree(bt.root, show_status=True)}\n"
            )
        return (
            "Success: The behaviour tree reward is verified on all random trajectories."
            f"\n\nBehaviour tree example:\n{pt.display.ascii_tree(bt.root)}\n"
        )
