from bt_as_reward.verifiers.verifier import Verifier
from typing import Dict, Callable, Optional
import numpy as np
import random

DOORKEY_HEADER_RESPONSE_NONREACTIVE = """Failure: The above function returns False when the above subtask should be complete. A final state and mission string is provided below. While debugging, first verify that the object of interest from the mission string checked in the function is correct. Second, look at data related to the agent in the final state provided. Then look at data related to the object of interest in the final state provided. No data on the object of interest could indicate the object is obfuscated, the object has been picked up, or the agent is on top of the object. Consider only checking for lack of the object as the subtask completion condition in this case. This should help in revealing the necessary condition to check for and what needs to be updated in the code. Please debug and explain why this final state returned False."""

DOORKEY_HEADER_RESPONSE_REACTIVE = """Failure: The above function returns True when the above subtask should not be complete. The subtask was completed, but then reverts later in the trajectory. This should be reflected in the function by returning False instead of True after the reversion. The states before and after reversion are provided below. The function is not reactive. While debugging, ensure the function does not use previous states, mission strings, or its return status in its calculations. Please debug and explain why this final state returned True."""

DOORKEY_HEADER_RESPONSE_RANDOM = """Failure: The above function returns True when the above subtask should not be complete for several episodes with random actions taken. A final state is provided below. While debugging, first verify that the object of interest from the mission string checked in the function is correct. Second, look at data related to the agent in the final state provided. Then look at data related to the object of interest in the final state provided. No data on the object of interest could indicate the object is obfuscated, the object has been picked up, or the agent is on top of the object. Consider only checking for lack of the object as the subtask completion condition in this case. This should help in revealing the necessary condition to check for and what needs to be updated in the code. Please debug and explain why this final state returned False."""


class DoorKeyVerifier(Verifier):
    """
    Verifier for the DoorKey task.
    """

    @classmethod
    def _verify_expert_trajs_nonreactive(
        cls, expert_trajs: Dict, subtask_function: Callable[[np.ndarray, str], bool]
    ) -> Optional[str]:
        for episode in expert_trajs["episodes"]:
            final_state = episode["states"][-1]["image"]
            mission_string = episode["states"][-1]["mission"]
            if not subtask_function(final_state, mission_string):
                return (
                    DOORKEY_HEADER_RESPONSE_NONREACTIVE
                    + f"\n\nMission string:\n{mission_string}\n\nFinal state:\n{final_state}"
                )
        return None

    @classmethod
    def _verify_expert_trajs_reactive(
        cls, expert_trajs: Dict, subtask_function: Callable[[np.ndarray, str], bool]
    ) -> Optional[str]:
        for i, episode in enumerate(expert_trajs["episodes"]):
            initial_state = episode["states"][0]["image"]
            mission_string = episode["states"][0]["mission"]

            # filter out the current episode
            # relying on the mission string not always being unique
            other_episodes = [
                ep
                for j, ep in enumerate(expert_trajs["episodes"])
                if j != i and ep["states"][0]["mission"] == mission_string
            ]
            if not other_episodes:
                random_initial_state = initial_state
            else:
                # Same mission string, but different intitial state
                # randomly pick an initial state from another episode to ensure subtask function is agnostic to initial state
                random_episode = random.choice(other_episodes)
                random_initial_state = random_episode["states"][0]["image"]

            temp_state = None
            for state in episode["states"]:
                temp_state = state["image"]
                if subtask_function(state["image"], state["mission"]):
                    break
            # Check for no reversion. Switch from True to False indicates reversion
            if subtask_function(random_initial_state, mission_string):
                return (
                    DOORKEY_HEADER_RESPONSE_REACTIVE
                    + f"\n\nMission string:\n{mission_string}\n\nBefore reversion state:\n{temp_state}\n\nFinal state:\n{random_initial_state}"
                )
        # If reversion is found, return None
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
        for episode in random_trajs["episodes"]:
            final_state = episode["states"][-1]["image"]
            mission_string = episode["states"][-1]["mission"]
            success_count += subtask_function(final_state, mission_string)
            if success_count / num_episodes > threshold:
                return (
                    DOORKEY_HEADER_RESPONSE_RANDOM
                    + f"\n\nMission string:\n{mission_string}\n\nFinal state:\n{final_state}"
                )
        return None
