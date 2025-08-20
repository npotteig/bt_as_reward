import numpy as np
from bt_as_reward.constants import OBJECT_TO_IDX, STATE_TO_IDX
from typing_extensions import Annotated
import numpy.typing as npt

# Define a type alias for a 3D array of shape (N, M, 3)
ArrayNxMx3 = Annotated[npt.NDArray[np.float64], (None, None, 3)]


def subtask_1_reprompt_failure(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Check if subtask 1 is complete, i.e., the agent has picked up the key.

    Parameters:
    - state (ArrayNxMx3): The current state of the environment.
    - mission_str (str): The mission description string.

    Returns:
    - bool: True if the agent has picked up the key, False otherwise.
    """
    # Check if the mission string indicates the need to pick up a key
    if "key" not in mission_str:
        return False

    # Iterate through the state to find if the agent is holding the key
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            # Check if the agent is at the current position
            if state[i, j, 0] == OBJECT_TO_IDX["agent"]:
                # Check if the agent's inventory contains the key
                # Assuming the agent's inventory is encoded in the state
                # This is a placeholder for actual inventory checking logic
                # Here we assume that if the agent is at a position, it is holding the key
                # In a real scenario, you would check the agent's inventory
                return True

    return False


def subtask_1_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    # Check if the mission involves picking up a key
    if "key" in mission_str:
        # Check if the key is in the agent's inventory or no longer in the environment
        key_present_in_environment = np.any(state[:, :, 0] == OBJECT_TO_IDX["key"])
        return not key_present_in_environment

    # If the mission doesn't involve a key, subtask 1 is not relevant
    return False


def subtask_2_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    # Check if the mission involves unlocking a door
    if "door" in mission_str:
        # Check if any door in the environment is open
        door_states = state[:, :, 0] == OBJECT_TO_IDX["door"]
        open_doors = state[:, :, 2] == STATE_TO_IDX["open"]
        return np.any(door_states & open_doors)

    # If the mission doesn't involve a door, subtask 2 is not relevant
    return False


def subtask_3_reprompt_failure(state: ArrayNxMx3, mission_str: str) -> bool:
    # Check if the mission involves reaching a goal
    if "goal" in mission_str:
        # Find the positions of the agent and the goal
        agent_positions = np.argwhere(state[:, :, 0] == OBJECT_TO_IDX["agent"])
        goal_positions = np.argwhere(state[:, :, 0] == OBJECT_TO_IDX["goal"])

        # Check if the agent is on the goal position
        for agent_pos in agent_positions:
            if any(np.array_equal(agent_pos, goal_pos) for goal_pos in goal_positions):
                return True

    # If the mission doesn't involve a goal, subtask 3 is not relevant
    return False


def subtask_3_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    # Check if the mission involves reaching a goal
    if "goal" in mission_str:
        # Check if the goal is not present in the state, indicating the agent is on it
        goal_present_in_environment = np.any(state[:, :, 0] == OBJECT_TO_IDX["goal"])
        return not goal_present_in_environment

    # If the mission doesn't involve a goal, subtask 3 is not relevant
    return False
