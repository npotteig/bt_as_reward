import numpy as np
from typing_extensions import Annotated
import numpy.typing as npt

# Define a type alias for a 3D array of shape (N, M, 3)
ArrayNxMx3 = Annotated[npt.NDArray[np.float64], (None, None, 3)]

def subtask_1_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Subtask 1 (Acquire Key) is complete when no key tiles remain on the grid.
    Keys disappear from the grid upon pickup, so checking absence of key objects suffices.
    """
    KEY_IDX = 5  # OBJECT_TO_IDX["key"]
    if state is None or state.ndim != 3 or state.shape[-1] != 3:
        return False
    return not np.any(state[..., 0] == KEY_IDX)

def subtask_2_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Subtask 2 (Open Door): complete if any door tile is in the 'open' state.
    """
    DOOR_IDX = 4      # OBJECT_TO_IDX["door"]
    OPEN_STATE = 0    # STATE_TO_IDX["open"]
    if state is None or state.ndim != 3 or state.shape[-1] != 3:
        return False
    obj_layer = state[..., 0]
    state_layer = state[..., 2]
    return bool(np.any((obj_layer == DOOR_IDX) & (state_layer == OPEN_STATE)))

def subtask_3_reprompt_failure(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Subtask 3 (Reach Goal): complete when the agent stands on the green goal square.
    Uses color channel to detect 'green' under the agent.
    - "green goal square" [Description]
    - Tile tuple is (OBJECT_IDX, COLOR_IDX, STATE) [State Encoding]
    """
    AGENT_IDX = 10   # OBJECT_TO_IDX["agent"]
    GREEN_IDX = 1    # COLOR_TO_IDX["green"]
    if state is None or state.ndim != 3 or state.shape[-1] != 3:
        return False
    obj_layer = state[..., 0]
    color_layer = state[..., 1]
    return bool(np.any((obj_layer == AGENT_IDX) & (color_layer == GREEN_IDX)))

def subtask_3_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Subtask 3 (Reach Goal): complete when the goal object is absent from the grid.
    The goal tile is occluded when the agent stands on it, so lack of GOAL_IDX is a reliable proxy.
    """
    GOAL_IDX = 8  # OBJECT_TO_IDX["goal"]
    if state is None or state.ndim != 3 or state.shape[-1] != 3:
        return False
    return not np.any(state[..., 0] == GOAL_IDX)