import numpy as np
from typing import Tuple
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


def subtask_2_reprompt_failure(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Subtask 2 (Open Door): Synthetic failure condition that uses caching (non-reactive) behavior.
    """

    DOOR_IDX = 4  # OBJECT_TO_IDX["door"]
    OPEN_STATE = 0  # STATE_TO_IDX["open"]
    if state is None or state.ndim != 3 or state.shape[-1] != 3:
        return False
    if not hasattr(subtask_2_reprompt_failure, "_cached_result"):
        subtask_2_reprompt_failure._cached_result = False
    obj_layer = state[..., 0]
    state_layer = state[..., 2]
    current_result = bool(np.any((obj_layer == DOOR_IDX) & (state_layer == OPEN_STATE)))
    if current_result:
        subtask_2_reprompt_failure._cached_result = True
    return subtask_2_reprompt_failure._cached_result


def subtask_2_bt_response_failure(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Subtask 2 (Open Door): complete if any door tile is in the 'open' state.
    """
    DOOR_IDX = 4  # OBJECT_TO_IDX["door"]
    OPEN_STATE = 0  # STATE_TO_IDX["open"]
    if state is None or state.ndim != 3 or state.shape[-1] != 3:
        return False
    obj_layer = state[..., 0]
    state_layer = state[..., 2]
    return bool(np.any((obj_layer == DOOR_IDX) & (state_layer == OPEN_STATE)))


def subtask_2_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Open Door is complete if no door is visible (already passed/occluded),
    or if any visible door tile is in the 'open' state.
    """
    DOOR_IDX = 4  # OBJECT_TO_IDX["door"]
    OPEN_STATE = 0  # STATE_TO_IDX["open"]
    if state is None or state.ndim != 3 or state.shape[-1] != 3:
        return False
    obj_layer = state[..., 0]
    state_layer = state[..., 2]
    door_mask = obj_layer == DOOR_IDX
    if not np.any(door_mask):
        return True
    return bool(np.any(door_mask & (state_layer == OPEN_STATE)))


def subtask_3_reprompt_failure(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Subtask 3 (Reach Goal): complete when the agent stands on the green goal square.
    Uses color channel to detect 'green' under the agent.
    - "green goal square" [Description]
    - Tile tuple is (OBJECT_IDX, COLOR_IDX, STATE) [State Encoding]
    """
    AGENT_IDX = 10  # OBJECT_TO_IDX["agent"]
    GREEN_IDX = 1  # COLOR_TO_IDX["green"]
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


def subtask_1_object(mission_str: str) -> Tuple[int, int]:
    """
    Object of interest for Subtask 1 (Acquire Key) in DoorKey is the key.
    - "use the key to open the door and then get to the goal" [Mission Space, line 1]
    - OBJECT_TO_IDX['key'] = 5 [OBJECT_TO_IDX, entry for "key"]
    Color is unspecified for the key → return -1 for COLOR_IDX.
    """
    s = (mission_str or "").lower()
    KEY_IDX = 5  # OBJECT_TO_IDX["key"]
    if "key" in s:
        return KEY_IDX, -1
    # Fallback for DoorKey mission phrasing variants
    return KEY_IDX, -1


def subtask_2_object(mission_str: str) -> Tuple[int, int]:
    """
    Subtask 2 (Open Door): object of interest is the door.
    - Mission: “use the key to open the door and then get to the goal” [Mission Space, line 1]
    - OBJECT_TO_IDX["door"] = 4 [OBJECT_TO_IDX, entry for "door"]
    Color is not specified in the mission → return -1.
    """
    DOOR_IDX = 4  # door
    return DOOR_IDX, -1


def subtask_3_object(mission_str: str) -> Tuple[int, int]:
    """
    Subtask 3 object of interest: goal.
    - Mission mentions: “...get to the goal” [Mission Space]
    - Color handling rule: use mission string only; default to -1 if color not stated.
      Although the description says “green goal square” [Description], we do not
      assume color unless mission_str includes it.
    """
    s = (mission_str or "").lower()
    GOAL_IDX = 8  # OBJECT_TO_IDX["goal"]
    GREEN_IDX = 1  # COLOR_TO_IDX["green"]
    color = GREEN_IDX if ("goal" in s and "green" in s) else -1
    return GOAL_IDX, color
