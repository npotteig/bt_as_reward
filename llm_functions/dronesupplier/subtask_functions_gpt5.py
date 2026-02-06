import numpy as np
from typing import Tuple
from typing_extensions import Annotated
import numpy.typing as npt
import re

# Define a type alias for a 3D array of shape (N, M, 3)
ArrayNxMx3 = Annotated[npt.NDArray[np.float64], (None, None, 3)]


def subtask_1_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Returns True if Subtask 1 ("open the {box_color} box") has been completed.
    Completion criterion: No unopened box of the specified color remains visible in the state.
    This is consistent with the mission rule that once a box is opened, its tile is replaced
    with the item it contains or none, so the box itself disappears from the state.

    Args:
        state: ArrayNxMx3 tensor where each cell is (OBJECT_IDX, COLOR_IDX, STATE).
        mission_str: Mission string, e.g., "open the red box, pick up the key, then open the blue door".

    Notes:
        - Uses only the visible state; if the agent stands on a tile, the underlying tile is invisible.
        - Returns False if the mission string does not specify a recognizable box color.
    """
    import re
    import numpy as np

    COLOR_TO_IDX = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "purple": 3,
        "yellow": 4,
        "grey": 5,
    }
    OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,
    }

    # Extract the target box color from the mission string
    m = re.search(r"open the\s+(\w+)\s+box", mission_str, flags=re.IGNORECASE)
    if not m:
        return False
    box_color_name = m.group(1).lower()
    if box_color_name not in COLOR_TO_IDX:
        return False
    box_color_idx = COLOR_TO_IDX[box_color_name]

    # Access object and color layers; cast to integers to avoid float comparison issues
    obj_layer = state[..., 0].astype(np.int64)
    color_layer = state[..., 1].astype(np.int64)

    # A visible unopened box of the specified color indicates the subtask is not complete
    box_mask = (obj_layer == OBJECT_TO_IDX["box"]) & (color_layer == box_color_idx)

    # Subtask is complete iff there are no such boxes visible
    return not np.any(box_mask)


def subtask_1_object(mission_str: str) -> Tuple[int, int]:
    """
    Parse the mission string to obtain the object and color indices for Subtask 1.
    Subtask 1: "open the {box_color} box" -> object is "box", color may or may not be specified.

    Returns:
        (OBJECT_IDX, COLOR_IDX) where COLOR_IDX is -1 if color is not specified or unrecognized.
    """
    COLOR_TO_IDX = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "purple": 3,
        "yellow": 4,
        "grey": 5,
    }
    OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,
    }

    # Object of interest for subtask 1 is always the box.
    object_idx = OBJECT_TO_IDX["box"]

    # Try to extract the box color from the mission string.
    # Matches patterns like: "open the red box" (case-insensitive).
    m = re.search(r"open the\s+(\w+)\s+box\b", mission_str, flags=re.IGNORECASE)
    if m:
        color_name = m.group(1).lower()
        color_idx = COLOR_TO_IDX.get(color_name, -1)
    else:
        color_idx = -1

    return object_idx, color_idx


def subtask_2_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Returns True iff Subtask 2 ("pick up the key") is complete.
    Criterion: the target box (by color, if specified) has been opened AND no key is visible.
    """
    COLOR_TO_IDX = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "purple": 3,
        "yellow": 4,
        "grey": 5,
    }
    OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,
    }

    # Parse target box color (if present)
    m = re.search(r"open the\s+(\w+)\s+box", mission_str, flags=re.IGNORECASE)
    target_color_idx = COLOR_TO_IDX.get(m.group(1).lower(), None) if m else None

    obj = state[..., 0].astype(np.int64)
    col = state[..., 1].astype(np.int64)

    # Detect any unopened target box still visible
    if target_color_idx is None:
        unopened_target_box_present = np.any(obj == OBJECT_TO_IDX["box"])
    else:
        unopened_target_box_present = np.any(
            (obj == OBJECT_TO_IDX["box"]) & (col == target_color_idx)
        )

    # Detect any visible key
    key_visible = np.any(obj == OBJECT_TO_IDX["key"])

    # Complete when the target box is no longer present (opened) and the key is no longer visible (picked up)
    return (not unopened_target_box_present) and (not key_visible)


def subtask_2_object(mission_str: str) -> Tuple[int, int]:
    COLOR_TO_IDX = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "purple": 3,
        "yellow": 4,
        "grey": 5,
    }
    OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,
    }

    object_idx = OBJECT_TO_IDX["key"]

    # Optionally extract a key color if present (default to -1 if not specified)
    m = re.search(
        r"pick up the\s+(?:(red|green|blue|purple|yellow|grey)\s+)?key\b",
        mission_str,
        flags=re.IGNORECASE,
    )
    color_idx = COLOR_TO_IDX[m.group(1).lower()] if (m and m.group(1)) else -1

    return object_idx, color_idx


def subtask_3_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Returns True iff Subtask 3 ("open the {door_color} door") is complete.
    Criterion: No target door (by color if specified) remains in a non-open state (closed or locked).
    """
    COLOR_TO_IDX = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "purple": 3,
        "yellow": 4,
        "grey": 5,
    }
    OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,
    }
    STATE_TO_IDX = {"open": 0, "closed": 1, "locked": 2}

    m = re.search(r"open the\s+(\w+)\s+door", mission_str, flags=re.IGNORECASE)
    door_color_idx = COLOR_TO_IDX.get(m.group(1).lower(), None) if m else None

    obj = state[..., 0].astype(np.int64)
    col = state[..., 1].astype(np.int64)
    stt = state[..., 2].astype(np.int64)

    door_mask = obj == OBJECT_TO_IDX["door"]
    if door_color_idx is not None:
        door_mask &= col == door_color_idx

    non_open_target_door = door_mask & (
        (stt == STATE_TO_IDX["closed"]) | (stt == STATE_TO_IDX["locked"])
    )
    return not np.any(non_open_target_door)


def subtask_3_object(mission_str: str) -> Tuple[int, int]:
    COLOR_TO_IDX = {
        "red": 0,
        "green": 1,
        "blue": 2,
        "purple": 3,
        "yellow": 4,
        "grey": 5,
    }
    OBJECT_TO_IDX = {
        "unseen": 0,
        "empty": 1,
        "wall": 2,
        "floor": 3,
        "door": 4,
        "key": 5,
        "ball": 6,
        "box": 7,
        "goal": 8,
        "lava": 9,
        "agent": 10,
    }

    object_idx = OBJECT_TO_IDX["door"]
    m = re.search(
        r"open the\s+(?:(red|green|blue|purple|yellow|grey)\s+)?door\b",
        mission_str,
        flags=re.IGNORECASE,
    )
    color_idx = COLOR_TO_IDX[m.group(1).lower()] if (m and m.group(1)) else -1
    return object_idx, color_idx
