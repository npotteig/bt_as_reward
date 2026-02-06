import re
from typing import Tuple
from typing_extensions import Annotated
import numpy as np
import numpy.typing as npt

# Define a type alias for a 3D array of shape (N, M, 3)
ArrayNxMx3 = Annotated[npt.NDArray[np.float64], (None, None, 3)]


def subtask_1_bt_response_error(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Returns True if the key-room door (the {keyroom_color} room) is open (STATE=0), else False.
    Mission format: "get the {lockedroom_color} key from the {keyroom_color} room,
                     unlock the {door_color} door and go to the goal"
    """
    # Encodings (from State Encoding / OBJECT_TO_IDX / COLOR_TO_IDX / STATE_TO_IDX)
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

    # Parse mission string to get keyroom_color
    colors = r"(red|green|blue|purple|yellow|grey)"
    pat = rf"^\s*get the {colors} key from the {colors} room, unlock the {colors} door and go to the goal\s*$"
    m = re.match(pat, mission_str.strip().lower())
    if not m:
        # If mission doesn't match expected pattern, conservatively return False
        return False
    # m.groups(): (lockedroom_color, keyroom_color, door_color)
    keyroom_color = m.group(2)
    keyroom_color_idx = COLOR_TO_IDX[keyroom_color]

    # Check if any door of keyroom_color is open
    H, W, _ = state.shape
    for y in range(H):
        for x in range(W):
            obj_idx, color_idx, door_state = state[y, x]
            if (
                int(obj_idx) == OBJECT_TO_IDX["door"]
                and int(color_idx) == keyroom_color_idx
                and int(door_state) == STATE_TO_IDX["open"]
            ):
                return True
    return False


def subtask_1_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Subtask 1: Open the key-room door (door of color {keyroom_color}).
    Returns True if:
      - At least one door of color {keyroom_color} is open (STATE=0), or
      - No door of color {keyroom_color} exists in the state (treat as complete per guidance).
    Otherwise, returns False.
    """
    # Encodings per State Encoding
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

    # Parse mission to extract keyroom_color
    s = mission_str.strip().lower()
    colors = r"(red|green|blue|purple|yellow|grey)"
    pat = rf"^\s*get the {colors} key from the {colors} room, unlock the {colors} door and go to the goal\s*$"
    m = re.match(pat, s)
    if not m:
        return False
    keyroom_color = m.group(2)
    keyroom_color_idx = COLOR_TO_IDX[keyroom_color]

    H, W, _ = state.shape
    found_keyroom_door = False
    for y in range(H):
        for x in range(W):
            obj_idx, color_idx, door_state = state[y, x]
            if (
                int(obj_idx) == OBJECT_TO_IDX["door"]
                and int(color_idx) == keyroom_color_idx
            ):
                found_keyroom_door = True
                if int(door_state) == STATE_TO_IDX["open"]:
                    return True  # explicit success when an identified door is open

    # If no such door is present in the grid, consider subtask complete as per guidance
    if not found_keyroom_door:
        return True

    return False


def subtask_1_object(mission_str: str) -> Tuple[int, int]:
    """
    Returns (OBJECT_IDX, COLOR_IDX) for the object of interest in Subtask 1.
    Subtask 1: open the key-room door (a door of color {keyroom_color}).

    - Mission format: "get the {lockedroom_color} key from the {keyroom_color} room,
                       unlock the {door_color} door and go to the goal" [Mission Space]
    - OBJECT_TO_IDX['door'] == 4 [OBJECT_TO_IDX]
    - COLOR_TO_IDX mapping: red=0, green=1, blue=2, purple=3, yellow=4, grey=5 [COLOR_TO_IDX]

    If the mission string does not match the expected pattern, returns color_idx = -1.
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

    s = mission_str.strip().lower()
    colors = r"(red|green|blue|purple|yellow|grey)"
    pat = rf"get the {colors} key from the {colors} room, unlock the {colors} door and go to the goal"
    m = re.fullmatch(pat, s)

    color_idx = -1
    if m:
        # groups: (lockedroom_color, keyroom_color, door_color)
        keyroom_color = m.group(2)
        color_idx = COLOR_TO_IDX.get(keyroom_color, -1)

    return OBJECT_TO_IDX["door"], color_idx


def subtask_2_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Returns True if the target key (lockedroom_color) has been picked up,
    approximated by absence of any key of that color on the grid.
    Mission format: "get the {lockedroom_color} key from the {keyroom_color} room,
                     unlock the {door_color} door and go to the goal"
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

    s = mission_str.strip().lower()
    colors = r"(red|green|blue|purple|yellow|grey)"
    pat = rf"^\s*get the {colors} key from the {colors} room, unlock the {colors} door and go to the goal\s*$"
    m = re.match(pat, s)
    if not m:
        return False

    lockedroom_color = m.group(1)
    key_color_idx = COLOR_TO_IDX[lockedroom_color]
    key_obj_idx = OBJECT_TO_IDX["key"]

    H, W, _ = state.shape
    for y in range(H):
        for x in range(W):
            obj_idx, color_idx, _ = state[y, x]
            if int(obj_idx) == key_obj_idx and int(color_idx) == key_color_idx:
                return False  # key still present on grid
    return True  # key no longer on grid => considered picked up


def subtask_2_object(mission_str: str) -> Tuple[int, int]:
    """
    Subtask 2 target: the key of color {lockedroom_color}.
    Mission format: "get the {lockedroom_color} key from the {keyroom_color} room, unlock the {door_color} door and go to the goal" [Mission Space].
    OBJECT_TO_IDX['key'] = 5; COLOR_TO_IDX: red=0, green=1, blue=2, purple=3, yellow=4, grey=5 [State Encoding].
    Returns (OBJECT_IDX, COLOR_IDX); COLOR_IDX=-1 if mission doesn't match.
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

    s = mission_str.strip().lower()
    colors = r"(red|green|blue|purple|yellow|grey)"
    pat = rf"get the {colors} key from the {colors} room, unlock the {colors} door and go to the goal"
    m = re.fullmatch(pat, s)

    color_idx = -1
    if m:
        lockedroom_color = m.group(1)
        color_idx = COLOR_TO_IDX.get(lockedroom_color, -1)

    return OBJECT_TO_IDX["key"], color_idx


def subtask_3_bt_response_error(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Returns True if the target locked door (door_color) is open (STATE=0), else False.
    Mission format: "get the {lockedroom_color} key from the {keyroom_color} room,
                     unlock the {door_color} door and go to the goal"
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

    s = mission_str.strip().lower()
    colors = r"(red|green|blue|purple|yellow|grey)"
    pat = rf"^\s*get the {colors} key from the {colors} room, unlock the {colors} door and go to the goal\s*$"
    m = re.match(pat, s)
    if not m:
        return False

    door_color_idx = COLOR_TO_IDX[m.group(3)]
    H, W, _ = state.shape
    for y in range(H):
        for x in range(W):
            obj_idx, color_idx, st = state[y, x]
            if (
                int(obj_idx) == OBJECT_TO_IDX["door"]
                and int(color_idx) == door_color_idx
                and int(st) == STATE_TO_IDX["open"]
            ):
                return True
    return False


def subtask_3_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Subtask 3: Unlock/open the locked door (door_color).
    Returns True if:
      - At least one door of color {door_color} is open (STATE=0), or
      - No door of color {door_color} exists in the current observation
        (treat as complete when object is not observed per guidance).
    Otherwise, returns False.
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

    s = mission_str.strip().lower()
    colors = r"(red|green|blue|purple|yellow|grey)"
    pat = rf"^\s*get the {colors} key from the {colors} room, unlock the {colors} door and go to the goal\s*$"
    m = re.match(pat, s)
    if not m:
        return False

    door_color = m.group(3)
    door_color_idx = COLOR_TO_IDX[door_color]

    H, W, _ = state.shape
    found_target_door = False
    for y in range(H):
        for x in range(W):
            obj_idx, color_idx, st = state[y, x]
            if (
                int(obj_idx) == OBJECT_TO_IDX["door"]
                and int(color_idx) == door_color_idx
            ):
                found_target_door = True
                if int(st) == STATE_TO_IDX["open"]:
                    return True  # success: target door is open

    # If the target door is not observed at all, treat as complete per guidance
    if not found_target_door:
        return True

    return False


def subtask_3_object(mission_str: str) -> Tuple[int, int]:
    """
    Subtask 3 target: the door of color {door_color}.
    Returns (OBJECT_IDX, COLOR_IDX); COLOR_IDX = -1 if mission doesn't match.
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

    s = mission_str.strip().lower()
    colors = r"(red|green|blue|purple|yellow|grey)"
    pat = rf"get the {colors} key from the {colors} room, unlock the {colors} door and go to the goal"
    m = re.fullmatch(pat, s)

    color_idx = -1
    if m:
        door_color = m.group(3)
        color_idx = COLOR_TO_IDX.get(door_color, -1)

    return OBJECT_TO_IDX["door"], color_idx


def subtask_4_complete(state: ArrayNxMx3, mission_str: str) -> bool:
    """
    Returns True if the agent has reached the goal.
    Basis: mission requires to "go to the goal" [Mission Space] and tiles encode OBJECT_IDX with goal=8, agent=10 [State Encoding].

    Heuristic (agnostic to map size and coordinates):
    - If both agent and goal layers can coexist, success when any agent cell coincides with a goal cell.
    - If single-layer encoding replaces the goal with the agent on contact, success when no goal tile remains.
    """
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

    H, W, _ = state.shape
    goal_positions = set()
    agent_positions = set()

    for y in range(H):
        for x in range(W):
            obj_idx = int(state[y, x, 0])
            if obj_idx == OBJECT_TO_IDX["goal"]:
                goal_positions.add((x, y))
            elif obj_idx == OBJECT_TO_IDX["agent"]:
                agent_positions.add((x, y))

    # Case 1: layered/dual representation (unlikely with single tuple, but safe to check)
    if goal_positions and (agent_positions & goal_positions):
        return True
    # Case 2: single-layer replacement on reach — goal disappears when agent occupies it
    if not goal_positions and agent_positions:
        return True

    return False


def subtask_4_object(mission_str: str) -> Tuple[int, int]:
    """
    Subtask 4 target is the goal; it has no color.
    Returns (OBJECT_IDX, COLOR_IDX) = (goal, -1).
    """
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
    return OBJECT_TO_IDX["goal"], -1
