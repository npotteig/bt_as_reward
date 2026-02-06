from typing import Tuple


def subtask_1_complete(state: dict, mission_str: str) -> bool:
    """
    Determine if Subtask 1 (Grasp Block) is complete.

    Success criteria:
      - Proximity: block is very close to the gripper (using relative block-to-gripper position).
      - Gripper closed: both finger displacements indicate a closed state.
      - Stability: block relative velocity magnitude w.r.t. gripper is low (indicating a stable grasp).

    Notes from the provided specification:
      - "The gripper can be opened or closed in order to perform the grasping operation of pick and move." [Mission: Pick And Move 2]
      - The observation dictionary encodes the necessary signals:
        * "Relative block x/y/z position ... Equals to xblock - xgripper" at indices 6–8 [State Encoding, observation rows 6–8].
        * "Joint displacement of the right/left gripper finger" at indices 9–10 [State Encoding, observation rows 9–10].
        * "Relative block linear velocity in x/y/z direction ... with respect to the gripper" at indices 14–16 [State Encoding, observation rows 14–16].

    The function remains agnostic to environment size by not using any absolute target coordinates.
    Thresholds used are local to the grasp interaction (contact, closure, and stability), not global scene extents.
    """
    import numpy as np

    # Basic validation of expected keys and shapes
    if not isinstance(state, dict) or "observation" not in state:
        return False
    observation = state.get("observation", None)
    if (
        observation is None
        or not hasattr(observation, "__len__")
        or len(observation) < 17
    ):
        return False

    # Extract signals based on the documented state encoding
    # Relative block position w.r.t. gripper (x,y,z): indices 6:9 [State Encoding, rows 6–8]
    rel_block_to_gripper = np.asarray(observation[6:9], dtype=float)
    # Finger displacements (right, left): indices 9 and 10 [State Encoding, rows 9–10]
    right_disp = float(observation[9])
    left_disp = float(observation[10])
    # Relative block velocity w.r.t. gripper (x,y,z): indices 14:17 [State Encoding, rows 14–16]
    rel_block_vel = np.asarray(observation[14:17], dtype=float)

    # Compute norms for proximity and stability checks
    dist = float(np.linalg.norm(rel_block_to_gripper))
    relvel_mag = float(np.linalg.norm(rel_block_vel))

    # Tunable local thresholds (interaction-scale; not absolute coordinates)
    CONTACT_DIST_THRESH = 0.03  # meters: near-contact proximity
    GRIPPER_CLOSED_THRESH = (
        0.03  # meters: both finger displacements should be small to indicate closure
    )
    RELVEL_THRESH = 0.01  # m/s: low relative velocity indicates a stable grasp

    fingers_closed = (right_disp <= GRIPPER_CLOSED_THRESH) and (
        left_disp <= GRIPPER_CLOSED_THRESH
    )
    proximity_ok = dist <= CONTACT_DIST_THRESH
    stability_ok = relvel_mag <= RELVEL_THRESH

    return bool(proximity_ok and fingers_closed and stability_ok)

def subtask_1_reprompt_failure(state: dict, mission_str: str) -> bool:
    """
    Determine if Subtask 1 (Grasp Block) is complete.

    Success criteria:
      - Proximity: block is very close to the gripper (using relative block-to-gripper position).
      - Gripper closed: both finger displacements indicate a closed state.
      - Stability: block relative velocity magnitude w.r.t. gripper is low (indicating a stable grasp).

    Notes from the provided specification:
      - "The gripper can be opened or closed in order to perform the grasping operation of pick and move." [Mission: Pick And Move 2]
      - The observation dictionary encodes the necessary signals:
        * "Relative block x/y/z position ... Equals to xblock - xgripper" at indices 6–8 [State Encoding, observation rows 6–8].
        * "Joint displacement of the right/left gripper finger" at indices 9–10 [State Encoding, observation rows 9–10].
        * "Relative block linear velocity in x/y/z direction ... with respect to the gripper" at indices 14–16 [State Encoding, observation rows 14–16].

    The function remains agnostic to environment size by not using any absolute target coordinates.
    Thresholds used are local to the grasp interaction (contact, closure, and stability), not global scene extents.
    """
    import numpy as np

    # Basic validation of expected keys and shapes
    if not isinstance(state, dict) or "observation" not in state:
        return False
    observation = state.get("observation", None)
    if observation is None or not hasattr(observation, "__len__") or len(observation) < 17:
        return False

    # Extract signals based on the documented state encoding
    # Relative block position w.r.t. gripper (x,y,z): indices 6:9 [State Encoding, rows 6–8]
    rel_block_to_gripper = np.asarray(observation[6:9], dtype=float)
    # Finger displacements (right, left): indices 9 and 10 [State Encoding, rows 9–10]
    right_disp = float(observation[9])
    left_disp = float(observation[10])
    # Relative block velocity w.r.t. gripper (x,y,z): indices 14:17 [State Encoding, rows 14–16]
    rel_block_vel = np.asarray(observation[14:17], dtype=float)

    # Compute norms for proximity and stability checks
    dist = float(np.linalg.norm(rel_block_to_gripper))
    relvel_mag = float(np.linalg.norm(rel_block_vel))

    # Tunable local thresholds (interaction-scale; not absolute coordinates)
    CONTACT_DIST_THRESH = 0.02       # meters: near-contact proximity
    GRIPPER_CLOSED_THRESH = 0.01     # meters: both finger displacements should be small to indicate closure
    RELVEL_THRESH = 0.01             # m/s: low relative velocity indicates a stable grasp

    fingers_closed = (right_disp <= GRIPPER_CLOSED_THRESH) and (left_disp <= GRIPPER_CLOSED_THRESH)
    proximity_ok = dist <= CONTACT_DIST_THRESH
    stability_ok = relvel_mag <= RELVEL_THRESH

    return bool(proximity_ok and fingers_closed and stability_ok)


def subtask_1_object(mission_str: str) -> Tuple[int, int]:
    """
    Return the OBJECT_IDX and COLOR_IDX for Subtask 1 (object to grasp).

    Citations:
    - "pick up and move the block to the {color} target location" [Mission Space, Paragraph 1]
    - OBJECT_TO_IDX = {"block": 0, "target": 1, "agent": 2} [Mapping, Paragraph 1]
    - COLOR_TO_IDX = {"green": 0, "yellow": 1, "lightblue": 2, "magenta": 3, "darkblue": 4, "red": 5} [Mapping, Paragraph 2]

    For Subtask 1, the object of interest is the block. The mission color token
    refers to the target, not the block, so the block's color is typically unspecified.
    Default color_idx to -1 when no explicit block color is present in the mission string.
    """
    OBJECT_TO_IDX = {
        "block": 0,
        "target": 1,
        "agent": 2,
    }
    COLOR_TO_IDX = {
        "green": 0,
        "yellow": 1,
        "lightblue": 2,
        "magenta": 3,
        "darkblue": 4,
        "red": 5,
    }

    s = (mission_str or "").lower()

    # Subtask 1 always targets the block for grasping.
    obj_idx = OBJECT_TO_IDX["block"]

    # Attempt to infer a color adjective applied to the block specifically (rare in this mission space).
    # We avoid picking up the target's color by requiring the color to be adjacent to "block".
    color_idx = -1
    for color, idx in COLOR_TO_IDX.items():
        # Match patterns like "red block" or "block red"
        if f"{color} block" in s or f"block {color}" in s:
            color_idx = idx
            break

    return obj_idx, color_idx


def subtask_2_complete(state: dict, mission_str: str) -> bool:
    """
    Determine if Subtask 2 (Move Block to {color} Target, keep gripper closed) is complete.

    Success requires:
      - The block position (achieved_goal) is close to the selected target position (desired_goal entry for the mission's color).
      - The gripper fingers are closed (or near-closed) at the target.

    Relevant provided information (quoted):
      - Mission Space: "pick up and move the block to the {color} target location" [Mission Space, Paragraph 1].
      - desired_goal encoding:
        * "Green final goal block position in the x coordinate" and corresponding y, z [State Encoding, desired_goal table rows 0–2].
        * "Yellow final goal block position in the x coordinate" and corresponding y, z [State Encoding, desired_goal table rows 3–5].
      - achieved_goal encoding:
        * "Current block position in the x/y/z coordinate" [State Encoding, achieved_goal table rows 0–2].
      - observation indices for gripper closure:
        * "Joint displacement of the right gripper finger" (index 9)
        * "Joint displacement of the left gripper finger" (index 10)
        [State Encoding, observation table rows 9–10].
      - Gripper use at target:
        * "The gripper should not be opened at the target, only moved to the target to complete the mission." [Mission, Paragraph 1].

    The function remains agnostic to environment size: it uses local interaction thresholds rather than any absolute coordinates.
    """
    import numpy as np

    # Basic validation of expected keys and shapes
    if not isinstance(state, dict):
        return False
    if (
        "desired_goal" not in state
        or "achieved_goal" not in state
        or "observation" not in state
    ):
        return False

    desired_goal = state.get("desired_goal")
    achieved_goal = state.get("achieved_goal")
    observation = state.get("observation")

    # Validate array lengths based on provided encodings
    if (
        desired_goal is None
        or not hasattr(desired_goal, "__len__")
        or len(desired_goal) < 6
    ):
        # Expecting both green (0:3) and yellow (3:6) targets per provided table
        return False
    if (
        achieved_goal is None
        or not hasattr(achieved_goal, "__len__")
        or len(achieved_goal) < 3
    ):
        return False
    if (
        observation is None
        or not hasattr(observation, "__len__")
        or len(observation) < 11
    ):
        return False

    # Parse mission color (supports at least 'green' and 'yellow' as per desired_goal entries)
    s = (mission_str or "").lower()
    mission_color = None
    if "yellow" in s:
        mission_color = "yellow"
    elif "green" in s:
        mission_color = "green"
    else:
        # No valid color specified to choose the target
        return False

    # Select target position according to mission color
    desired_goal = np.asarray(desired_goal, dtype=float)
    if mission_color == "green":
        target_pos = desired_goal[0:3]
    else:  # "yellow"
        target_pos = desired_goal[3:6]

    # Current block position from achieved_goal
    block_pos = np.asarray(achieved_goal[0:3], dtype=float)

    # Gripper finger displacements (closure proxy)
    right_disp = float(observation[9])
    left_disp = float(observation[10])

    # Local interaction thresholds (agnostic to global workspace scale)
    TARGET_DIST_THRESH = 0.05  # meters: tolerance for block being at the target
    GRIPPER_CLOSED_THRESH = 0.03  # meters: near-closed finger displacements at target

    # Evaluate proximity to target and gripper closure
    dist_to_target = float(np.linalg.norm(block_pos - target_pos))
    proximity_ok = dist_to_target <= TARGET_DIST_THRESH
    fingers_closed = (right_disp <= GRIPPER_CLOSED_THRESH) and (
        left_disp <= GRIPPER_CLOSED_THRESH
    )

    return bool(proximity_ok and fingers_closed)


def subtask_2_object(mission_str: str) -> Tuple[int, int]:
    """
    Return the OBJECT_IDX and COLOR_IDX for Subtask 2 (move the block to the {color} target).

    Rationale with citations:
    - The mission specifies the colored target: "pick up and move the block to the {color} target location"
      [Mission Space, Paragraph 1]. Therefore, the object of interest in Subtask 2 is the target.
    - The provided mappings state: OBJECT_TO_IDX = {"block": 0, "target": 1, "agent": 2}
      and COLOR_TO_IDX = {"green": 0, "yellow": 1, "lightblue": 2, "magenta": 3, "darkblue": 4, "red": 5}
      [Mappings, Code Block after State Encoding].
    - desired_goal encodes two targets (green indices 0–2, yellow indices 3–5)
      [State Encoding, desired_goal table rows 0–5]. We still parse all known colors from COLOR_TO_IDX,
      defaulting to -1 if none are present in the mission string.

    Returns:
      (obj_idx, color_idx): obj_idx corresponds to the target, color_idx corresponds to the target color
      specified in the mission string, or -1 if no color token is found.
    """
    OBJECT_TO_IDX = {
        "block": 0,
        "target": 1,
        "agent": 2,
    }
    COLOR_TO_IDX = {
        "green": 0,
        "yellow": 1,
        "lightblue": 2,
        "magenta": 3,
        "darkblue": 4,
        "red": 5,
    }

    s = (mission_str or "").lower()

    # Subtask 2 focuses on the target (colored by the mission).
    obj_idx = OBJECT_TO_IDX["target"]

    # Find the first occurring known color token in the mission string.
    # If none found, default to -1.
    first_pos = None
    color_idx = -1
    for color, idx in COLOR_TO_IDX.items():
        pos = s.find(color)
        if pos != -1 and (first_pos is None or pos < first_pos):
            first_pos = pos
            color_idx = idx

    return obj_idx, color_idx
