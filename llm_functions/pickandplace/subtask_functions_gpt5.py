from typing import Tuple
import re


def subtask_1_complete(state: dict, mission_str: str) -> bool:
    """
    Determine if Subtask 1 (Pick Block) is complete: block is grasped by the gripper.

    Uses only the provided observation dict and relative quantities so it remains agnostic
    to absolute environment coordinates.

    References from the provided specification:
    - observation layout (25,):
      indices 0-2 gripper position "robot0:grip"; 3-5 block position "object0";
      6-8 block position relative to gripper (block - gripper);
      9-10 right/left gripper finger joint displacements;
      14-16 block linear velocity relative to gripper. [State Encoding table]
    - "The gripper can be opened or closed in order to perform the graspping operation..." [Mission]
    - "The action space ... last action that controls closing and opening of the gripper." [Action Space]

    Heuristic for a successful grasp:
    - The block is co-located with the gripper: small relative displacement norm (indices 6:9).
    - The block moves with the gripper: small relative velocity norm (indices 14:17).
    - The gripper is closed: small sum of finger displacements (indices 9 and 10).
    """
    import numpy as np

    # Accept either 'observation' (per spec) or a flat 'state' vector if provided by wrappers.
    obs_vec = None
    if isinstance(state, dict):
        if "observation" in state:
            obs_vec = state["observation"]
        elif "state" in state:
            # Some wrappers might place the same 25-dim under 'state'
            maybe = state["state"]
            if isinstance(maybe, (list, tuple, np.ndarray)):
                obs_vec = maybe
            elif isinstance(maybe, dict) and "observation" in maybe:
                obs_vec = maybe["observation"]

    if obs_vec is None:
        return False

    obs = np.asarray(obs_vec, dtype=float).reshape(-1)
    if obs.size < 25:
        return False

    # Extract relevant signals (indices per State Encoding table)
    rel_pos = obs[6:9]  # relative block position w.r.t. gripper [6:9]
    finger_r = obs[9]  # right finger joint displacement [9]
    finger_l = obs[10]  # left finger joint displacement [10]
    rel_vel = obs[14:17]  # relative block linear velocity w.r.t. gripper [14:17]

    # Robustness: handle NaNs
    if (
        not np.isfinite(rel_pos).all()
        or not np.isfinite(rel_vel).all()
        or not np.isfinite([finger_r, finger_l]).all()
    ):
        return False

    # Tunable, dimensioned thresholds (not absolute world coordinates)
    REL_POS_EPS = 0.03  # meters: co-location tolerance
    REL_VEL_EPS = 0.05  # m/s: relative motion tolerance
    FINGER_CLOSE_SUM_MAX = 0.06  # meters: small gap implies closed gripper

    # Compute conditions
    co_located = float(np.linalg.norm(rel_pos)) <= REL_POS_EPS
    co_moving = float(np.linalg.norm(rel_vel)) <= REL_VEL_EPS
    gripper_closed = float(abs(finger_r) + abs(finger_l)) <= FINGER_CLOSE_SUM_MAX

    # Subtask 1 (Pick) is complete if the block is grasped securely.
    return bool(co_located and co_moving and gripper_closed)


def subtask_1_object(mission_str: str) -> Tuple[int, int]:
    """
    Return (OBJECT_IDX, COLOR_IDX) for Subtask 1 (Pick the block).

    Rationale with citations:
    - The mission explicitly involves a block as the manipulated object:
      "pickup and move the block to the target location" [Mission Space, Paragraph 1]
      Therefore, Subtask 1 (Pick) targets the block.
    - Object and color indices are defined by the provided mappings:
      OBJECT_TO_IDX = {"block": 0, "target": 1, "agent": 2}
      COLOR_TO_IDX = {"green": 0, "yellow": 1, "lightblue": 2, "magenta": 3, "darkblue": 4, "red": 5}
      [Provided Mappings, Paragraph 1]

    Parsing notes:
    - Color defaults to -1 if not specified in the mission string.
    - If a color adjective appears immediately before "block" (e.g., "red block",
      "light blue block"), that color is selected.
    - Handles variants like "light blue" -> "lightblue" and "dark blue" -> "darkblue".
    """
    # Use provided globals if available; otherwise, fall back to the specified mappings.
    try:
        COLOR_MAP = COLOR_TO_IDX  # type: ignore[name-defined]
    except NameError:
        COLOR_MAP = {
            "green": 0,
            "yellow": 1,
            "lightblue": 2,
            "magenta": 3,
            "darkblue": 4,
            "red": 5,
        }

    try:
        OBJECT_MAP = OBJECT_TO_IDX  # type: ignore[name-defined]
    except NameError:
        OBJECT_MAP = {"block": 0, "target": 1, "agent": 2}

    # Subtask 1 is always the "block" (pick) in this mission.
    object_idx = OBJECT_MAP.get("block", 0)

    # Normalize the mission string for color extraction.
    text = (mission_str or "").lower()
    # Normalize separators and multiword colors.
    text = text.replace("-", " ")
    text = re.sub(r"\blight\s+blue\b", "lightblue", text)
    text = re.sub(r"\bdark\s+blue\b", "darkblue", text)
    text = re.sub(r"\s+", " ", text).strip()

    color_idx = -1

    # Prefer color adjective immediately preceding "block".
    color_words = list(COLOR_MAP.keys())
    color_alt_pattern = "|".join(re.escape(c) for c in color_words)
    pattern = re.compile(rf"\b(?:(?P<color>{color_alt_pattern})\s+)?block\b")
    match = pattern.search(text)
    if match and match.group("color"):
        color_idx = COLOR_MAP[match.group("color")]
    else:
        # Fallback: if exactly one color appears anywhere and "block" is mentioned, use it.
        present_colors = [
            c for c in color_words if re.search(rf"\b{re.escape(c)}\b", text)
        ]
        if "block" in text and len(present_colors) == 1:
            color_idx = COLOR_MAP[present_colors[0]]

    return object_idx, color_idx


def subtask_2_complete(state: dict, mission_str: str) -> bool:
    """
    Determine if Subtask 2 (Move and Hold At Target) is complete:
    - the block is at the desired target position, and
    - the block is stably held/placed (low absolute linear and angular velocities).

    Key quoted references to provided info:
    - "The task is also continuing which means that the robot has to maintain the block in the target position for an indefinite period of time." [Mission, Paragraph 1]
      -> We add stability checks (low velocities) in addition to position.
    - desired_goal: "(3,) ... the three cartesian coordinates of the desired final block position [x,y,z]." [desired_goal table]
    - achieved_goal: "(3,) ... the current block position [x,y,z]." [achieved_goal table]
    - observation layout (25,):
      indices 14-16 "Relative block linear velocity ... with respect to the gripper" and
      20-22 "End effector linear velocity ..." and
      17-19 "Block angular velocity ..." [State Encoding table rows 14-22,17-19]
      -> Absolute block linear velocity ≈ rel_block_vel + ee_vel.

    Notes:
    - No absolute coordinates are hardcoded; thresholds are generic tolerances.
    - Robust to different wrappers: accepts state["observation"] or state["state"],
      and uses achieved_goal/desired_goal if available.
    """
    import numpy as np

    # Extract flat observation vector
    obs_vec = None
    dg = None
    ag = None
    if isinstance(state, dict):
        # Goal-aware keys
        dg = state.get("desired_goal", None)
        ag = state.get("achieved_goal", None)

        # Observation vector may be under 'observation' or nested under 'state'
        if "observation" in state:
            obs_vec = state["observation"]
        elif "state" in state:
            maybe = state["state"]
            if isinstance(maybe, (list, tuple, np.ndarray)):
                obs_vec = maybe
            elif isinstance(maybe, dict) and "observation" in maybe:
                obs_vec = maybe["observation"]

    if obs_vec is None:
        return False

    obs = np.asarray(obs_vec, dtype=float).reshape(-1)
    if obs.size < 25:
        return False

    # Use achieved/desired goals if present; otherwise, fallback to obs for achieved.
    if ag is None:
        ag = obs[3:6]  # "Block x/y/z position" [State Encoding table rows 3-5]
    if dg is None:
        # Cannot evaluate distance to target without desired_goal
        return False

    ag = np.asarray(ag, dtype=float).reshape(-1)
    dg = np.asarray(dg, dtype=float).reshape(-1)
    if ag.size != 3 or dg.size != 3:
        return False
    if not np.isfinite(ag).all() or not np.isfinite(dg).all():
        return False

    # Extract velocities for stability tests
    rel_lin_vel = obs[
        14:17
    ]  # "Relative block linear velocity ... w.r.t. the gripper" [State Encoding table rows 14-16]
    ee_lin_vel = obs[
        20:23
    ]  # "End effector linear velocity ..." [State Encoding table rows 20-22]
    ang_vel = obs[
        17:20
    ]  # "Block angular velocity ..." [State Encoding table rows 17-19]

    if not (
        np.isfinite(rel_lin_vel).all()
        and np.isfinite(ee_lin_vel).all()
        and np.isfinite(ang_vel).all()
    ):
        return False

    # Compute absolute block linear velocity: v_block ≈ v_rel + v_ee
    abs_block_lin_vel = rel_lin_vel + ee_lin_vel

    # Thresholds (agnostic to absolute environment size)
    POS_EPS = 0.05  # meters: target proximity tolerance (Updated from 0.03 using re-prompting from verifier)
    LIN_VEL_EPS = 0.05  # m/s: stability tolerance for linear speed
    ANG_VEL_EPS = 0.50  # rad/s: stability tolerance for angular speed

    # Conditions
    at_target = float(np.linalg.norm(ag - dg)) <= POS_EPS
    lin_stable = float(np.linalg.norm(abs_block_lin_vel)) <= LIN_VEL_EPS
    ang_stable = float(np.linalg.norm(ang_vel)) <= ANG_VEL_EPS

    # Subtask 2 is complete if the block is at the target and stably held/placed.
    return bool(at_target and lin_stable and ang_stable)


def subtask_2_object(mission_str: str) -> Tuple[int, int]:
    """
    Return (OBJECT_IDX, COLOR_IDX) for Subtask 2 (Move/Hold at Target).

    Subtask 2 focuses on the target location as the object of interest.
    Color defaults to -1 unless the mission string explicitly specifies a color
    directly before "target" (or "target location" / "goal").
    """
    # Use provided globals if available; otherwise, define sensible defaults.
    try:
        COLOR_MAP = COLOR_TO_IDX  # type: ignore[name-defined]
    except NameError:
        COLOR_MAP = {
            "green": 0,
            "yellow": 1,
            "lightblue": 2,
            "magenta": 3,
            "darkblue": 4,
            "red": 5,
        }

    try:
        OBJECT_MAP = OBJECT_TO_IDX  # type: ignore[name-defined]
    except NameError:
        OBJECT_MAP = {"block": 0, "target": 1, "agent": 2}

    # Subtask 2 interacts with the target (place/hold at target).
    object_idx = OBJECT_MAP.get("target", 1)

    # Normalize the mission text.
    text = (mission_str or "").lower()
    text = text.replace("-", " ")
    # Normalize multiword colors.
    text = re.sub(r"\blight\s+blue\b", "lightblue", text)
    text = re.sub(r"\bdark\s+blue\b", "darkblue", text)
    text = re.sub(r"\s+", " ", text).strip()

    color_idx = -1
    color_words = list(COLOR_MAP.keys())
    color_alt_pattern = "|".join(re.escape(c) for c in color_words)

    # Look for a color adjective immediately before "target", "target location", or "goal".
    pattern = re.compile(
        rf"\b(?:(?P<color>{color_alt_pattern})\s+)?(?P<noun>target(?:\s+location)?|goal)\b"
    )
    match = pattern.search(text)
    if match and match.group("color"):
        color_idx = COLOR_MAP[match.group("color")]

    return object_idx, color_idx
