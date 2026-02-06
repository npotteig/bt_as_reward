MINIGRID_COLOR_TO_IDX = {
    "red": 0,
    "green": 1,
    "blue": 2,
    "purple": 3,
    "yellow": 4,
    "grey": 5,
}
MINIGRID_IDX_TO_COLOR = {v: k for k, v in MINIGRID_COLOR_TO_IDX.items()}

MINIGRID_OBJECT_TO_IDX = {
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
MINIGRID_IDX_TO_OBJECT = {v: k for k, v in MINIGRID_OBJECT_TO_IDX.items()}

MINIGRID_STATE_TO_IDX = {
    "open": 0,
    "closed": 1,
    "locked": 2,
}
MINIGRID_IDX_TO_STATE = {v: k for k, v in MINIGRID_STATE_TO_IDX.items()}

MUJOCO_COLOR_TO_IDX = {
    "green": 0,
    "yellow": 1,
}
MUJOCO_IDX_TO_COLOR = {v: k for k, v in MUJOCO_COLOR_TO_IDX.items()}

MUJOCO_OBJECT_TO_IDX = {"block": 0, "target": 1, "agent": 2}
MUJOCO_IDX_TO_OBJECT = {v: k for k, v in MUJOCO_OBJECT_TO_IDX.items()}
