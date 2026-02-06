import minigrid
import numpy as np
from bt_as_reward.constants import MINIGRID_COLOR_TO_IDX, MINIGRID_IDX_TO_COLOR, MINIGRID_OBJECT_TO_IDX
import gymnasium as gym
import z3
import re

from bt_as_reward.verifiers.minigrid_z3 import verify_spec
from bt_as_reward.envs.lockedroom_small import LockedRoomSmallEnv
from bt_as_reward.envs.drone_supplier import DroneSupplierSmallEnv
from minigrid.core.world_object import Door, Key
from minigrid.wrappers import FullyObsWrapper

T = 25  # Time horizon
N = 10

def subtask_lockedroom_key_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    # Check mission_keys[0] color key has been picked up
    # Cannot index using z3 variables directly, so we create conditions for each color and combine them
    conditions = []
    for color_idx in MINIGRID_IDX_TO_COLOR.keys():
        conditions.append(
            z3.Implies(
                mission_keys[0] == color_idx,
                z3.And(
                    key_x_colors[color_idx] == -1,  # key has been picked up
                    key_y_colors[color_idx] == -1
                )
            )
        )
    
    return z3.And(conditions)

def near_lockedroom_key_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    conditions = []
    for color_idx in MINIGRID_IDX_TO_COLOR.keys():
        conditions.append(
            z3.Implies(
                mission_keys[0] == color_idx,
                z3.Or(
                    z3.And(x == key_x_colors[color_idx] - 1, y == key_y_colors[color_idx]),  # left
                    z3.And(x == key_x_colors[color_idx] + 1, y == key_y_colors[color_idx]),  # right
                    z3.And(x == key_x_colors[color_idx],     y == key_y_colors[color_idx] - 1),  # down
                    z3.And(x == key_x_colors[color_idx],     y == key_y_colors[color_idx] + 1)   # up
                )
            )
        )
    
    return z3.And(conditions)

def subtask_lockedroom_first_door_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    # Check mission_doors[1] color door is open or occluded
    conditions = []
    for color_idx in MINIGRID_IDX_TO_COLOR.keys():
        conditions.append(
            z3.Implies(
                mission_doors[0] == color_idx,
                z3.Or(door_state_colors[color_idx] == 0, door_state_colors[color_idx] == -1)  # door is open or occluded
            )
        )
    
    return z3.And(conditions)

def near_lockedroom_first_door_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    conditions = []
    for color_idx in MINIGRID_IDX_TO_COLOR.keys():
        conditions.append(
            z3.Implies(
                mission_doors[0] == color_idx,
                z3.Or(
                    z3.And(x == door_x_colors[color_idx] - 1, y == door_y_colors[color_idx]),  # left
                    z3.And(x == door_x_colors[color_idx] + 1, y == door_y_colors[color_idx]),  # right
                    z3.And(x == door_x_colors[color_idx],     y == door_y_colors[color_idx] - 1),  # down
                    z3.And(x == door_x_colors[color_idx],     y == door_y_colors[color_idx] + 1)   # up
                )
            )
        )
    
    return z3.And(conditions)

def subtask_lockedroom_second_door_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    # Check mission_doors[1] color door is open or occluded
    conditions = []
    for color_idx in MINIGRID_IDX_TO_COLOR.keys():
        conditions.append(
            z3.Implies(
                mission_doors[1] == color_idx,
                z3.Or(door_state_colors[color_idx] == 0, door_state_colors[color_idx] == -1)  # door is open or occluded
            )
        )
    
    return z3.And(conditions)

def near_lockedroom_second_door_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    conditions = []
    for color_idx in MINIGRID_IDX_TO_COLOR.keys():
        conditions.append(
            z3.Implies(
                mission_doors[1] == color_idx,
                z3.Or(
                    z3.And(x == door_x_colors[color_idx] - 1, y == door_y_colors[color_idx]),  # left
                    z3.And(x == door_x_colors[color_idx] + 1, y == door_y_colors[color_idx]),  # right
                    z3.And(x == door_x_colors[color_idx],     y == door_y_colors[color_idx] - 1),  # down
                    z3.And(x == door_x_colors[color_idx],     y == door_y_colors[color_idx] + 1)   # up
                )
            )
        )
    
    return z3.And(conditions)

def subtask_lockedroom_goal_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    return z3.Or(z3.And(
                    x == goal_x,
                    y == goal_y
                ),
                goal_x == -1,  # goal is occluded
    )

def near_lockedroom_goal_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    return z3.Or(
        z3.And(x == goal_x - 1, y == goal_y),  # left
        z3.And(x == goal_x + 1, y == goal_y),  # right
        z3.And(x == goal_x,     y == goal_y - 1),  # down
        z3.And(x == goal_x,     y == goal_y + 1)   # up
    )

def subtask_doorkey_key_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    return z3.And(
        key_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]] == -1,  # key has been picked up
        key_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]] == -1
    )
    
def near_doorkey_key_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    return z3.Or(
        z3.And(x == key_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]] - 1, y == key_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]]),  # left
        z3.And(x == key_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]] + 1, y == key_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]]),  # right
        z3.And(x == key_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]],     y == key_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]] - 1),  # down
        z3.And(x == key_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]],     y == key_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]] + 1)   # up
    )

def subtask_doorkey_door_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    return z3.Or(door_state_colors[MINIGRID_COLOR_TO_IDX["yellow"]] == 0, door_state_colors[MINIGRID_COLOR_TO_IDX["yellow"]] == -1)  # door is open or occluded

def near_doorkey_door_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    return z3.Or(
        z3.And(x == door_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]] - 1, y == door_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]]),  # left
        z3.And(x == door_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]] + 1, y == door_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]]),  # right
        z3.And(x == door_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]],     y == door_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]] - 1),  # down
        z3.And(x == door_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]],     y == door_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]] + 1)   # up
    )

def subtask_doorkey_goal_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    return z3.Or(z3.And(
                    x == goal_x,
                    y == goal_y
                ),
                goal_x == -1,  # goal is occluded
    )

def near_doorkey_goal_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    return z3.Or(
        z3.And(x == goal_x - 1, y == goal_y),  # left
        z3.And(x == goal_x + 1, y == goal_y),  # right
        z3.And(x == goal_x,     y == goal_y - 1),  # down
        z3.And(x == goal_x,     y == goal_y + 1)   # up
    )

def subtask_dronesupplier_box_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    conditions = []
    for box_idx in range(len(mission_boxes)):
        conditions.append(
            z3.Implies(
                mission_boxes[box_idx] == box_idx,  # box is required
                z3.And(
                    box_x_colors[box_idx] == -1,  # box has been picked up
                    box_y_colors[box_idx] == -1
                )
            )
        )
    return z3.And(conditions)

def near_dronesupplier_box_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    conditions = []
    for box_idx in range(len(mission_boxes)):
        conditions.append(
            z3.Implies(
                mission_boxes[box_idx] == box_idx,  # box is required
                z3.Or(
                    z3.And(x == box_x_colors[box_idx] - 1, y == box_y_colors[box_idx]),  # left
                    z3.And(x == box_x_colors[box_idx] + 1, y == box_y_colors[box_idx]),  # right
                    z3.And(x == box_x_colors[box_idx],     y == box_y_colors[box_idx] - 1),  # down
                    z3.And(x == box_x_colors[box_idx],     y == box_y_colors[box_idx] + 1)   # up
                )
            )
        )
    return z3.And(conditions)

def subtask_dronesupplier_key_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    conditions = []
    for key_idx in range(len(mission_keys)):
        for box_idx in range(len(mission_boxes)):
            conditions.append(
                z3.Implies(
                    z3.And(
                        mission_keys[key_idx] == key_idx,
                        mission_boxes[box_idx] == box_idx
                    ),
                    z3.And(
                        key_x_colors[key_idx] == -1,  # key has been picked up
                        key_y_colors[key_idx] == -1,
                        box_x_colors[box_idx] == -1,  # box has been destroyed
                        box_y_colors[box_idx] == -1
                    )
                )
            )
    return z3.And(conditions)

def near_dronesupplier_key_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    conditions = []
    for key_idx in range(len(mission_keys)):
        conditions.append(
            z3.Implies(
                mission_keys[key_idx] == key_idx,
                z3.Or(
                    z3.And(x == key_x_colors[key_idx] - 1, y == key_y_colors[key_idx]),  # left
                    z3.And(x == key_x_colors[key_idx] + 1, y == key_y_colors[key_idx]),  # right
                    z3.And(x == key_x_colors[key_idx],     y == key_y_colors[key_idx] - 1),  # down
                    z3.And(x == key_x_colors[key_idx],     y == key_y_colors[key_idx] + 1)   # up
                )
            )
        )
    return z3.And(conditions)

def subtask_dronesupplier_door_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    conditions = []
    for door_idx in range(len(mission_doors)):
        conditions.append(
            z3.Implies(
                mission_doors[door_idx] == door_idx,
                z3.Or(door_state_colors[door_idx] == 0, door_state_colors[door_idx] == -1)  # door is open or occluded
            )
        )
    
    return z3.And(conditions)

def near_dronesupplier_door_condition(state, mission):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y = state
    mission_keys, mission_doors, mission_boxes = mission
    
    conditions = []
    for door_idx in range(len(mission_doors)):
        conditions.append(
            z3.Implies(
                mission_doors[door_idx] == door_idx,
                z3.Or(
                    z3.And(x == door_x_colors[door_idx] - 1, y == door_y_colors[door_idx]),  # left
                    z3.And(x == door_x_colors[door_idx] + 1, y == door_y_colors[door_idx]),  # right
                    z3.And(x == door_x_colors[door_idx],     y == door_y_colors[door_idx] - 1),  # down
                    z3.And(x == door_x_colors[door_idx],     y == door_y_colors[door_idx] + 1)   # up
                )
            )
        )
    
    return z3.And(conditions)

if __name__ == "__main__":
    import time
    start_time = time.time()
    
    # ----
    # Test DoorKey environment specifications
    # ----
    
    # out, res, _ = verify_spec("subtask_completion", subtask_complete_func=subtask_doorkey_key_condition)
    # print(out)
    # Negative Subtask Completion Check
    # restricted_init_states = []
    # for i in range(N):
    #     print(f"Negative Subtask Completion Test {i}")
    #     _, res, init_state = verify_spec(
    #         "subtask_completion", 
    #         subtask_complete_func=subtask_doorkey_key_condition,
    #         negative_check=True
    #     )
    #     if not res:
    #         print("  Negative Subtask Completion failed.")
    #         break
    #     restricted_init_states.append(init_state)
    #     print("  Negative Subtask Completion holds.")
        
    # out, res, _ = verify_spec(
    #     "near_object", 
    #     subtask_complete_func=subtask_doorkey_key_condition,
    #     near_object_func=near_doorkey_key_condition
    # )
    # print(out)
    # Negative Near Object Check
    # restricted_init_states = []
    # for i in range(N):
    #     print(f"Negative Near Object Test {i}")
    #     _, res, init_state = verify_spec(
    #         "near_object", 
    #         subtask_complete_func=subtask_doorkey_key_condition,
    #         near_object_func=near_doorkey_key_condition,
    #         negative_check=True,
    #         restricted_initial_states=restricted_init_states,
    #     )
    #     if not res:
    #         print("  Negative Near Object failed.")
    #         break
    #     restricted_init_states.append(init_state)
    #     print("  Negative Near Object holds.")
        
    
    # out, res, _ = verify_spec("subtask_completion", subtask_complete_func=subtask_doorkey_door_condition)
    # print(out)
    # out, res, _ = verify_spec(
    #     "near_object", 
    #     subtask_complete_func=subtask_doorkey_door_condition,
    #     near_object_func=near_doorkey_door_condition
    # )
    # print(out)
    
    # out, res, _ = verify_spec("subtask_completion", subtask_complete_func=subtask_doorkey_goal_condition)
    # print(out)
    # out, res, _ = verify_spec(
    #     "near_object", 
    #     subtask_complete_func=subtask_doorkey_goal_condition,
    #     near_object_func=near_doorkey_goal_condition
    # )
    # print(out)
    
    # restricted_init_states = []
    # for i in range(N):
    #     print(f"Test {i}")
    #     _, res, init_state = verify_spec(
    #         "persistence", 
    #         subtask_complete_func=[subtask_doorkey_key_condition, subtask_doorkey_door_condition, subtask_doorkey_goal_condition],
    #         restricted_initial_states=restricted_init_states,
    #     )
    #     if not res:
    #         print("  Persistence failed.")
    #         # Check which function failed
    #         for func in [subtask_doorkey_key_condition, subtask_doorkey_door_condition, subtask_doorkey_goal_condition]:
    #             _, res_func, _ = verify_spec(
    #                 "persistence", 
    #                 subtask_complete_func=[func],
    #                 restricted_initial_states=restricted_init_states,
    #             )
    #             if not res_func:
    #                 print(f"  Persistence failed for function: {func.__name__}")
    #                 break
    #         break
    #     restricted_init_states.append(init_state)
    #     print("  Persistence holds.")
    
    
    # ----
    # Test LockedRoom environment specifications
    # ----
    
    # out, res, _ = verify_spec("subtask_completion", 
    #             subtask_complete_func=subtask_lockedroom_key_condition,
    #             env_name="MiniGrid-LockedRoom-v0",
    #             num_mission_keys=1,
    #             num_mission_doors=2,
    #             T=T
    #             )
    # print(out)
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_lockedroom_key_condition,
    #     near_object_func=near_lockedroom_key_condition,
    #     env_name="MiniGrid-LockedRoom-v0",
    #     num_mission_keys=1,
    #     num_mission_doors=2,
    #     T=T
    # )
    # print(out)
    
    # out, res, _ = verify_spec("subtask_completion", 
    #             subtask_complete_func=subtask_lockedroom_first_door_condition,
    #             env_name="MiniGrid-LockedRoom-v0",
    #             num_mission_keys=1,
    #             num_mission_doors=2,
    #             T=T
    #             )
    # print(out)
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_lockedroom_first_door_condition,
    #     near_object_func=near_lockedroom_first_door_condition,
    #     env_name="MiniGrid-LockedRoom-v0",
    #     num_mission_keys=1,
    #     num_mission_doors=2,
    #     T=T
    # )
    # print(out)
    
    # out, res, _ = verify_spec("subtask_completion", 
    #             subtask_complete_func=subtask_lockedroom_second_door_condition,
    #             env_name="MiniGrid-LockedRoom-v0",
    #             num_mission_keys=1,
    #             num_mission_doors=2,
    #             T=T
    #             )
    # print(out)
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_lockedroom_second_door_condition,
    #     near_object_func=near_lockedroom_second_door_condition,
    #     env_name="MiniGrid-LockedRoom-v0",
    #     num_mission_keys=1,
    #     num_mission_doors=2,
    #     T=T
    # )
    # print(out)
    
    # out, res, _ = verify_spec("subtask_completion", 
    #             subtask_complete_func=subtask_lockedroom_goal_condition,
    #             env_name="MiniGrid-LockedRoom-v0",
    #             num_mission_keys=1,
    #             num_mission_doors=2,
    #             T=T
    #             )
    # print(out)
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_lockedroom_goal_condition,
    #     near_object_func=near_lockedroom_goal_condition,
    #     env_name="MiniGrid-LockedRoom-v0",
    #     num_mission_keys=1,
    #     num_mission_doors=2,
    #     T=T
    # )
    # print(out)
    
    # restricted_init_states = []
    # for i in range(N):
    #     print(f"Test {i}")
    #     _, res, init_state = verify_spec(
    #         "persistence", 
    #         subtask_complete_func=[subtask_lockedroom_key_condition, subtask_lockedroom_first_door_condition, subtask_lockedroom_second_door_condition, subtask_lockedroom_goal_condition],
    #         env_name="MiniGrid-LockedRoom-v0",
    #         num_mission_doors=2,
    #         num_mission_keys=1,
    #         restricted_initial_states=restricted_init_states,
    #     )
    #     if not res:
    #         print("  Persistence failed.")
    #         # Check which function failed
    #         for func in [subtask_lockedroom_key_condition, subtask_lockedroom_first_door_condition, subtask_lockedroom_second_door_condition, subtask_lockedroom_goal_condition]:
    #             _, res_func, _ = verify_spec(
    #                 "persistence", 
    #                 subtask_complete_func=[func],
    #                 env_name="MiniGrid-LockedRoom-v0",
    #                 num_mission_doors=2,
    #                 num_mission_keys=1,
    #                 restricted_initial_states=restricted_init_states,
    #             )
    #             if not res_func:
    #                 print(f"  Persistence failed for function: {func.__name__}")
    #                 break
    #         break
    #     restricted_init_states.append(init_state)
    #     print("  Persistence holds.")
    
    # ---- 
    # Test DroneSupplier Environment specifications
    # ----
    
    # out, res, _ = verify_spec("subtask_completion", 
    #             subtask_complete_func=subtask_dronesupplier_box_condition,
    #             env_name="DroneSupplier-v0",
    #             num_mission_keys=1,
    #             num_mission_doors=1,
    #             num_mission_boxes=1,
    #             T=T
    #             )
    # print(out)
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_dronesupplier_box_condition,
    #     near_object_func=near_dronesupplier_box_condition,
    #     env_name="DroneSupplier-v0",
    #     num_mission_keys=1,
    #     num_mission_doors=1,
    #     num_mission_boxes=1,
    #     T=T
    # )
    # print(out)
    
    # out, res, _ = verify_spec("subtask_completion",
    #             subtask_complete_func=subtask_dronesupplier_key_condition,
    #             env_name="DroneSupplier-v0",
    #             num_mission_keys=1,
    #             num_mission_doors=1,
    #             num_mission_boxes=1,
    #             T=T
    #             )
    # print(out)
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_dronesupplier_key_condition,
    #     near_object_func=near_dronesupplier_key_condition,
    #     env_name="DroneSupplier-v0",
    #     num_mission_keys=1,
    #     num_mission_doors=1,
    #     num_mission_boxes=1,
    #     T=T
    # )
    # print(out)
    
    # out, res, _ = verify_spec("subtask_completion",
    #             subtask_complete_func=subtask_dronesupplier_door_condition,
    #             env_name="DroneSupplier-v0",
    #             num_mission_keys=1,
    #             num_mission_doors=1,
    #             num_mission_boxes=1,
    #             T=T
    #             )
    # print(out)
    # out, res, _ = verify_spec(
    #     "near_object",
    #     subtask_complete_func=subtask_dronesupplier_door_condition,
    #     near_object_func=near_dronesupplier_door_condition,
    #     env_name="DroneSupplier-v0",
    #     num_mission_keys=1,
    #     num_mission_doors=1,
    #     num_mission_boxes=1,
    #     T=T
    # )
    # print(out)
    
    restricted_init_states = []
    for i in range(N):
        print(f"Test {i}")
        _, res, init_state = verify_spec(
            "persistence", 
            subtask_complete_func=[subtask_dronesupplier_box_condition, subtask_dronesupplier_key_condition, subtask_dronesupplier_door_condition],               
            env_name="DroneSupplier-v0",
            num_mission_doors=1,
            num_mission_keys=1,
            num_mission_boxes=1,
        )
        if not res:
            print("  Persistence failed.")
            # Check which function failed
            for func in [subtask_dronesupplier_box_condition, subtask_dronesupplier_key_condition, subtask_dronesupplier_door_condition]:
                _, res_func, _ = verify_spec(
                    "persistence", 
                    subtask_complete_func=[func],
                    env_name="DroneSupplier-v0",
                    num_mission_doors=1,
                    num_mission_keys=1,
                    num_mission_boxes=1,
                )
                if not res_func:
                    print(f"  Persistence failed for function: {func.__name__}")
                    break
            break
        restricted_init_states.append(init_state)
        print("  Persistence holds.")
    
    print(f"Total time: {time.time() - start_time} seconds")
        
