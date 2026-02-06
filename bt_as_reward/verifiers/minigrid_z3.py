import z3
import random
from minigrid.wrappers import ImgObsWrapper, FullyObsWrapper
import gymnasium as gym

from bt_as_reward.constants import MINIGRID_COLOR_TO_IDX, MINIGRID_IDX_TO_COLOR, MINIGRID_OBJECT_TO_IDX
from bt_as_reward.envs.lockedroom_small import LockedRoomSmallEnv
from bt_as_reward.envs.drone_supplier import DroneSupplierSmallEnv

def _env_constraints(env_name, 
                     state_vars, 
                     s: z3.Solver, 
                     env, 
                     initial_door_pos=None, 
                     initial_key_pos=None,
                     initial_goal_pos=None,
                     initial_mission_keys=None,
                     initial_mission_doors=None,
                     initial_mission_boxes=None,
                     T=25,
                     negative_check=False):
    x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y, mission_keys, mission_doors, mission_boxes = state_vars
    
    match env_name:
        case "MiniGrid-DoorKey-6x6-v0":
            # Initial state
            s.add(x[0] < door_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]],
                key_x_colors[0][MINIGRID_COLOR_TO_IDX["yellow"]] < door_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]], 
                key_x_colors[0][MINIGRID_COLOR_TO_IDX["yellow"]] > 0,
                key_y_colors[0][MINIGRID_COLOR_TO_IDX["yellow"]] > 0,
                key_y_colors[0][MINIGRID_COLOR_TO_IDX["yellow"]] < env.unwrapped.height-1,
                door_state_colors[0][MINIGRID_COLOR_TO_IDX["yellow"]] == 2)
            
            if initial_door_pos is not None:
                s.add(door_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]] == initial_door_pos[0],
                      door_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]] == initial_door_pos[1])
            if initial_key_pos is not None:
                s.add(key_x_colors[0][MINIGRID_COLOR_TO_IDX["yellow"]] == initial_key_pos[0],
                      key_y_colors[0][MINIGRID_COLOR_TO_IDX["yellow"]] == initial_key_pos[1])
            
            # Set goal position
            s.add(goal_x == env.unwrapped.width - 2, goal_y == env.unwrapped.height - 2)
            s.add(mission_keys[0] == MINIGRID_COLOR_TO_IDX["yellow"])
            s.add(mission_doors[0] == MINIGRID_COLOR_TO_IDX["yellow"])
            
            # Mission Condition: Reach Goal
            if not negative_check:
                s.add(z3.And(x[T-1] == env.unwrapped.width - 2, y[T-1] == env.unwrapped.height - 2))
            
            # Door position constraints - valid spawn range
            for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                s.add(box_contains[color_idx] == -1)  # no boxes in this env
                
                if color_idx == MINIGRID_COLOR_TO_IDX["yellow"]:
                    s.add(door_x_colors[color_idx] > 1, door_x_colors[color_idx] < 4)
                    s.add(door_y_colors[color_idx] > 0, door_y_colors[color_idx] < 5)
                else:
                    s.add(door_x_colors[color_idx] == -1, door_y_colors[color_idx] == -1)
            
            for t in range(T):
                for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                    # Boxes are not present in this env
                    s.add(box_x_colors[t][color_idx] == -1, box_y_colors[t][color_idx] == -1)
                    
                    if color_idx == MINIGRID_COLOR_TO_IDX["yellow"]:
                        # door state constraints
                        s.add(
                            z3.Or(door_state_colors[t][color_idx] == 0, 
                                door_state_colors[t][color_idx] == 1, 
                                door_state_colors[t][color_idx] == 2)
                        )  # 0=open, 1=closed, 2=locked
                        s.add(
                            key_x_colors[t][color_idx] != door_x_colors[color_idx]
                        )
                        
                        s.add(
                            z3.Or(
                                key_x_colors[t][color_idx] != goal_x,
                                key_y_colors[t][color_idx] != goal_y
                            )
                        )
                    else:
                        s.add(door_state_colors[t][color_idx] == -1)  # not present
                        s.add(key_x_colors[t][color_idx] == -1, key_y_colors[t][color_idx] == -1) # not present
                
                # Cannot be at yellow door position when door is closed or locked and cannot pass through wall
                s.add(
                    z3.Implies(
                        x[t] == door_x_colors[MINIGRID_COLOR_TO_IDX["yellow"]],
                        z3.And(
                            y[t] == door_y_colors[MINIGRID_COLOR_TO_IDX["yellow"]],
                            door_state_colors[t][MINIGRID_COLOR_TO_IDX["yellow"]] == 0
                        )
                    )
                )
        case "MiniGrid-LockedRoom-v0":
            obs, _ = env.reset()
            
            walls = []
            for x_pos in range(obs.shape[0]):
                for y_pos in range(obs.shape[1]):
                    if obs[x_pos, y_pos, 0] == MINIGRID_OBJECT_TO_IDX["wall"]:
                        walls.append((x_pos, y_pos))
            
            # Agent starts in the middle region
            s.add(
                x[0] == 3
            )
            
            if initial_door_pos is not None:
                for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                    s.add(
                        door_x_colors[color_idx] == initial_door_pos[color_idx][0],
                        door_y_colors[color_idx] == initial_door_pos[color_idx][1]
                    )
            else:
                # Shuffle list of door locations and assign doors (2, 1), (2, 3), (2, 5), (4, 1), (4, 3), (4, 5)
                door_locations = [(2, 1), (2, 3), (2, 5), (4, 1), (4, 3), (4, 5)]
                shuffled_door_locs = door_locations.copy()
                random.shuffle(shuffled_door_locs)
                for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                    s.add(
                        door_x_colors[color_idx] == shuffled_door_locs[color_idx][0],
                        door_y_colors[color_idx] == shuffled_door_locs[color_idx][1]
                    )
            
            if initial_mission_keys is not None:
                s.add(mission_keys[0] == initial_mission_keys[0])
            
            if initial_mission_doors is not None:
                s.add(mission_doors[0] == initial_mission_doors[0])
                s.add(mission_doors[1] == initial_mission_doors[1])
            
            
            # Key to the locked room, door to key and door to locked room are different colors
            s.add(
                mission_keys[0] == mission_doors[1],
                mission_doors[0] != mission_doors[1],
            )
            
            # Mission Condition: Reach Goal
            if not negative_check:
                s.add(z3.And(x[T-1] == goal_x, y[T-1] == goal_y))
            
            for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                s.add(
                    box_contains[color_idx] == -1
                )
                
                # Door at mission_doors[1] is initially locked, all other are closed
                s.add(
                    z3.If(
                        color_idx == mission_doors[1],
                        door_state_colors[0][color_idx] == 2,  # locked
                        door_state_colors[0][color_idx] == 1   # closed
                    )
                )
                
                # Set goal location based on the mission_doors[1] color door
                s.add(
                    z3.Implies(
                        color_idx == mission_doors[1],
                        z3.If(
                            door_x_colors[color_idx] == 2,
                            z3.And(goal_x == 1, goal_y == door_y_colors[color_idx]),
                            z3.And(goal_x == 5, goal_y == door_y_colors[color_idx])
                        )
                    )
                )
                
                for color_idx2 in MINIGRID_IDX_TO_COLOR.keys():
                    # Set mission_keys[0] color key location based on mission_doors[0] color door
                    s.add(
                        z3.Implies(
                            z3.And(color_idx == mission_doors[0], color_idx2 == mission_keys[0]),
                            z3.If(
                                door_x_colors[color_idx] == 2,
                                z3.And(key_x_colors[0][color_idx2] == 1, key_y_colors[0][color_idx2] == door_y_colors[color_idx]),
                                z3.And(key_x_colors[0][color_idx2] == 5, key_y_colors[0][color_idx2] == door_y_colors[color_idx])
                            )
                        )
                    )
            
            for t in range(T):
                for wx, wy in walls:
                    # Agent cannot be at wall positions
                    if wx > 0 and wx < env.unwrapped.width-1 and wy > 0 and wy < env.unwrapped.height-1:
                        s.add(
                            z3.Or(
                                x[t] != wx,
                                y[t] != wy
                            )
                        )
                
                for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                    # Boxes are not present in this env
                    s.add(box_x_colors[t][color_idx] == -1, box_y_colors[t][color_idx] == -1)
                    
                    # door state constraints
                    s.add(
                        z3.Or(door_state_colors[t][color_idx] == 0, 
                            door_state_colors[t][color_idx] == 1, 
                            door_state_colors[t][color_idx] == 2)
                    )  # 0=open, 1=closed, 2=locked
                    
                    # Cannot be at door position when door is closed or locked
                    s.add(
                        z3.Implies(
                            z3.And(x[t] == door_x_colors[color_idx], y[t] == door_y_colors[color_idx]),
                            door_state_colors[t][color_idx] == 0
                        )
                    )
                    
                    s.add(
                        key_x_colors[t][color_idx] != 2,
                        key_x_colors[t][color_idx] != 4
                    )
                    
                    for wx, wy in walls:
                        # Key cannot be at wall positions
                        if wx > 0 and wx < env.unwrapped.width-1 and wy > 0 and wy < env.unwrapped.height-1:
                            s.add(
                                z3.Or(
                                    key_x_colors[t][color_idx] != wx,
                                    key_y_colors[t][color_idx] != wy
                                )
                            )
                    
                    # Key cannot be at goal position
                    s.add(
                        z3.Or(
                            key_x_colors[t][color_idx] != goal_x,
                            key_y_colors[t][color_idx] != goal_y
                        )
                    )
                    
                    s.add(
                        z3.Implies(
                            color_idx != mission_keys[0],
                            z3.And(
                                key_x_colors[t][color_idx] == -1,
                                key_y_colors[t][color_idx] == -1
                            )
                        )
                    )
        case "DroneSupplier-v0":
            # Agent starts at (4, 4) direction 0
            s.add(
                x[0] == 4, y[0] == 4, dir[0] == 0, goal_x == -1, goal_y == -1
            )
            
            # Mission key and door must be the same color
            s.add(
                mission_keys[0] == mission_doors[0]
            )
            
            if initial_door_pos is not None:
                for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                    s.add(
                        door_x_colors[color_idx] == initial_door_pos[color_idx][0],
                        door_y_colors[color_idx] == initial_door_pos[color_idx][1]
                    )
            if initial_mission_keys is not None:
                s.add(mission_keys[0] == initial_mission_keys[0])
            if initial_mission_doors is not None:
                s.add(mission_doors[0] == initial_mission_doors[0])
            if initial_mission_boxes is not None:
                s.add(mission_boxes[0] == initial_mission_boxes[0])
            
            # Box initial positions (3, 1), (4, 1), (3, 3), (4, 3), (3, 6), (4, 6)
            box_locations = [(3, 1), (4, 1), (3, 3), (4, 3), (3, 6), (4, 6)]
            # Door initial positions (1, 1), (6, 1), (1, 3), (6, 3), (1, 6), (6, 6)
            door_locations = [(1, 1), (6, 1), (1, 3), (6, 3), (1, 6), (6, 6)]
            shuffled_door_locs = door_locations.copy()
            random.shuffle(shuffled_door_locs)
            for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                # Mission Condition: Mission door is open
                if not negative_check:
                    s.add(
                        z3.Implies(
                            color_idx == mission_doors[0],
                            door_state_colors[T-1][color_idx] == 0
                        )
                    )
                s.add(
                    box_x_colors[0][color_idx] == box_locations[color_idx][0],
                    box_y_colors[0][color_idx] == box_locations[color_idx][1]
                )
                if initial_door_pos is None:
                    s.add(
                        door_x_colors[color_idx] == shuffled_door_locs[color_idx][0],
                        door_y_colors[color_idx] == shuffled_door_locs[color_idx][1]
                    )
                
                # Door at mission_doors[1] is initially locked, all other are closed
                s.add(
                    z3.If(
                        color_idx == mission_doors[0],
                        door_state_colors[0][color_idx] == 2,  # locked
                        door_state_colors[0][color_idx] == 1   # closed
                    )
                )
                
                # keys are initially -1
                s.add(
                    key_x_colors[0][color_idx] == -1,
                    key_y_colors[0][color_idx] == -1
                )
                
                # Set box_contains based on mission_boxes and mission_keys
                s.add(
                    z3.If(
                        color_idx == mission_boxes[0],
                        box_contains[color_idx] == mission_keys[0],
                        box_contains[color_idx] == -1
                    )
                )
            
            for t in range(T):
                for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                    # door state constraints
                    s.add(
                        z3.Or(door_state_colors[t][color_idx] == 0, 
                            door_state_colors[t][color_idx] == 1, 
                            door_state_colors[t][color_idx] == 2)
                    )  # 0=open, 1=closed, 2=locked
                    # Cannot be at door position when door is closed or locked
                    s.add(
                        z3.Implies(
                            z3.And(x[t] == door_x_colors[color_idx], y[t] == door_y_colors[color_idx]),
                            door_state_colors[t][color_idx] == 0
                        )
                    )
                    
                    for dx, dy in door_locations:
                        # Box cannot be at door positions
                        s.add(
                            z3.Or(
                                key_x_colors[t][color_idx] != dx,
                                key_y_colors[t][color_idx] != dy
                            )
                        )
                    
                    for i, (bx, by) in enumerate(box_locations):
                        s.add(
                            z3.Implies(
                                z3.And(key_x_colors[t][color_idx] == bx, key_y_colors[t][color_idx] == by),
                                box_x_colors[t][i] == -1,
                            )
                        )
                
            
                
                
    

def verify_spec(spec_type: str, 
                subtask_complete_func=None, 
                near_object_func=None, 
                initial_door_state=None, 
                initial_key_state=None,
                initial_goal_state=None,
                initial_mission_keys=None,
                initial_mission_doors=None,
                initial_mission_boxes=None,
                env_name="MiniGrid-DoorKey-6x6-v0",
                num_mission_keys=1,
                num_mission_doors=1,
                num_mission_boxes=0,
                T=25,
                restricted_initial_states=None,
                negative_check=False,
                timeout=900000):
    match env_name:
        case "MiniGrid-DoorKey-6x6-v0":
            env = gym.make(env_name)
        case "MiniGrid-LockedRoom-v0":
            env = ImgObsWrapper(FullyObsWrapper(LockedRoomSmallEnv(size=7)))
        case "DroneSupplier-v0":
            env = DroneSupplierSmallEnv(size=8)

    # State variables
    x = [z3.Int(f"x_{t}") for t in range(T)]
    y = [z3.Int(f"y_{t}") for t in range(T)]
    dir = [z3.Int(f"dir_{t}") for t in range(T)]
    
    # For every color of door, we can have a separate door variable
    door_x_colors = [z3.Int(f"door_x_{color}") for color in MINIGRID_COLOR_TO_IDX.keys()]
    door_y_colors = [z3.Int(f"door_y_{color}") for color in MINIGRID_COLOR_TO_IDX.keys()]
    door_state_colors = [[z3.Int(f"door_state_{color}_{t}") for color in MINIGRID_COLOR_TO_IDX.keys()] for t in range(T)]
    door_occ_x_colors = [[z3.Int(f"door_occ_x_{color}_{t}") for color in MINIGRID_COLOR_TO_IDX.keys()] for t in range(T)]
    door_occ_y_colors = [[z3.Int(f"door_occ_y_{color}_{t}") for color in MINIGRID_COLOR_TO_IDX.keys()] for t in range(T)]
    door_occ_state_colors = [[z3.Int(f"door_occ_state_{color}_{t}") for color in MINIGRID_COLOR_TO_IDX.keys()] for t in range(T)]
    
    key_x_colors = [[z3.Int(f"key_x_{color}_{t}") for color in MINIGRID_COLOR_TO_IDX.keys()] for t in range(T)]
    key_y_colors = [[z3.Int(f"key_y_{color}_{t}") for color in MINIGRID_COLOR_TO_IDX.keys()] for t in range(T)]
    
    # Boxes can contain the color of the key they hold else -1 for empty
    box_x_colors = [[z3.Int(f"box_x_{color}_{t}") for color in MINIGRID_COLOR_TO_IDX.keys()] for t in range(T)]
    box_y_colors = [[z3.Int(f"box_y_{color}_{t}") for color in MINIGRID_COLOR_TO_IDX.keys()] for t in range(T)]
    box_contains = [z3.Int(f"box_contains_{color}") for color in MINIGRID_COLOR_TO_IDX.keys()]
    
    goal_x = z3.Int(f"goal_x")
    goal_y = z3.Int(f"goal_y")
    goal_occ_x = [z3.Int(f"goal_occ_x_{t}") for t in range(T)]
    goal_occ_y = [z3.Int(f"goal_occ_y_{t}") for t in range(T)]
    
    # Mission variables
    mission_keys = [z3.Int(f"mission_key_{i}") for i in range(num_mission_keys)]
    mission_doors = [z3.Int(f"mission_door_{i}") for i in range(num_mission_doors)]
    mission_boxes = [z3.Int(f"mission_box_{i}") for i in range(num_mission_boxes)]

    s = z3.Solver()
    s.set(timeout=timeout)
    
    if restricted_initial_states is not None:
        for state in restricted_initial_states:
            init_x, init_y, init_dir, init_door_x_colors, init_door_y_colors, init_door_state_colors, init_key_x_colors, init_key_y_colors, init_box_x_colors, init_box_y_colors, init_box_contains, init_goal_x, init_goal_y, init_mission_keys, init_mission_doors, init_mission_boxes = state
            s.add(z3.Or(
                x[0] != init_x,
                y[0] != init_y,
                dir[0] != init_dir,
                *[door_x_colors[i] != init_door_x_colors[i] for i in range(len(MINIGRID_COLOR_TO_IDX))],
                *[door_y_colors[i] != init_door_y_colors[i] for i in range(len(MINIGRID_COLOR_TO_IDX))],
                *[door_state_colors[0][i] != init_door_state_colors[i] for i in range(len(MINIGRID_COLOR_TO_IDX))],
                *[key_x_colors[0][i] != init_key_x_colors[i] for i in range(len(MINIGRID_COLOR_TO_IDX))],
                *[key_y_colors[0][i] != init_key_y_colors[i] for i in range(len(MINIGRID_COLOR_TO_IDX))],
                *[box_x_colors[0][i] != init_box_x_colors[i] for i in range(len(MINIGRID_COLOR_TO_IDX))],
                *[box_y_colors[0][i] != init_box_y_colors[i] for i in range(len(MINIGRID_COLOR_TO_IDX))],
                *[box_contains[i] != init_box_contains[i] for i in range(len(MINIGRID_COLOR_TO_IDX))],
                goal_x != init_goal_x,
                goal_y != init_goal_y,
                *[mission_keys[i] != init_mission_keys[i] for i in range(num_mission_keys)],
                *[mission_doors[i] != init_mission_doors[i] for i in range(num_mission_doors)],
                *[mission_boxes[i] != init_mission_boxes[i] for i in range(num_mission_boxes)],
            ))
    
    # Mission variables represent color indices for objects in the mission [0, 5]
    s.add([z3.And(mission_keys[i] >= 0, mission_keys[i] < len(MINIGRID_COLOR_TO_IDX)) for i in range(num_mission_keys)])
    s.add([z3.And(mission_doors[i] >= 0, mission_doors[i] < len(MINIGRID_COLOR_TO_IDX)) for i in range(num_mission_doors)])
    s.add([z3.And(mission_boxes[i] >= 0, mission_boxes[i] < len(MINIGRID_COLOR_TO_IDX)) for i in range(num_mission_boxes)])
    
    s.add([z3.And(box_contains[i] >= -1, box_contains[i] < len(MINIGRID_COLOR_TO_IDX)) for i in range(len(MINIGRID_COLOR_TO_IDX))])
    
    _env_constraints(env_name, 
                        (x, y, dir, door_x_colors, door_y_colors, door_state_colors, key_x_colors, key_y_colors, box_x_colors, box_y_colors, box_contains, goal_x, goal_y, mission_keys, mission_doors, mission_boxes), 
                        s,
                        env,
                        initial_door_pos=initial_door_state,
                        initial_key_pos=initial_key_state,
                        initial_goal_pos=initial_goal_state,
                        initial_mission_keys=initial_mission_keys,
                        initial_mission_doors=initial_mission_doors,
                        initial_mission_boxes=initial_mission_boxes,
                        T=T,
                        negative_check=negative_check or (spec_type=="near_object")
                    )
    
    for t in range(T):
        s.add(x[t] > 0, 
            y[t] > 0, 
            x[t] < env.unwrapped.width-1, 
            y[t] < env.unwrapped.height-1, 
            dir[t] >= 0, 
            dir[t] < 4)
        
        # Occluded if agent is standing on the goal
        s.add(
            z3.Implies(
                z3.And(x[t] == goal_x, y[t] == goal_y),
                z3.And(goal_occ_x[t] == -1, goal_occ_y[t] == -1)
            )
        )
        # Visible otherwise
        s.add(
            z3.Implies(
                z3.Or(x[t] != goal_x, y[t] != goal_y),
                z3.And(goal_occ_x[t] == goal_x, goal_occ_y[t] == goal_y)
            )
        )
        
        for color_idx in MINIGRID_IDX_TO_COLOR.keys():
            s.add(z3.Or(
                z3.And(
                    key_x_colors[t][color_idx] == -1,  # key has been picked up
                    key_y_colors[t][color_idx] == -1
                ),
                z3.And(
                    key_x_colors[t][color_idx] > 0, 
                    key_y_colors[t][color_idx] > 0, 
                    key_x_colors[t][color_idx] < env.unwrapped.width-1, 
                    key_y_colors[t][color_idx] < env.unwrapped.height-1
                )
            ))
            
            
            # key cannot be at agent position
            s.add(
                z3.Or(
                    key_x_colors[t][color_idx] != x[t],
                    key_y_colors[t][color_idx] != y[t]
                )
            )
            
            # box cannot be at agent position
            s.add(
                z3.Or(
                    box_x_colors[t][color_idx] != x[t],
                    box_y_colors[t][color_idx] != y[t]
                )
            )
            
            # Occluded if the agent is standing on the door
            s.add(
                z3.Implies(
                    z3.And(x[t] == door_x_colors[color_idx], y[t] == door_y_colors[color_idx]),
                    z3.And(door_occ_x_colors[t][color_idx] == -1, door_occ_y_colors[t][color_idx] == -1, door_occ_state_colors[t][color_idx] == -1)
                )
            )
            # Visible otherwise
            s.add(
                z3.Implies(
                    z3.Or(x[t] != door_x_colors[color_idx], y[t] != door_y_colors[color_idx]),
                    z3.And(door_occ_x_colors[t][color_idx] == door_x_colors[color_idx], 
                           door_occ_y_colors[t][color_idx] == door_y_colors[color_idx], 
                           door_occ_state_colors[t][color_idx] == door_state_colors[t][color_idx])
                )
            )
        

    # Dynamics : move forward, turn left, turn right, toggle door, unlock door, pickup key, drop key
    for t in range(T - 1):
        s.add(
            z3.Or(

                # -------- forward --------
                z3.And(
                    # position update depends on direction
                    z3.Or(
                        z3.And(dir[t] == 0, x[t+1] == x[t] + 1,     y[t+1] == y[t]),
                        z3.And(dir[t] == 1, x[t+1] == x[t], y[t+1] == y[t] + 1),
                        z3.And(dir[t] == 2, x[t+1] == x[t] - 1,     y[t+1] == y[t]),
                        z3.And(dir[t] == 3, x[t+1] == x[t], y[t+1] == y[t] - 1)
                    ),
                    dir[t+1] == dir[t],
                    z3.And(
                        *[key_x_colors[t+1][color_idx] == key_x_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[key_y_colors[t+1][color_idx] == key_y_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[door_state_colors[t+1][color_idx] == door_state_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_x_colors[t+1][color_idx] == box_x_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_y_colors[t+1][color_idx] == box_y_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    )
                ),

                # -------- turn left --------
                z3.And(
                    x[t+1] == x[t],
                    y[t+1] == y[t],
                    dir[t+1] == (dir[t] + 3) % 4,
                    z3.And(
                        *[key_x_colors[t+1][color_idx] == key_x_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[key_y_colors[t+1][color_idx] == key_y_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[door_state_colors[t+1][color_idx] == door_state_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_x_colors[t+1][color_idx] == box_x_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_y_colors[t+1][color_idx] == box_y_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    )
                ),

                # -------- turn right --------
                z3.And(
                    x[t+1] == x[t],
                    y[t+1] == y[t],
                    dir[t+1] == (dir[t] + 1) % 4,
                    z3.And(
                        *[key_x_colors[t+1][color_idx] == key_x_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[key_y_colors[t+1][color_idx] == key_y_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[door_state_colors[t+1][color_idx] == door_state_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_x_colors[t+1][color_idx] == box_x_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_y_colors[t+1][color_idx] == box_y_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    )
                ),
                
                # -------- toggle door --------
                *[z3.And(
                    # agent must be in the correct adjacent square
                    # manhattan distance of 1 to door
                    z3.Or(
                        z3.And(x[t] == door_x_colors[color_idx] - 1, y[t] == door_y_colors[color_idx], dir[t] == 0),  # left
                        z3.And(x[t] == door_x_colors[color_idx] + 1, y[t] == door_y_colors[color_idx], dir[t] == 2),  # right
                        z3.And(x[t] == door_x_colors[color_idx],     y[t] == door_y_colors[color_idx] - 1, dir[t] == 3),  # down
                        z3.And(x[t] == door_x_colors[color_idx],     y[t] == door_y_colors[color_idx] + 1, dir[t] == 1)   # up
                    ),
                    
                    # toggle door
                    z3.Or(
                        # door was open, now closed
                        z3.And(
                            door_state_colors[t][color_idx] == 0,
                            door_state_colors[t+1][color_idx] == 1
                        ),
                        # door was closed, now open
                        z3.And(
                            door_state_colors[t][color_idx] == 1,
                            door_state_colors[t+1][color_idx] == 0
                        ),
                        # door was locked, now open
                        z3.And(
                            # key must have been present in a previous timestep
                            z3.Or([key_x_colors[tt][color_idx] != -1 for tt in range(t)]),
                            key_x_colors[t][color_idx] == -1,  # have the key
                            door_state_colors[t][color_idx] == 2,
                            door_state_colors[t+1][color_idx] == 0
                        )
                    ),
                    
                    # position and direction unchanged
                    x[t+1] == x[t],
                    y[t+1] == y[t],
                    dir[t+1] == dir[t],
                    z3.And(
                        *[key_x_colors[t+1][c_idx] == key_x_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[key_y_colors[t+1][c_idx] == key_y_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_x_colors[t+1][c_idx] == box_x_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_y_colors[t+1][c_idx] == box_y_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    # door state of other colors unchanged
                    *[door_state_colors[t+1][other_idx] == door_state_colors[t][other_idx] 
                        for other_idx in MINIGRID_IDX_TO_COLOR.keys() if other_idx != color_idx]
                ) for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
                
                # ----- Pickup Key -----
                *[z3.And(
                    # agent must be in the correct adjacent square
                    # manhattan distance of 1 to key
                    z3.Or(
                        z3.And(x[t] == key_x_colors[t][color_idx] - 1, y[t] == key_y_colors[t][color_idx], dir[t] == 0),  # left
                        z3.And(x[t] == key_x_colors[t][color_idx] + 1, y[t] == key_y_colors[t][color_idx], dir[t] == 2),  # right
                        z3.And(x[t] == key_x_colors[t][color_idx],     y[t] == key_y_colors[t][color_idx] - 1, dir[t] == 3),  # down
                        z3.And(x[t] == key_x_colors[t][color_idx],     y[t] == key_y_colors[t][color_idx] + 1, dir[t] == 1)   # up
                    ),
                    # pick up key
                    key_x_colors[t+1][color_idx] == -1,
                    key_y_colors[t+1][color_idx] == -1,
                    # key position for other colors unchanged
                    *[key_x_colors[t+1][other_idx] == key_x_colors[t][other_idx] 
                        for other_idx in MINIGRID_IDX_TO_COLOR.keys() if other_idx != color_idx],
                    *[key_y_colors[t+1][other_idx] == key_y_colors[t][other_idx] 
                        for other_idx in MINIGRID_IDX_TO_COLOR.keys() if other_idx != color_idx],
                    # position, direction, door state unchanged
                    x[t+1] == x[t],
                    y[t+1] == y[t],
                    dir[t+1] == dir[t],
                    z3.And(
                        *[door_state_colors[t+1][c_idx] == door_state_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_x_colors[t+1][c_idx] == box_x_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_y_colors[t+1][c_idx] == box_y_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    )
                ) for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
                
                # ----- Drop Key -----
                *[z3.And(
                    # key must have been present in a previous timestep
                    z3.Or([key_x_colors[tt][color_idx] != -1 for tt in range(t)]),
                    # agent must not already be carrying the key
                    key_x_colors[t][color_idx] == -1,
                    key_y_colors[t][color_idx] == -1,
                    # drop key in front of agent
                    z3.Or(
                        z3.And(dir[t] == 0, key_x_colors[t+1][color_idx] == x[t] + 1,     key_y_colors[t+1][color_idx] == y[t]),
                        z3.And(dir[t] == 1, key_x_colors[t+1][color_idx] == x[t], key_y_colors[t+1][color_idx] == y[t] + 1),
                        z3.And(dir[t] == 2, key_x_colors[t+1][color_idx] == x[t] - 1,     key_y_colors[t+1][color_idx] == y[t]),
                        z3.And(dir[t] == 3, key_x_colors[t+1][color_idx] == x[t], key_y_colors[t+1][color_idx] == y[t] - 1)
                    ),
                    # key position for other colors unchanged
                    *[key_x_colors[t+1][other_idx] == key_x_colors[t][other_idx] 
                        for other_idx in MINIGRID_IDX_TO_COLOR.keys() if other_idx != color_idx],
                    *[key_y_colors[t+1][other_idx] == key_y_colors[t][other_idx] 
                        for other_idx in MINIGRID_IDX_TO_COLOR.keys() if other_idx != color_idx],
                    # position, direction, door state unchanged
                    x[t+1] == x[t],
                    y[t+1] == y[t],
                    dir[t+1] == dir[t],
                    z3.And(
                        *[door_state_colors[t+1][color_idx] == door_state_colors[t][color_idx] for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_x_colors[t+1][c_idx] == box_x_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    ),
                    z3.And(
                        *[box_y_colors[t+1][c_idx] == box_y_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    )
                ) for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
                
                # ---- Open Box ----
                *[z3.And(
                    # agent must be in the correct adjacent square
                    # manhattan distance of 1 to box
                    z3.Or(
                        z3.And(x[t] == box_x_colors[t][color_idx] - 1, y[t] == box_y_colors[t][color_idx], dir[t] == 0),  # left
                        z3.And(x[t] == box_x_colors[t][color_idx] + 1, y[t] == box_y_colors[t][color_idx], dir[t] == 2),  # right
                        z3.And(x[t] == box_x_colors[t][color_idx],     y[t] == box_y_colors[t][color_idx] - 1, dir[t] == 3),  # down
                        z3.And(x[t] == box_x_colors[t][color_idx],     y[t] == box_y_colors[t][color_idx] + 1, dir[t] == 1)   # up
                    ),
                    # open box to reveal key inside if any
                    *[z3.If(
                        box_contains[color_idx] == c_idx,
                        z3.And(
                            key_x_colors[t+1][c_idx] == box_x_colors[t][color_idx],
                            key_y_colors[t+1][c_idx] == box_y_colors[t][color_idx],
                        ),
                        z3.And(
                            key_x_colors[t+1][c_idx] == key_x_colors[t][c_idx],
                            key_y_colors[t+1][c_idx] == key_y_colors[t][c_idx],
                        )
                    ) for c_idx in MINIGRID_IDX_TO_COLOR.keys()],
                    
                    # box position becomes -1 (removed)
                    box_x_colors[t+1][color_idx] == -1,
                    box_y_colors[t+1][color_idx] == -1,
                    # other box positions unchanged
                    *[box_x_colors[t+1][other_idx] == box_x_colors[t][other_idx] for other_idx in MINIGRID_IDX_TO_COLOR.keys() if other_idx != color_idx],
                    *[box_y_colors[t+1][other_idx] == box_y_colors[t][other_idx] for other_idx in MINIGRID_IDX_TO_COLOR.keys() if other_idx != color_idx],
                    # position, direction, door state unchanged
                    x[t+1] == x[t],
                    y[t+1] == y[t],
                    dir[t+1] == dir[t],
                    z3.And(
                        *[door_state_colors[t+1][c_idx] == door_state_colors[t][c_idx] for c_idx in MINIGRID_IDX_TO_COLOR.keys()]
                    )
                ) for color_idx in MINIGRID_IDX_TO_COLOR.keys()]
            )
        )
    
    if spec_type == "subtask_completion":
        if subtask_complete_func is None:
            raise ValueError("subtask_complete_func must be provided for subtask_completion spec type")
        # Check negation of spec: f is false at all timesteps before reaching goal
        # This can also be used as the negative check if the goal is not reached find a trajectory where f is never true
        s.add(z3.And([z3.Not(subtask_complete_func((x[t], y[t], dir[t], door_occ_x_colors[t], door_occ_y_colors[t], door_occ_state_colors[t], key_x_colors[t], key_y_colors[t], box_x_colors[t], box_y_colors[t], box_contains, goal_occ_x[t], goal_occ_y[t]), (mission_keys, mission_doors, mission_boxes))) for t in range(T)]))
    elif spec_type == "near_object":
        if near_object_func is None:
            raise ValueError("near_object_func must be provided for near_object spec type")
        if subtask_complete_func is None:
            raise ValueError("subtask_complete_func must be provided for near_object spec type")
        if not negative_check:
            s.add(
                z3.Or([
                    z3.And(
                        subtask_complete_func((x[t], y[t], dir[t], door_occ_x_colors[t], door_occ_y_colors[t], door_occ_state_colors[t], key_x_colors[t], key_y_colors[t], box_x_colors[t], box_y_colors[t], box_contains, goal_occ_x[t], goal_occ_y[t]), (mission_keys, mission_doors, mission_boxes)),
                        z3.Not(subtask_complete_func((x[t-1], y[t-1], dir[t-1], door_occ_x_colors[t-1], door_occ_y_colors[t-1], door_occ_state_colors[t-1], key_x_colors[t-1], key_y_colors[t-1], box_x_colors[t-1], box_y_colors[t-1], box_contains, goal_occ_x[t-1], goal_occ_y[t-1]), (mission_keys, mission_doors, mission_boxes))),
                        z3.Not(near_object_func((x[t-1], y[t-1], dir[t-1], door_occ_x_colors[t-1], door_occ_y_colors[t-1], door_occ_state_colors[t-1], key_x_colors[t-1], key_y_colors[t-1], box_x_colors[t-1], box_y_colors[t-1], box_contains, goal_occ_x[t-1], goal_occ_y[t-1]), (mission_keys, mission_doors, mission_boxes)))
                    )
                    for t in range(1, T)
                ])
            )
        else:
            # Negative check: Same as completion find a trajectory where near_object is never true
            s.add(z3.And([z3.Not(near_object_func((x[t], y[t], dir[t], door_occ_x_colors[t], door_occ_y_colors[t], door_occ_state_colors[t], key_x_colors[t], key_y_colors[t], box_x_colors[t], box_y_colors[t], box_contains, goal_occ_x[t], goal_occ_y[t]), (mission_keys, mission_doors, mission_boxes))) for t in range(T)]))
    elif spec_type == "persistence":
        if subtask_complete_func is None:
            raise ValueError("subtask_complete_func must be provided for persistence spec type")
        if not isinstance(subtask_complete_func, list):
            raise ValueError("subtask_complete_func must be a list of functions for persistence spec type")
        # Check for a trajectory when f becomes true it remains true until goal is reached
        for t in range(T - 1):
            for i in range(len(subtask_complete_func)):
                s.add(
                    z3.Implies(
                        subtask_complete_func[i]((x[t], y[t], dir[t], door_occ_x_colors[t], door_occ_y_colors[t], door_occ_state_colors[t],
                                            key_x_colors[t], key_y_colors[t], box_x_colors[t], box_y_colors[t], box_contains, goal_occ_x[t], goal_occ_y[t]), (mission_keys, mission_doors, mission_boxes)),
                        subtask_complete_func[i]((x[t+1], y[t+1], dir[t+1], door_occ_x_colors[t+1], door_occ_y_colors[t+1], door_occ_state_colors[t+1],
                                            key_x_colors[t+1], key_y_colors[t+1], box_x_colors[t+1], box_y_colors[t+1], box_contains, goal_occ_x[t+1], goal_occ_y[t+1]), (mission_keys, mission_doors, mission_boxes))
                    )
                )


    # Solve
    if s.check() == z3.sat:
        m = s.model()
        initial_state = (
            m[x[0]], m[y[0]], m[dir[0]],
            [m[door_x_colors[color_idx]] for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
            [m[door_y_colors[color_idx]] for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
            [m[door_state_colors[0][color_idx]] for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
            [m[key_x_colors[0][color_idx]] for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
            [m[key_y_colors[0][color_idx]] for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
            [m[box_x_colors[0][color_idx]] for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
            [m[box_y_colors[0][color_idx]] for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
            [m[box_contains[color_idx]] for color_idx in MINIGRID_IDX_TO_COLOR.keys()],
            m[goal_x], m[goal_y],
            [m[mission_keys[i]] for i in range(num_mission_keys)],
            [m[mission_doors[i]] for i in range(num_mission_doors)],
            [m[mission_boxes[i]] for i in range(num_mission_boxes)],
        )
        if spec_type == "persistence" or negative_check:
            return "", True, initial_state
        out = ""
        out += f"Counterexample trajectory found for {spec_type} with subtask_complete_func ({subtask_complete_func.__name__ if subtask_complete_func else 'N/A'}) and near_object_func ({near_object_func.__name__ if near_object_func else 'N/A'}):\n"
        out += f"Mission: {'mission_keys=' + ','.join([MINIGRID_IDX_TO_COLOR[m.evaluate(mission_keys[i]).as_long()] for i in range(num_mission_keys)])}, {'mission_doors=' + ','.join([MINIGRID_IDX_TO_COLOR[m.evaluate(mission_doors[i]).as_long()] for i in range(num_mission_doors)])}, {'mission_boxes=' + ','.join([MINIGRID_IDX_TO_COLOR[m.evaluate(mission_boxes[i]).as_long()] for i in range(num_mission_boxes)])}\n"
        # add box_contains
        out += "Box contains: " + ", ".join([f"{MINIGRID_IDX_TO_COLOR[color_idx]}: {MINIGRID_IDX_TO_COLOR[m.evaluate(box_contains[color_idx]).as_long()] if m.evaluate(box_contains[color_idx]).as_long() != -1 else '-1'}" for color_idx in MINIGRID_IDX_TO_COLOR.keys()]) + "\n"
        for t in range(T):
            out += ("-"*20) + "\n"
            out += f"t={t}: x={m[x[t]]}, y={m[y[t]]}, dir={m[dir[t]]}, goal_x={m[goal_occ_x[t]]}, goal_y={m[goal_occ_y[t]]}\n"
            for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                out += f"key_color={MINIGRID_IDX_TO_COLOR[color_idx]}: key_x={m[key_x_colors[t][color_idx]]}, key_y={m[key_y_colors[t][color_idx]]}\n"
            for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                out += f"door_color={MINIGRID_IDX_TO_COLOR[color_idx]}: door_x={m[door_occ_x_colors[t][color_idx]]}, door_y={m[door_occ_y_colors[t][color_idx]]}, door_state={m[door_occ_state_colors[t][color_idx]]}\n"
            for color_idx in MINIGRID_IDX_TO_COLOR.keys():
                out += f"box_color={MINIGRID_IDX_TO_COLOR[color_idx]}: box_x={m[box_x_colors[t][color_idx]]}, box_y={m[box_y_colors[t][color_idx]]}\n"
            if subtask_complete_func:
                out += f"subtask_complete_func={m.evaluate(subtask_complete_func((x[t], y[t], dir[t], door_occ_x_colors[t], door_occ_y_colors[t], door_occ_state_colors[t], key_x_colors[t], key_y_colors[t], box_x_colors[t], box_y_colors[t], box_contains, goal_occ_x[t], goal_occ_y[t]), (mission_keys, mission_doors, mission_boxes)))}\n"
            if near_object_func:
                out += f"near_object_func={m.evaluate(near_object_func((x[t], y[t], dir[t], door_occ_x_colors[t], door_occ_y_colors[t], door_occ_state_colors[t], key_x_colors[t], key_y_colors[t], box_x_colors[t], box_y_colors[t], box_contains, goal_occ_x[t], goal_occ_y[t]), (mission_keys, mission_doors, mission_boxes)))}\n"
            out += ("-"*20) + "\n"
        return out, False, initial_state
    else:
        if spec_type == "persistence" or negative_check:
            return "", False, ()
        out = ""
        print(f"No counterexample trajectory exists; {spec_type} is true for subtask_complete_func ({subtask_complete_func.__name__ if subtask_complete_func else 'N/A'}) and near_object_func ({near_object_func.__name__ if near_object_func else 'N/A'}).")
        return out, True, ()