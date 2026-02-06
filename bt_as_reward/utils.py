import os
import json
import re
from matplotlib import text
import requests
import ast
from pathlib import Path
from typing import Tuple, List, Callable, Optional
from types import ModuleType
import importlib.util
import numpy as np
import z3
from bt_as_reward.constants import (
    MINIGRID_IDX_TO_OBJECT,
    MINIGRID_IDX_TO_COLOR,
    MINIGRID_OBJECT_TO_IDX,
    MINIGRID_STATE_TO_IDX,
    MINIGRID_COLOR_TO_IDX,
    MUJOCO_IDX_TO_COLOR,
    MUJOCO_IDX_TO_OBJECT,
)

def load_function_from_str(source: str,function_name: str):
    module = ModuleType("isolated")
    exec(source, module.__dict__)
    return getattr(module, function_name, None)


def load_function_from_file(file_path: str, function_name: str) -> Optional[Callable]:
    """
    Dynamically load a function from a specified file path.

    :param file_path: Path to the Python file containing the function.
    :param function_name: Name of the function to load.
    :return: The loaded function or None if not found.
    """
    spec = importlib.util.spec_from_file_location("module.name", file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    return getattr(module, function_name, None)


def load_functions_from_file(
    file_path: str, function_names: List[str]
) -> List[Callable]:
    """
    Load multiple functions from a specified file path.

    :param file_path: Path to the Python file containing the functions.
    :param function_names: List of function names to load.
    :return: List of loaded functions.
    """
    functions = []
    for function_name in function_names:
        func = load_function_from_file(file_path, function_name)
        if func is None:
            raise ValueError(f"Function {function_name} not found in {file_path}.")
        functions.append(func)
    return functions

def minigrid_sym_traj_to_str(sym_traj: List, subtask_complete_traj: List, near_object_traj: Optional[List] = None) -> str:
    out = ""
    out += f"Mission: {'mission_keys=' + ','.join([MINIGRID_IDX_TO_COLOR[sym_traj[0][1][0][i].as_long()] for i in range(len(sym_traj[0][1][0]))])}, {'mission_doors=' + ','.join([MINIGRID_IDX_TO_COLOR[sym_traj[0][1][1][i].as_long()] for i in range(len(sym_traj[0][1][1]))])}, {'mission_boxes=' + ','.join([MINIGRID_IDX_TO_COLOR[sym_traj[0][1][2][i].as_long()] for i in range(len(sym_traj[0][1][2]))])}\n"
    out += "Box contains: " + ", ".join([f"{MINIGRID_IDX_TO_COLOR[color_idx]}: {MINIGRID_IDX_TO_COLOR[sym_traj[0][0][10][color_idx].as_long()] if sym_traj[0][0][10][color_idx].as_long() != -1 else '-1'}" for color_idx in MINIGRID_IDX_TO_COLOR.keys()]) + "\n"
    for t, (sym_state, _) in enumerate(sym_traj):
        agent_x = sym_state[0].as_long()
        agent_y = sym_state[1].as_long()
        agent_dir = sym_state[2].as_long()
        door_x_colors = [c.as_long() for c in sym_state[3]]
        door_y_colors = [c.as_long() for c in sym_state[4]]
        door_state_colors = [c.as_long() for c in sym_state[5]]
        key_x_colors = [c.as_long() for c in sym_state[6]]
        key_y_colors = [c.as_long() for c in sym_state[7]]
        box_x_colors = [c.as_long() for c in sym_state[8]]
        box_y_colors = [c.as_long() for c in sym_state[9]]
        goal_x = sym_state[11].as_long()
        goal_y = sym_state[12].as_long()

        out += ("-"*20) + "\n"
        out += f"t={t}: x={agent_x}, y={agent_y}, dir={agent_dir}, goal_x={goal_x}, goal_y={goal_y}\n"
        for color_idx in MINIGRID_IDX_TO_COLOR.keys():
            out += f"key_color={MINIGRID_IDX_TO_COLOR[color_idx]}: key_x={key_x_colors[color_idx]}, key_y={key_y_colors[color_idx]}\n"
        for color_idx in MINIGRID_IDX_TO_COLOR.keys():
            out += f"door_color={MINIGRID_IDX_TO_COLOR[color_idx]}: door_x={door_x_colors[color_idx]}, door_y={door_y_colors[color_idx]}, door_state={door_state_colors[color_idx]}\n"
        for color_idx in MINIGRID_IDX_TO_COLOR.keys():
            out += f"box_color={MINIGRID_IDX_TO_COLOR[color_idx]}: box_x={box_x_colors[color_idx]}, box_y={box_y_colors[color_idx]}\n"
        out += f"subtask_complete_func={bool(subtask_complete_traj[t])}\n"
        if near_object_traj is not None:
            out += f"near_object_func={bool(near_object_traj[t])}\n"
        out += ("-"*20) + "\n"
    return out

def mujoco_sym_traj_to_str(sym_traj: List, subtask_complete_traj: List, near_object_traj: Optional[List] = None) -> str:
    out = ""
    out += f"Mission Goal: {MUJOCO_IDX_TO_COLOR[sym_traj[0][1][0].as_long()]}\n"
    for t, (sym_state, _) in enumerate(sym_traj):
        gripper_x = sym_state[0].as_decimal(5)
        gripper_y = sym_state[1].as_decimal(5)
        gripper_z = sym_state[2].as_decimal(5)
        gripper_disp = sym_state[3].as_decimal(5)
        block_x = sym_state[4].as_decimal(5)
        block_y = sym_state[5].as_decimal(5)
        block_z = sym_state[6].as_decimal(5)
        gripper_vel_x = sym_state[7].as_decimal(5)
        gripper_vel_y = sym_state[8].as_decimal(5)
        gripper_vel_z = sym_state[9].as_decimal(5)
        block_vel_x = sym_state[10].as_decimal(5)
        block_vel_y = sym_state[11].as_decimal(5)
        block_vel_z = sym_state[12].as_decimal(5)
        goal_x = [c.as_decimal(5) for c in sym_state[13]]
        goal_y = [c.as_decimal(5) for c in sym_state[14]]
        goal_z = [c.as_decimal(5) for c in sym_state[15]]
        
        out += ("-"*20) + "\n"
        out += f"t={t}, Gripper Pos: ({gripper_x}, {gripper_y}, {gripper_z}), Gripper Disp: {gripper_disp}\n"
        for i in range(2):
            out += f"Goal {MUJOCO_IDX_TO_COLOR[i]} Pos: ({goal_x[i]}, {goal_y[i]}, {goal_z[i]})\n"
        out += f"Gripper Vel: ({gripper_vel_x}, {gripper_vel_y}, {gripper_vel_z})\n"
        out += f"Block Pos: ({block_x}, {block_y}, {block_z})\n"
        out += f"Block Vel: ({block_vel_x}, {block_vel_y}, {block_vel_z})\n"
        out += f"subtask_complete_func={bool(subtask_complete_traj[t])}\n"
        if near_object_traj is not None:
            out += f"near_object_func={bool(near_object_traj[t])}\n"
        out += ("-"*20) + "\n"

        
    return out

def minigrid_state_mission_to_z3(state: np.ndarray, mission_str: str) -> Tuple[Tuple[z3.ExprRef, ...], Tuple[z3.ExprRef, ...]]:
    agent_pos = None
    agent_dir = None
    grid_size_x, grid_size_y, _ = state.shape

    door_x_colors = [-1] * 6
    door_y_colors = [-1] * 6
    door_state_colors = [-1] * 6
    key_x_colors = [-1] * 6
    key_y_colors = [-1] * 6
    box_x_colors = [-1] * 6
    box_y_colors = [-1] * 6
    box_contains = [-1] * 6
    goal_x = -1
    goal_y = -1

    for i in range(grid_size_x):
        for j in range(grid_size_y):
            tile = state[i, j]
            obj_type = tile[0]
            color_type = tile[1]
            if obj_type == MINIGRID_OBJECT_TO_IDX["agent"]:
                agent_pos = (i, j)
                agent_dir = tile[2]
            elif obj_type == MINIGRID_OBJECT_TO_IDX["door"]:
                door_x_colors[color_type] = i
                door_y_colors[color_type] = j
                door_state_colors[color_type] = tile[2]
            elif obj_type == MINIGRID_OBJECT_TO_IDX["key"]:
                key_x_colors[color_type] = i
                key_y_colors[color_type] = j
            elif obj_type == MINIGRID_OBJECT_TO_IDX["box"]:
                box_x_colors[color_type] = i
                box_y_colors[color_type] = j
            elif obj_type == MINIGRID_OBJECT_TO_IDX["goal"]:
                goal_x = i
                goal_y = j

    # remove punctuation
    clean = re.sub(r"[^\w\s]", "", mission_str.lower())
    key_strings = re.findall(r"(\w+)\s+(key)", clean)
    door_strings = re.findall(r"(\w+)\s+(door|room)", clean)
    box_strings = re.findall(r"(\w+)\s+(box)", clean)
    
    mission_keys = [z3.IntVal(MINIGRID_COLOR_TO_IDX[color] if color in MINIGRID_COLOR_TO_IDX else 4) for color, _ in key_strings]
    mission_doors = [z3.IntVal(MINIGRID_COLOR_TO_IDX[color] if color in MINIGRID_COLOR_TO_IDX else 4) for color, _ in door_strings]
    mission_boxes = [z3.IntVal(MINIGRID_COLOR_TO_IDX[color] if color in MINIGRID_COLOR_TO_IDX else 4) for color, _ in box_strings]

    if mission_boxes:
        box_contains[mission_boxes[0].as_long()] = mission_keys[0].as_long()
    
    sym_state = (
        z3.IntVal(agent_pos[0]),
        z3.IntVal(agent_pos[1]),
        z3.IntVal(agent_dir),
        [z3.IntVal(c) for c in door_x_colors],
        [z3.IntVal(c) for c in door_y_colors],
        [z3.IntVal(c) for c in door_state_colors],
        [z3.IntVal(c) for c in key_x_colors],
        [z3.IntVal(c) for c in key_y_colors],
        [z3.IntVal(c) for c in box_x_colors],
        [z3.IntVal(c) for c in box_y_colors],
        [z3.IntVal(c) for c in box_contains],
        z3.IntVal(goal_x),
        z3.IntVal(goal_y),
    )
    return sym_state, (mission_keys, mission_doors, mission_boxes)

def mujoco_state_mission_to_z3(state: dict, mission_str: str) -> Tuple[Tuple[z3.ExprRef, ...], Tuple[z3.ExprRef, ...]]:
    observation = state["observation"]
    achieved_goal = state["achieved_goal"]
    desired_goal = state["desired_goal"]
    
    color_id = 0
    for i in MUJOCO_IDX_TO_COLOR:
        if MUJOCO_IDX_TO_COLOR[i] in mission_str:
            color_id = i
            break
    sym_mission_goals = (z3.IntVal(color_id),)
    
    sym_state = (
        z3.RealVal(observation[0]), # x
        z3.RealVal(observation[1]), # y
        z3.RealVal(observation[2]), # z
        z3.RealVal(observation[9]), # gripper displacement
        z3.RealVal(achieved_goal[0]), # block x
        z3.RealVal(achieved_goal[1]), # block y
        z3.RealVal(achieved_goal[2]), # block z
        z3.RealVal(observation[20]), # gripper vel x
        z3.RealVal(observation[21]), # gripper vel y
        z3.RealVal(observation[22]), # gripper vel z
        z3.RealVal(observation[21] + observation[14]), # block vel x
        z3.RealVal(observation[22] + observation[15]), # block vel y
        z3.RealVal(observation[23] + observation[16]), # block vel z
        [z3.RealVal(desired_goal[0]), z3.RealVal(desired_goal[3] if len(desired_goal) > 3 else -5)], # green goal x
        [z3.RealVal(desired_goal[1]), z3.RealVal(desired_goal[4] if len(desired_goal) > 3 else -5)], # green goal y
        [z3.RealVal(desired_goal[2]), z3.RealVal(desired_goal[5] if len(desired_goal) > 3 else -5)], # green goal z
    )
    return sym_state, sym_mission_goals

def minigrid_object_to_str(obj_tuple: Tuple[int, int]) -> str:
    """
    Convert object tuple to string representation.

    :param obj_tuple: Tuple of (OBJECT_IDX, COLOR_IDX) coordinates.
    :return: String representation of the object.
    """
    return (
        f"{MINIGRID_IDX_TO_COLOR[obj_tuple[1]]} {MINIGRID_IDX_TO_OBJECT[obj_tuple[0]]}"
        if obj_tuple[1] != -1
        else f"{MINIGRID_IDX_TO_OBJECT[obj_tuple[0]]}"
    )


def minigrid_create_distance_function(obj_idx: Tuple[int, int]) -> float:
    def distance_function(state: np.ndarray) -> float:
        agent_pos = None
        obj_pos = None
        for i in range(state.shape[0]):
            for j in range(state.shape[1]):
                if state[i, j, 0] == MINIGRID_OBJECT_TO_IDX["agent"]:
                    agent_pos = (i, j)
                if state[i, j, 0] == obj_idx[0] and (
                    obj_idx[1] == -1 or state[i, j, 1] == obj_idx[1]
                ):
                    obj_pos = (i, j)
        if agent_pos is None or obj_pos is None:
            return 0.0
        return np.sum(np.abs(np.array(agent_pos) - np.array(obj_pos)))

    return distance_function


def pprint_grid(state: np.ndarray) -> str:
    """
    Produce a pretty string of the environment's grid along with the agent.
    A grid cell is represented by 2-character string, the first one for
    the object and the second one for the color.
    """
    agent_pos = None
    agent_dir = None
    for i in range(state.shape[0]):
        for j in range(state.shape[1]):
            if state[i, j, 0] == MINIGRID_OBJECT_TO_IDX["agent"]:
                agent_pos = (i, j)
                agent_dir = state[i, j, 2]

    # Map of object types to short string
    OBJECT_TO_STR = {
        "wall": "W",
        "floor": "F",
        "door": "D",
        "key": "K",
        "ball": "A",
        "box": "B",
        "goal": "G",
        "lava": "V",
    }

    # Map agent's direction to short string
    AGENT_DIR_TO_STR = {0: ">", 1: "V", 2: "<", 3: "^"}

    output = ""

    for j in range(state.shape[1]):
        for i in range(state.shape[0]):
            if i == agent_pos[0] and j == agent_pos[1]:
                output += 2 * AGENT_DIR_TO_STR[agent_dir]
                continue

            tile = state[i, j]

            if tile[0] == MINIGRID_OBJECT_TO_IDX["empty"]:
                output += "  "
                continue

            if tile[0] == MINIGRID_OBJECT_TO_IDX["door"]:
                if tile[2] == MINIGRID_STATE_TO_IDX["open"]:
                    output += "__"
                elif tile[2] == MINIGRID_STATE_TO_IDX["locked"]:
                    output += "L" + MINIGRID_IDX_TO_COLOR[tile[1]][0].upper()
                else:
                    output += "D" + MINIGRID_IDX_TO_COLOR[tile[1]][0].upper()
                continue

            output += (
                OBJECT_TO_STR[MINIGRID_IDX_TO_OBJECT[tile[0]]]
                + MINIGRID_IDX_TO_COLOR[tile[1]][0].upper()
            )

        if j < state.shape[1] - 1:
            output += "\n"

    return output


def mujoco_object_to_str(obj_tuple: Tuple[int, int]) -> str:
    """
    Convert object tuple to string representation.

    :param obj_tuple: Tuple of (OBJECT_IDX, COLOR_IDX) coordinates.
    :return: String representation of the object.
    """
    return (
        f"{MUJOCO_IDX_TO_COLOR[obj_tuple[1]]} {MUJOCO_IDX_TO_OBJECT[obj_tuple[0]]}"
        if obj_tuple[1] != -1
        else f"{MUJOCO_IDX_TO_OBJECT[obj_tuple[0]]}"
    )


def mujoco_create_distance_function(obj_idx: Tuple[int, int]) -> float:
    def distance_function(state: dict) -> float:
        object_id, color_id = obj_idx
        if color_id == -1:
            color_id = 0
        if object_id:
            desired_goal = state["desired_goal"][3 * color_id : 3 * (color_id + 1)]
            achieved_goal = state["achieved_goal"]
            return np.linalg.norm(desired_goal[:2] - achieved_goal[:2]) - 0.05
        else:
            agent_position = state["observation"][:3]
            block_position = state["achieved_goal"]
            return np.linalg.norm(agent_position[:2] - block_position[:2])

    return distance_function

def amplify_available_models(base_url, api_key):
    url = f"{base_url}/available_models"
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return [output for output in response.json()["data"]["models"]]
    else:
        print(f"Failed to retrieve models: {response.status_code}")
        return []


def amplify_make_request(base_url, api_key, messages, ai_model="gpt-5", reasoningLevel="medium", max_tokens=128000):
    # URL for the Azure API
    url = f"{base_url}/chat"

    #ai_model = 'gpt-4o'
    api_key = os.environ.get("AMPLIFY_API_KEY")
    # Headers
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f"Bearer {api_key}"  # Replace "key" with your actual access token
    }
    
    # Data payload
    payload = {
        "data": {
            "model": ai_model,  # Replace with the model you want to use
            "temperature": 0.5,
            "max_tokens": max_tokens,
            "dataSources": [],
            "messages": messages,
            "options": {
                "ragOnly": False,
                "skipRag": True,
                "model": {"id" : ai_model},
                "reasoningLevel": reasoningLevel,
            }
        }
    }
    
    # Make the POST request
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    
    # Check for a successful response
    if response.status_code == 200:
        # Parse the JSON response
        response_data = response.json()
        txt = response_data.get("data", "")
        # Return the data entry if it exists
        return txt
    else:
        print(f"Request failed with status code {response.status_code}")
        print(response.text)
        response.raise_for_status()
        return None

def replace_between_functions(
    file_path: str,
    start_func: Optional[str],
    end_func: str,
    replacement: str,
):
    source = Path(file_path).read_text()
    lines = source.splitlines(keepends=True)

    tree = ast.parse(source)

    start_end_lineno = None
    end_start_lineno = None

    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if start_func is not None and node.name == start_func:
                # end_lineno exists in Python 3.8+
                start_end_lineno = node.end_lineno
            elif node.name == end_func:
                end_start_lineno = node.lineno

    if (start_func is not None and start_end_lineno is None) or end_start_lineno is None:
        raise ValueError("Could not find one or both functions")

    if start_func is None:
        start_end_lineno = 0
    
    new_lines = (
        lines[:start_end_lineno] +
        [replacement + "\n"] +
        lines[end_start_lineno - 1:]
    )

    Path(file_path).write_text("".join(new_lines))

