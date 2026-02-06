from typing import Sequence, Callable, Tuple, Dict, Optional, List, Any
import numpy as np
import py_trees as pt
from pydantic import BaseModel
import z3

from bt_as_reward.rewards.behaviours import (
    SubtaskCompleted,
    GoToObject,
    InteractObject,
    FailUntilSubtask,
)

"""
Masking Reward Behavior Tree (MRBT) Template Configuration and Creation

Template Structure:
Sequence
- Subtask 1
- Subtask 2
- Subtask 3
...
- Subtask N

Selector
- Subtask complete? (B(c_i))
- Sequence
	- GoTo Object! (B(d_i))
	- Interact Object! (B(bot))
"""


class BehaviourTreeConfig(BaseModel):
    """
    Configuration for creating a behaviour tree.

    :param subtask_names: List of subtask names.
    :param subtask_functions: List of functions to check if a subtask is completed.
    :param object_functions: List of functions to detect objects.
    :param object_to_str: Function to convert object indices to a string.
    :param create_distance_function: Function to create distance functions for objects.
    :param dependent_subtasks: Optional mapping of subtasks to their dependent subtasks.
    :param distance_threshold: Distance threshold for proximity checks.
    :param use_memory: Whether to use memory in the behaviour tree sequence nodes.
    """

    subtask_names: Sequence[str]
    subtask_functions: Sequence[Callable[[np.ndarray, str], bool]]
    object_functions: Sequence[Callable[[str], Tuple[int, int]]]
    object_to_str: Callable[[Tuple[int, int]], str]
    create_distance_function: Callable[[Tuple[int, int]], Callable[[np.ndarray], float]]
    dependent_subtasks: Optional[
        Dict[Callable[[Any, str], bool], List[Callable[[Any, str], bool]]]
    ] = None
    distance_threshold: float
    use_memory: bool = False


class BehaviourTreeReward:
    def __init__(self, root: pt.behaviour.Behaviour, blackboard: pt.blackboard.Client, state_mission_to_z3: Optional[Callable[[np.ndarray, str], z3.BoolRef]] = None):
        self.root = root
        self.root.setup_with_descendants()
        self.blackboard = blackboard
        self.state_mission_to_z3 = state_mission_to_z3
        
    def step_reward(
        self, state: np.ndarray, mission_str: str
    ) -> Tuple[float, np.ndarray]:
        """
        Step the behaviour tree and return the reward based on the updates of the behaviour nodes.
        """
        if self.state_mission_to_z3:
            state, mission_str = self.state_mission_to_z3(state, mission_str)
        self.blackboard.set("state", state)
        self.blackboard.set("mission_str", mission_str)
        self.blackboard.set("reward", 0.0)
        self.blackboard.set("action_mask", None)
        self.root.tick_once()
        return self.blackboard.reward, self.blackboard.action_mask

    @classmethod
    def _create_subtask(
        cls,
        subtask_name: str,
        object_name: str,
        subtask_function: Callable[[np.ndarray, str], bool],
        distance_function: Callable[[np.ndarray], float],
        distance_threshold: float,
        use_memory: bool = False,
        action_masks: Optional[Tuple[np.ndarray, np.ndarray]] = None,
        use_z3: bool = False,
    ) -> pt.behaviour.Behaviour:
        """
        Create a subtask behaviour that checks if the subtask is completed
        and if the object is available.
        """
        return pt.composites.Selector(
            name=subtask_name,
            children=[
                SubtaskCompleted(
                    name=f"{subtask_name}?", subtask_function=subtask_function, use_z3=use_z3
                ),
                pt.composites.Sequence(
                    name=f"Sequence {object_name}",
                    children=[
                        GoToObject(
                            name=f"GoTo {object_name}!",
                            distance_function=distance_function,
                            distance_threshold=distance_threshold,
                            action_mask=action_masks[0] if action_masks else None,
                            use_z3=use_z3,
                        ),
                        InteractObject(
                            name=f"Interact {object_name}!",
                            action_mask=action_masks[1] if action_masks else None,
                        ),
                    ],
                    memory=use_memory,
                ),
            ],
            memory=False,
        )

    @classmethod
    def create_bt(
        cls,
        mission_str: str,
        bt_config: BehaviourTreeConfig,
        action_masks: Optional[Dict[str, List[int]]] = None,
        state_mission_to_z3: Optional[Callable[[np.ndarray, str], z3.BoolRef]] = None,
    ):
        """
        Create a behaviour tree with the given subtask names and functions.

        :param initial_state: Initial state of the environment.
        :param mission_str: Mission string describing the task.
        :param bt_config: Configuration for the behaviour tree.
        :param action_masks: Optional action masks to restrict available actions.
        :return: A BehaviourTreeReward instance with the constructed behaviour tree.
        """
        assert (
            len(bt_config.subtask_names)
            == len(bt_config.subtask_functions)
            == len(bt_config.object_functions)
        ), "Subtask names, functions, and object functions must have the same length."
        blackboard = pt.blackboard.Client(name="Blackboard")
        blackboard.register_key(key="state", access=pt.common.Access.WRITE)
        blackboard.register_key(key="mission_str", access=pt.common.Access.WRITE)
        blackboard.register_key(key="reward", access=pt.common.Access.WRITE)
        blackboard.register_key(key="action_mask", access=pt.common.Access.WRITE)
        blackboard.set("mission_str", mission_str)
        blackboard.set("reward", 0.0)
        root = pt.composites.Sequence(name="Root", memory=bt_config.use_memory)
        for subtask_name, subtask_function, object_function in zip(
            bt_config.subtask_names,
            bt_config.subtask_functions,
            bt_config.object_functions,
        ):
            subtask_child = cls._create_subtask(
                subtask_name=subtask_name,
                object_name=bt_config.object_to_str(object_function(mission_str)) if state_mission_to_z3 is None else object_function.__name__,
                subtask_function=subtask_function,
                distance_function=bt_config.create_distance_function(
                    object_function(mission_str)
                ) if state_mission_to_z3 is None else object_function,
                distance_threshold=bt_config.distance_threshold,
                use_memory=bt_config.use_memory,
                action_masks=(
                    np.array(
                        action_masks.get(f"goto_{object_function.__name__}", None),
                        dtype=np.int32,
                    ),
                    np.array(
                        action_masks.get(f"interact_{object_function.__name__}", None),
                        dtype=np.int32,
                    ),
                )
                if action_masks
                else None,
                use_z3=state_mission_to_z3 is not None,
            )
            if (
                bt_config.dependent_subtasks is not None
                and bt_config.dependent_subtasks.get(subtask_function, [])
            ):
                dependent_string = "or ".join(
                    [
                        k.__name__
                        for k in bt_config.dependent_subtasks.get(subtask_function, [])
                    ]
                )
                subtask_child.children.insert(
                    0,
                    FailUntilSubtask(
                        name=f"Fail Until {dependent_string}",
                        dependent_subtasks=bt_config.dependent_subtasks.get(
                            subtask_function, []
                        ),
                    ),
                )
            root.add_child(subtask_child)
        return cls(root=root, blackboard=blackboard, state_mission_to_z3=state_mission_to_z3)
