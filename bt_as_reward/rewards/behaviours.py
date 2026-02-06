import py_trees as pt
from typing import Callable, Any
import numpy as np
import z3

"""
Behaviors for Masking Reward Behavior Trees (MRBTs)

Each behavior is equivalent to a masking behavior reward machine (MBRM).
Each MBRM outputs one action mask for all states (delta_eta(u, sigma) = eta_b for all u in U)

U (states):
S
F/R - Failure (if SubtaskComplete) or Running (if GoToObject/InteractObject)

u_0 - initial state: F/R

Propositions:
c_i - Subtask i is complete
d_i - Agent is within distance threshold of object for subtask i

Transitions:
F/R -- <rho, 0.5> --> S
F/R -- <rho, 0.0> --> F/R
S -- <rho, -0.5> --> F/R
S -- <rho, 0.0> --> S

Behavior RMs B(rho):

B(c_i): SubtaskCompleted
B(d_i): GoToObject
B(bot): InteractObject

Used to modify MRBT from BehaVerify feedback:
B(c_j ^ c_k ^ ...): FailUntilSubtask checks if any dependent subtasks are complete
"""


# inputs a list of dependent subtasks and check if they are complete. If so, output SUCCESS, else output what the decorated child status is.
class FailUntilSubtask(pt.behaviour.Behaviour):
    def __init__(self, name: str, dependent_subtasks: list[Callable[[Any, str], bool]]):
        super().__init__(name=name)
        self.dependent_subtasks = dependent_subtasks

        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key(key="state", access=pt.common.Access.READ)
        self.blackboard.register_key(key="mission_str", access=pt.common.Access.READ)

    def update(self):
        if any(
            [
                k(self.blackboard.state, self.blackboard.mission_str)
                for k in self.dependent_subtasks
            ]
        ):
            return pt.common.Status.SUCCESS
        return pt.common.Status.FAILURE


class SubtaskCompleted(pt.behaviour.Behaviour):
    """
    A behaviour that checks if a subtask is completed.
    """

    def __init__(self, 
                 name: str, 
                 subtask_function: Callable[[np.ndarray, str], bool],
                 use_z3: bool = False):
        super().__init__(name=name)
        self.subtask_function = subtask_function
        self.use_z3 = use_z3

        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key(key="state", access=pt.common.Access.READ)
        self.blackboard.register_key(key="mission_str", access=pt.common.Access.READ)
        self.blackboard.register_key(key="reward", access=pt.common.Access.WRITE)

    def setup(self, **kwargs):
        self.memory_status = pt.common.Status.FAILURE

    def update(self):
        res = False
        if self.use_z3:
            s = z3.Solver()
            s.add(self.subtask_function(self.blackboard.state, self.blackboard.mission_str))
            res = s.check() == z3.sat
        else:
            res = self.subtask_function(self.blackboard.state, self.blackboard.mission_str)
        if res:
            if self.memory_status == pt.common.Status.FAILURE:
                self.blackboard.reward += 0.5
            self.memory_status = pt.common.Status.SUCCESS
            return pt.common.Status.SUCCESS
        if self.memory_status == pt.common.Status.SUCCESS:
            self.blackboard.reward += -0.5
        self.memory_status = pt.common.Status.FAILURE
        return pt.common.Status.FAILURE


class GoToObject(pt.behaviour.Behaviour):
    """
    A behaviour that checks if agent is within proximity to object.
    """

    def __init__(
        self,
        name: str,
        distance_function: Callable[[np.ndarray], float],
        distance_threshold: float,
        action_mask: np.ndarray = None,
        use_z3: bool = False,
    ):
        super().__init__(name=name)
        self.distance_function = distance_function
        self.distance_threshold = distance_threshold
        self.action_mask = action_mask
        self.use_z3 = use_z3

        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key(key="state", access=pt.common.Access.READ)
        self.blackboard.register_key(key="mission_str", access=pt.common.Access.READ)
        self.blackboard.register_key(key="reward", access=pt.common.Access.WRITE)
        self.blackboard.register_key(key="action_mask", access=pt.common.Access.WRITE)

    def setup(self, **kwargs):
        self.memory_status = pt.common.Status.RUNNING

    def update(self):
        res = False
        if self.use_z3:
            s = z3.Solver()
            s.add(self.distance_function(self.blackboard.state, self.blackboard.mission_str))
            res = s.check() == z3.sat
        else:
            res = self.distance_function(self.blackboard.state) <= self.distance_threshold
        if res:
            if self.memory_status == pt.common.Status.RUNNING:
                # Reward for completion
                self.blackboard.reward += 0.5
            self.memory_status = pt.common.Status.SUCCESS
            return pt.common.Status.SUCCESS
        if self.memory_status == pt.common.Status.SUCCESS:
            # Enable reward penalty
            self.blackboard.reward += -0.5
        self.blackboard.action_mask = self.action_mask
        self.memory_status = pt.common.Status.RUNNING
        return pt.common.Status.RUNNING


class InteractObject(pt.behaviour.Behaviour):
    """
    A behaviour that checks if the agent can interact with the object.
    """

    def __init__(self, name: str, action_mask: np.ndarray = None):
        super().__init__(name=name)

        self.action_mask = action_mask
        self.blackboard = self.attach_blackboard_client(name="Blackboard")
        self.blackboard.register_key(key="action_mask", access=pt.common.Access.WRITE)

    def update(self):
        self.blackboard.action_mask = self.action_mask
        return pt.common.Status.RUNNING
