from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional, Tuple, List
import z3
import numpy as np
from collections import namedtuple
from bt_as_reward.rewards.bt import BehaviourTreeConfig

SubtaskZ3Output = namedtuple(
    "SubtaskZ3Output",
    ["positive_response", "negative_response"],
)

ObjectZ3Output = namedtuple(
    "ObjectZ3Output",
    ["positive_response", "negative_response"],
)

CompositionZ3Output = namedtuple(
    "CompositionZ3Output",
    ["positive_response"]
)

class SubtaskZ3Verifier(ABC):
    """
    Abstract base class for subtask Z3 verifiers.
    All verifiers should inherit from this class
    and implement the `_verify_positive` and `_verify_negative` methods.
    """

    @classmethod
    @abstractmethod
    def _verify_positive(
        cls, 
        env_name: str, 
        subtask_function: Callable[[Tuple, Tuple], z3.ExprRef], 
        mission_args: Dict,
        expert_trajs: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Verify that if the mission is eventually complete then the subtask function returns True.

        :param subtask_function: Function to check if a subtask is completed.
        :return: Re-prompt string if no such input exists, None o/w.
        """
        pass

    @classmethod
    @abstractmethod
    def _verify_negative(
        cls, 
        env_name: str, 
        subtask_function: Callable[[Tuple, Tuple], z3.ExprRef], 
        mission_args: Dict, n_trajs: int,
        random_trajs: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Verify that there exists n_trajs trajectories where the subtask remains False.

        :param subtask_function: Function to check if a subtask is completed.
        :param n_trajs: Number of trajectories to find.
        :return: Re-prompt string if no such input exists, None o/w.
        """
        pass

    @classmethod
    def verify(
        cls, 
        env_name: str, 
        subtask_function: Callable[[np.ndarray, str], bool], 
        mission_args: Dict,
        expert_trajs: Optional[Dict] = None,
        random_trajs: Optional[Dict] = None,
    ) -> SubtaskZ3Output:
        """
        Verify the given subtask function.

        :param subtask_function: Function to check if a subtask is completed.
        :return: SubtaskZ3Output.
        """
        return SubtaskZ3Output(
            positive_response=cls._verify_positive(env_name, subtask_function, mission_args, expert_trajs=expert_trajs),
            negative_response=cls._verify_negative(env_name, subtask_function, mission_args, n_trajs=10, random_trajs=random_trajs),
        )

class ObjectZ3Verifier(ABC):
    """
    Abstract base class for object Z3 verifiers.
    All object verifiers should inherit from this class and implement the `_verify_positive` and `_verify_negative` methods.
    """

    @classmethod
    @abstractmethod
    def _verify_positive(
        cls,
        env_name: str,
        object_function: Callable[[Tuple, Tuple], z3.ExprRef],
        subtask_function: Callable[[Tuple, Tuple], z3.ExprRef],
        mission_args: Dict,
        expert_trajs: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Verify that the agent is within proximity to the subtask object immediately before it is completed.

        :param object_function: Function to check for proximity to the object.
        :param subtask_function: Function to check if a subtask is completed.
        :return: Re-prompt string if no such input exists, None o/w.
        """
        pass

    @classmethod
    @abstractmethod
    def _verify_negative(
        cls,
        env_name: str,
        object_function: Callable[[Tuple, Tuple], z3.ExprRef],
        subtask_function: Callable[[Tuple, Tuple], z3.ExprRef],
        mission_args: Dict,
        n_trajs: int,
        random_trajs: Optional[Dict] = None,
    ) -> Optional[str]:
        """
        Verify that there exists n_trajs trajectories where the agent does not reach `in proximity` to the object detected by the function.

        :param object_function: Function to check for proximity to the object.
        :param subtask_function: Function to check if a subtask is completed.
        :param n_trajs: Number of trajectories to find.
        :return: Re-prompt string if no such input exists, None o/w.
        """
        pass

    @classmethod
    def verify(
        cls,
        env_name: str,
        object_function: Callable[[Tuple, Tuple], z3.ExprRef],
        subtask_function: Callable[[Tuple, Tuple], z3.ExprRef],
        mission_args: Dict,
        expert_trajs: Optional[Dict] = None,
        random_trajs: Optional[Dict] = None,
    ) -> ObjectZ3Output:
        """
        Verify the given object detection function.

        :param object_function: Function to check for proximity to the object.
        :param subtask_function: Function to check if a subtask is completed.
        :return: ObjectZ3Output.
        """
        return ObjectZ3Output(
            positive_response=cls._verify_positive(
                env_name, object_function, subtask_function, mission_args, expert_trajs=expert_trajs
            ),
            negative_response=cls._verify_negative(
                env_name, object_function, subtask_function, mission_args, n_trajs=10, random_trajs=random_trajs
            ),
        )

class CompositionZ3Verifier(ABC):
    """
    Abstract base class for composition Z3 verifiers.
    All composition verifiers should inherit from this class and implement the `verify` method.
    """

    @classmethod
    @abstractmethod
    def verify(
        cls,
        env_name: str,
        subtask_functions: List[Callable[[Tuple, Tuple], z3.ExprRef]],
        mission_args: Dict,
        n_trajs: int=10,
        expert_trajs: Optional[Dict] = None,
    ) -> CompositionZ3Output:
        """
        Verify there exists n_trajs trajectories where all subtasks persist completion if the mission is eventually complete.
        
        :param subtask_functions: List of subtask functions.
        :return: CompositionZ3Output.
        """
        pass


SubtaskVerifyOutput = namedtuple(
    "VerifyOutput",
    ["expert_response", "reactive_response", "random_response"],
)

ObjectVerifyOutput = namedtuple(
    "ObjectVerifyOutput",
    ["response"],
)

BTVerifyOutput = namedtuple(
    "BTVerifyOutput",
    ["expert_response", "random_response"],
)


class SubtaskVerifier(ABC):
    """
    Abstract base class for subtask verifiers.
    All verifiers should inherit from this class
    and implement the `_verify_expert_trajs`, `_verify_reactive_trajs`, and `_verify_random_trajs` methods.
    """

    @classmethod
    @abstractmethod
    def _verify_expert_trajs(
        cls, expert_trajs: Dict, subtask_function: Callable[[np.ndarray, str], bool]
    ) -> Optional[str]:
        """
        Verify expert trajectories for non-reactive task. Subtask function should return True in all expert trajs.

        :param expert_trajs: Dictionary of expert trajectories.
        :param subtask_function: Function to check if a subtask is completed.
        :return: Re-prompt string if any expert traj fails, None o/w.
        """
        pass

    @classmethod
    @abstractmethod
    def _verify_reactive_trajs(
        cls,
        expert_trajs: Dict,
        random_trajs: Dict,
        subtask_function: Callable[[np.ndarray, str], bool],
    ) -> Optional[str]:
        """
        Verify trajectories to ensure subtask function is reactive on input only.
        Subtask function should return same response invariant to index in the trajectory.
        Eliminates functions with external dependencies / hidden state that change over time.
        These functions can lead to "sticky" behavior and non-reactive responses.

        :param expert_trajs: Dictionary of expert trajectories.
        :param random_trajs: Dictionary of random trajectories.
        :param subtask_function: Function to check if a subtask is completed.
        :return: Re-prompt string if no switch in any expert traj, None o/w.
        """
        pass

    @classmethod
    @abstractmethod
    def _verify_random_trajs(
        self, random_trajs: Dict, subtask_function: Callable[[np.ndarray, str], bool]
    ) -> Optional[str]:
        """
        Verify random trajectories. Subtask function should return True in less than p% of the random trajs.

        :param random_trajs: Dictionary of random trajectories.
        :param subtask_function: Function to check if a subtask is completed.
        :return: Re-prompt string if greater than p%, None o/w.
        """
        pass

    @classmethod
    def verify(
        cls,
        expert_trajs: Dict,
        random_trajs: Dict,
        subtask_function: Callable[[np.ndarray, str], bool],
        random_threshold: float = 0.5,
    ) -> SubtaskVerifyOutput:
        """
        Verify the given subtask function.

        :param expert_trajs: Dictionary of expert trajectories.
        :param random_trajs: Dictionary of random trajectories.
        :param subtask_function: Function to check if a subtask is completed.
        :param random_threshold: Threshold for random trajectories verification.
        :return: True if the data is valid, False otherwise.
        """
        return SubtaskVerifyOutput(
            expert_response=cls._verify_expert_trajs(expert_trajs, subtask_function),
            reactive_response=cls._verify_reactive_trajs(
                expert_trajs, random_trajs, subtask_function
            ),
            random_response=cls._verify_random_trajs(
                random_trajs, subtask_function, random_threshold
            ),
        )


class ObjectVerifier(ABC):
    """
    Abstract base class for object verifiers.
    All object verifiers should inherit from this class and implement the `verify` and `_near_object` methods.
    """

    @classmethod
    @abstractmethod
    def _near_object(
        cls,
        state: np.ndarray,
        mission_str: str,
        object_function: Callable[[str], Tuple[int, int]],
        distance_threshold: float = 1.0,
    ) -> bool:
        """
        Verify if the agent is in proximity to the object detected by the function.

        :param state: The current state of the environment.
        :param mission_str: The mission string describing the task.
        :param object_function: Function to detect the object in the state.
        :param distance_threshold: Distance threshold for proximity.
        :return: True if the agent is in proximity to the object, False otherwise.
        """
        pass

    @classmethod
    @abstractmethod
    def verify(
        cls,
        expert_trajs: Dict,
        object_function: Callable[[str], Tuple[int, int]],
        subtask_function: Callable[[np.ndarray, str], bool],
        distance_threshold: Optional[float] = 1.0,
    ) -> ObjectVerifyOutput:
        """
        Verify if the object detection function.
        The agent reach `in proximity` to the object detected by the function, before the subtask is complete.

        :param state: The current state of the environment.
        :param mission_str: The mission string describing the task.
        :return: True if the object is present, False otherwise.
        """
        pass


class BTRewardVerifier(ABC):
    """
    Abstract base class for behaviour tree reward verifiers.
    All verifiers should inherit from this class and implement the `verify` method.
    """

    @classmethod
    @abstractmethod
    def _verify_expert_trajs(
        cls, expert_trajs: Dict, max_reward: float, bt_config: BehaviourTreeConfig
    ) -> Optional[str]:
        """
        Verify the expert trajectories for the behaviour tree reward.

        :param expert_trajs: Dictionary of expert trajectories.
        :param max_reward: Maximum reward achievable in the environment.
        :param bt_config: Configuration for the behaviour tree.
        :return: Re-prompt string if any expert traj fails, None o/w.
        """
        pass

    def _verify_random_trajs(
        self,
        random_trajs: Dict,
        max_reward: float,
        bt_config: BehaviourTreeConfig,
        random_threshold: float = 0.5,
    ) -> Optional[str]:
        """
        Verify the random trajectories for the behaviour tree reward.

        :param random_trajs: Dictionary of random trajectories.
        :param max_reward: Maximum reward achievable in the environment.
        :param bt_config: Configuration for the behaviour tree.
        :param random_threshold: Threshold for random trajectories verification.
        :return: Re-prompt string if any random traj fails, None o/w.
        """
        pass

    @classmethod
    def verify(
        cls,
        expert_trajs: Dict,
        random_trajs: Dict,
        max_reward: float,
        bt_config: BehaviourTreeConfig,
        random_threshold: float = 0.5,
    ) -> BTVerifyOutput:
        """
        Verify the behaviour tree reward based on the expert and random trajectories.

        :param expert_trajs: Dictionary of expert trajectories.
        :param random_trajs: Dictionary of random trajectories.
        :param max_reward: Maximum reward achievable in the environment.
        :param bt_config: Configuration for the behaviour tree.
        :param random_threshold: Threshold for random trajectories verification.
        :return: True if the behaviour tree is valid, False otherwise.
        """
        return BTVerifyOutput(
            expert_response=cls._verify_expert_trajs(
                expert_trajs, max_reward, bt_config
            ),
            random_response=cls._verify_random_trajs(
                random_trajs, max_reward, bt_config, random_threshold
            ),
        )
