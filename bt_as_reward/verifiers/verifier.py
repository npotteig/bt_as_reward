from abc import ABC, abstractmethod
from typing import Callable, Dict, Optional
import numpy as np
from collections import namedtuple

VerifyOutput = namedtuple(
    "VerifyOutput",
    ["expert_nonreactive_response", "expert_reactive_response", "random_response"],
)


class Verifier(ABC):
    """
    Abstract base class for verifiers.
    All verifiers should inherit from this class and implement the `verify` method.
    """

    @classmethod
    @abstractmethod
    def _verify_expert_trajs_nonreactive(
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
    def _verify_expert_trajs_reactive(
        cls, expert_trajs: Dict, subtask_function: Callable[[np.ndarray, str], bool]
    ) -> Optional[str]:
        """
        Verify expert trajectories for reactive task.
        Subtask function should switch from True to False in ALL expert trajs if subtask fails after succeeding (i.e. key drops).

        :param expert_trajs: Dictionary of expert trajectories.
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
    ) -> VerifyOutput:
        """
        Verify the given data.

        :param data: The data to be verified.
        :return: True if the data is valid, False otherwise.
        """
        return VerifyOutput(
            expert_nonreactive_response=cls._verify_expert_trajs_nonreactive(
                expert_trajs, subtask_function
            ),
            expert_reactive_response=cls._verify_expert_trajs_reactive(
                expert_trajs, subtask_function
            ),
            random_response=cls._verify_random_trajs(
                random_trajs, subtask_function, random_threshold
            ),
        )
