"""A simple 1D "Go Right" environment illustrating a second-order Markov status transition.

The environment is a 1D grid of configurable length. The agent can move left or right.
The status indicator transitions according to a second-order Markov chain, and the agent observes
the prize indicators by reaching the rightmost grid position. When the agent reaches the last state
with the status indicator at maximum intensity, the prize indicators go on and the agent receives the
reward. 
"""

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from gymnasium import Env, spaces

from goright.utils import State


# Actions
LEFT = 0
RIGHT = 1

# Second-order Markov transitions for the status indicator
STATUS_TRANSITION = {
    (0, 0): 5,
    (0, 5): 0,
    (0, 10): 5,
    (5, 0): 10,
    (5, 5): 10,
    (5, 10): 10,
    (10, 0): 0,
    (10, 5): 5,
    (10, 10): 0,
}


class GoRight(Env):
    """A custom Gymnasium environment for a simple 1D "Go Right" task.

    The agent is placed on a 1D grid of length `env_length`.
    Each step, the agent can move LEFT or RIGHT.
    The 'status_indicator' variable transitions according to a second-order Markov chain,
    where the next value depends on both the previous and current status values.
    The agent's goal is to collect rewards which depend on being at
    the rightmost position under certain status conditions.

    The observation space is a dictionary with three keys:
    - "position": Float in [0 - offset, length - 1 + offset] (1D)
    - "status_indicator": Float in [0 - offset, max_intensity + offset] (1D)
    - "prize_indicators": Binary indicators (with offset) of collected prizes.

    The reward function is:
    - +3.0 if the agent has all prize indicators == 1 while standing at the rightmost position.
    - 0.0 for LEFT moves (when not collecting).
    - -1.0 for RIGHT moves (when not collecting).

    Episodes never terminate by default (terminated=False, truncated=False) for each step.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = True,
        seed: Optional[int] = None,
    ) -> None:
        """Initializes the GoRight environment.

        Args:
            num_prize_indicators (int, optional): Number of prize indicators that can be collected.
            env_length (int, optional): Length of the 1D grid.
            status_intensities (List[int], optional): Possible intensity values for the status indicator.
            has_state_offset (bool, optional): Whether to add random offsets to observations for
                position/status/prize indicators.
            seed (Optional[int], optional): Seed for reproducibility.
        """
        super().__init__()

        self.num_prize_indicators = num_prize_indicators
        self.env_length = env_length
        self.status_intensities = status_intensities
        self.n_intensities = len(status_intensities)
        self.max_intensity = max(status_intensities)
        self.has_state_offset = has_state_offset
        self.intensity_to_idx = {val: i for i, val in enumerate(status_intensities)}

        self.max_offset_pos = 0.25
        self.max_offset_status = 1.25
        self.max_offset_prize = 0.25

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Dict(
            {
                "position": spaces.Box(
                    low=-self.max_offset_pos,
                    high=env_length - 1 + self.max_offset_pos,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "status_indicator": spaces.Box(
                    low=-self.max_offset_status,
                    high=self.max_intensity + self.max_offset_status,
                    shape=(1,),
                    dtype=np.float32,
                ),
                "prize_indicators": spaces.Box(
                    low=-self.max_offset_prize,
                    high=1.0 + self.max_offset_prize,
                    shape=(self.num_prize_indicators,),
                    dtype=np.float32,
                ),
            }
        )

        self.state: State

        if seed is not None:
            self.seed(seed)

    def seed(self, seed: Optional[int] = None) -> None:
        """Sets the random seed for the environment.

        Args:
            seed (Optional[int]): An integer seed for reproducibility.
        """
        super().reset(seed=seed)

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[str, Union[np.float32, np.ndarray]], Dict[str, Any]]:
        """Resets the environment to its initial state.

        Args:
            seed (Optional[int], optional): Seed for reproducibility.
            options (Optional[Dict[str, Any]], optional): Additional options (unused).

        Returns:
            Tuple[Dict[str, Union[np.float32, np.ndarray]], Dict[str, Any]]:
                A tuple (observation, info) where:
                  observation is a dict with "position", "status_indicator",
                  and "prize_indicators".
                  info is an additional dictionary, empty by default.
        """
        super().reset(seed=seed)

        if self.has_state_offset:
            position_offset = self.np_random.uniform(
                -self.max_offset_pos, self.max_offset_pos
            )
            status_offset = self.np_random.uniform(
                -self.max_offset_status, self.max_offset_status
            )
            prize_offset = self.np_random.uniform(
                -self.max_offset_prize, self.max_offset_prize, size=self.num_prize_indicators
            )
            self.offset = np.concatenate(
                [
                    [position_offset, status_offset, status_offset],
                    prize_offset,
                ]
            )
        else:
            self.offset = None

        self.state = State(
            position=0,
            previous_status_indicator=self.np_random.choice(self.status_intensities),
            current_status_indicator=self.np_random.choice(self.status_intensities),
            prize_indicators=np.zeros((self.num_prize_indicators)),
            offset=self.offset,
        )

        return self.state.get_observation(), {}

    def step(
        self, action: int
    ) -> Tuple[
        Dict[str, Union[np.float32, np.ndarray]], float, bool, bool, Dict[str, Any]
    ]:
        """Runs one timestep of the environment's dynamics.

        The agent takes an action (0=LEFT, 1=RIGHT), which updates the position and
        status indicators, and potentially collects or shifts the prize indicators.

        Args:
            action (int): The action taken by the agent: 0 (LEFT) or 1 (RIGHT).

        Returns:
            observation (Dict[str, Union[np.float32, np.ndarray]]):
                Updated observation after the step.
            reward (float): Immediate reward from this step.
            terminated (bool): Whether this episode has ended (False by default here).
            truncated (bool): Whether this episode was truncated (False by default).
            info (Dict[str, Any]): Additional info dictionary, empty by default.
        """
        if self.state is None:
            raise ValueError("State has not been initialized. Call reset() first.")

        (
            position,
            previous_status,
            current_status,
            *prize_indicators
        ) = self.state.get_state()

        prize_indicators = np.array(prize_indicators)

        next_pos = self._compute_next_position(action=action, position=position)
        next_status = STATUS_TRANSITION.get((previous_status, current_status), 0)
        next_prize_indicators = self._compute_next_prize_indicators(
            next_position=next_pos,
            next_status=next_status,
            prize_indicators=prize_indicators,
            position=position,
        )

        reward = self._compute_reward(
            prize_indicators=prize_indicators,
            action=action,
            next_pos=next_pos,
        )

        self.state.set_state(
            position=next_pos,
            previous_status_indicator=current_status,
            current_status_indicator=next_status,
            prize_indicators=next_prize_indicators,
        )

        return self.state.get_observation(), reward, False, False, {}

    def _compute_next_position(self, action: int, position: int) -> int:
        """Calculates the agent's next position based on the current position and the action.

        Args:
            action (int): 0=LEFT, 1=RIGHT.
            position (int): The agent's current position on the grid.

        Returns:
            int: Clamped next position within [0, self.env_length - 1].
        """
        direction = 1 if action == RIGHT else -1
        return int(np.clip(position + direction, 0, self.env_length - 1))

    def _compute_next_prize_indicators(
        self,
        next_position: float,
        next_status: int,
        prize_indicators: np.ndarray,
        position: float,
    ) -> np.ndarray:
        """Computes updated prize indicators after the agent moves.

        If the agent arrives at the rightmost position (self.env_length - 1) and the status intensity
        is at maximum, all indicators are set to 1. Otherwise, this function may
        shift or reset the prize indicators based on the agent's movement.

        Args:
            next_position (float): The agent's next position on the grid.
            next_status (int): The environment's next status intensity.
            prize_indicators (np.ndarray): Current prize indicators.
            position (float): The agent's previous position on the grid.

        Returns:
            np.ndarray: The updated prize indicators.
        """
        if int(next_position) == self.env_length - 1:
            if int(position) == self.env_length - 2:
                if next_status == self.max_intensity:
                    return np.ones_like(prize_indicators, dtype=int)
            elif all(prize_indicators == 1):
                return prize_indicators
            else:
                return self._shift_prize_indicators(prize_indicators)
        return np.zeros_like(prize_indicators, dtype=int)

    def _shift_prize_indicators(self, prize_indicators: np.ndarray) -> np.ndarray:
        """Shifts the '1' in prize_indicators to simulate incremental collection.

        The logic here places a '1' in the leftmost slot if no indicator is set.
        Otherwise, it moves the '1' from its current position to the next slot,
        simulating step-by-step accumulation.

        Args:
            prize_indicators (np.ndarray): Current prize indicators array.

        Returns:
            np.ndarray: Updated prize indicators.
        """
        if all(prize_indicators < 0.5):
            prize_indicators[0] = 1
            prize_indicators[1:] = 0
        else:
            one_index = np.argmax(prize_indicators)
            prize_indicators[one_index] = 0
            if one_index < self.num_prize_indicators - 1:
                prize_indicators[one_index + 1] = 1
        return prize_indicators

    def _compute_reward(
        self, prize_indicators: np.ndarray, action: int, next_pos: int
    ) -> float:
        """Computes the reward based on the new prize indicators, action, and agent position.

        Logic:
         - If the agent has all indicators == 1 and will be at the rightmost position, reward = +3.0
         - If not collecting, reward = 0.0 when moving LEFT, and -1.0 when moving RIGHT.

        Args:
            prize_indicators (np.ndarray): The updated prize indicators after this step.
            action (int): The action just taken by the agent (0=LEFT, 1=RIGHT).
            next_pos (int): The agent's updated position.

        Returns:
            float: The reward for this step.
        """
        if all(prize_indicators == 1) and (next_pos == self.env_length - 1):
            return 3.0
        if action == LEFT:
            return 0.0
        return -1.0
