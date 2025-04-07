"""Utility module defining the `State` dataclass for the GoRight environment."""

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass
class State:
    """A container for the internal state of the GoRight environment.

    This class stores:
      - position (int): The agent's current position in the 1D grid.
      - previous_status_indicator (int): The previous status intensity.
      - current_status_indicator (int): The current status intensity.
      - prize_indicators (np.ndarray): Array of prize indicators (0 or 1).
      - offset (Optional[np.ndarray]): If not None, random offsets applied to the observation.
      - mask (np.ndarray): Boolean mask to hide any state components from the observation.
        (Defaults to hiding the previous_status_indicator.)
      - previous_status_idx (int): Index in the state array corresponding to the previous status.
      - intensities (np.ndarray): Optional reference array of valid intensities (e.g., [0, 5, 10]).
    """

    position: int
    previous_status_indicator: int
    current_status_indicator: int
    prize_indicators: np.ndarray
    offset: Optional[np.ndarray] = None
    show_status_ind: bool = True
    show_prev_status_ind: bool = False

    def get_observation(self) -> Dict[str, np.ndarray]:
        """Returns the masked and optionally offset state as a dictionary.

        The final observation dictionary has:
          - "position": np.array(shape=(1,))
          - "status_indicator": np.array(shape=(1,))
          - "prize_indicators": np.array(shape=(num_prize_indicators,))

        Returns:
            Dict[str, Union[np.float32, np.ndarray]]: The masked, offset state.
        """
        raw_state = self.get_state()
        if self.offset is not None:
            raw_state = raw_state + self.offset

        observation = {
            "position": np.array([raw_state[0]], dtype=np.float32),
        }

        if self.show_prev_status_ind:
            observation["prev_status_indicator"] = np.array(
                [raw_state[1]], dtype=np.float32
            )

        if self.show_status_ind:
            observation["status_indicator"] = np.array([raw_state[2]], dtype=np.float32)
        observation["prize_indicators"] = np.array(raw_state[3:], dtype=np.float32)

        return observation

    def get_state(self) -> np.ndarray:
        """Retrieves the full internal state as a NumPy array in the order.

        [position, previous_status_indicator, current_status_indicator, ...prize_indicators].

        Returns:
            np.ndarray: The concatenated state array.
        """
        return np.concatenate(
            [
                [
                    self.position,
                    self.previous_status_indicator,
                    self.current_status_indicator,
                ],
                self.prize_indicators,
            ]
        )

    def set_state(
        self,
        position: int,
        current_status_indicator: int,
        prize_indicators: np.ndarray,
        previous_status_indicator: int = 0,
    ) -> None:
        """Updates the internal state.

        Args:
            position (int): The agent's position in the 1D grid.
            current_status_indicator (int): The updated status intensity.
            prize_indicators (np.ndarray): The updated prize indicators.
            previous_status_indicator (int, optional): The previous status intensity
                (defaults to 0 if not explicitly set).
        """
        self.position = position
        self.previous_status_indicator = previous_status_indicator
        self.current_status_indicator = current_status_indicator
        self.prize_indicators = prize_indicators
