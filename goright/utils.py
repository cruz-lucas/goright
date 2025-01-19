"""Utility module defining the `State` dataclass for the GoRight environment."""

from dataclasses import dataclass
from typing import Dict, Optional, Union

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

    mask: Optional[np.ndarray] = None
    previous_status_idx: int = 1
    intensities: np.ndarray = np.array([0, 5, 10])

    def __post_init__(self) -> None:
        """Create a default mask that hides the 'previous_status_indicator' from observations.

        The order in get_state() is:
          - 0 -> position
          - 1 -> previous_status_indicator
          - 2 -> current_status_indicator
          - 3.. -> prize_indicators

        By default, we set mask[1] = False to exclude the previous_status_indicator from the observation.
        """
        if self.mask is None:
            full_state_size = self.get_state().shape[0]
            self.mask = np.ones(full_state_size, dtype=bool)
            self.mask[self.previous_status_idx] = False

    def get_observation(self) -> Dict[str, Union[np.float32, np.ndarray]]:
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

        masked_state = raw_state[self.mask]

        return {
            "position": np.array(masked_state[0], dtype=np.float32).reshape(1),
            "status_indicator": np.array(masked_state[1], dtype=np.float32).reshape(1),
            "prize_indicators": np.array(masked_state[2:], dtype=np.float32),
        }

    def get_state(self) -> np.ndarray:
        """Retrieves the full internal state as a NumPy array in the order:
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

