"""A simple 1D "Go Right" environment illustrating a second-order Markov status transition.

The environment is a 1D grid of configurable length. The agent can move left or right.
The status indicator transitions according to a second-order Markov chain, and the agent observes
the prize indicators by reaching the rightmost grid position. When the agent reaches the last state
with the status indicator at maximum intensity, the prize indicators go on and the agent receives the
reward.
"""

from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np
from goright.utils import State
from gymnasium import spaces
from gymnasium.error import DependencyNotInstalled


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


class GoRight(gym.Env):
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

    metadata = {
        "render_modes": ["human", "ansi", "rgb_array"],
        "render_fps": 4,
    }

    def __init__(
        self,
        num_prize_indicators: int = 2,
        env_length: int = 11,
        status_intensities: List[int] = [0, 5, 10],
        has_state_offset: bool = True,
        show_status_ind: bool = True,
        show_prev_status_ind: bool = False,
        seed: Optional[int] = None,
        render_mode: Optional[str] = None,
    ) -> None:
        """Initializes the GoRight environment.

        Args:
            num_prize_indicators (int, optional): Number of prize indicators that can be collected.
            env_length (int, optional): Length of the 1D grid.
            status_intensities (List[int], optional): Possible intensity values for the status indicator.
            has_state_offset (bool, optional): Whether to add random offsets to observations for
                position/status/prize indicators.
            show_status_ind (bool, optional): Flag to include the current status indicator in the observation. Defaults to True.
            show_prev_status_ind (bool, optional): Flag to include the previous status indicator in the observation. Defaults to False.
            seed (Optional[int], optional): Seed for reproducibility.
            render_mode (Optional[str], optional): Render mode.
        """
        super().__init__()

        self.num_prize_indicators = num_prize_indicators
        self.env_length = env_length
        self.status_intensities = status_intensities
        self.n_intensities = len(status_intensities)
        self.max_intensity = max(status_intensities)
        self.has_state_offset = has_state_offset
        self.intensity_to_idx = {val: i for i, val in enumerate(status_intensities)}
        self.show_status_ind = show_status_ind
        self.show_prev_status_ind = show_prev_status_ind

        self.max_offset_pos = 0.25
        self.max_offset_status = 1.25
        self.max_offset_prize = 0.25

        self.action_space = spaces.Discrete(2)

        position_box = spaces.Box(
            low=-self.max_offset_pos,
            high=env_length - 1 + self.max_offset_pos,
            shape=(1,),
            dtype=np.float32,
        )
        status_box = spaces.Box(
            low=-self.max_offset_status,
            high=self.max_intensity + self.max_offset_status,
            shape=(1,),
            dtype=np.float32,
        )
        prize_box = spaces.Box(
            low=-self.max_offset_prize,
            high=1.0 + self.max_offset_prize,
            shape=(self.num_prize_indicators,),
            dtype=np.float32,
        )

        if show_status_ind:
            if show_prev_status_ind:
                self.observation_space = spaces.Dict(
                    {
                        "position": position_box,
                        "prev_status_indicator": status_box,
                        "status_indicator": status_box,
                        "prize_indicators": prize_box,
                    }
                )
            else:
                self.observation_space = spaces.Dict(
                    {
                        "position": position_box,
                        "status_indicator": status_box,
                        "prize_indicators": prize_box,
                    }
                )

        else:
            if show_prev_status_ind:
                self.observation_space = spaces.Dict(
                    {
                        "position": position_box,
                        "prev_status_indicator": status_box,
                        "prize_indicators": prize_box,
                    }
                )
            else:
                self.observation_space = spaces.Dict(
                    {
                        "position": position_box,
                        "prize_indicators": prize_box,
                    }
                )

        self.state: State
        self.render_mode = render_mode

        # For Pygame:
        self.window = None
        self.clock = None
        self.window_size = (800, 300)
        self.last_reward = 0.0

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
    ) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
        """Resets the environment to its initial state.

        Args:
            seed (Optional[int], optional): Seed for reproducibility.
            options (Optional[Dict[str, Any]], optional): Additional options (unused).

        Returns:
            Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
                A tuple (observation, info) where:
                  observation is a dict with "position", "status_indicator",
                  and "prize_indicators".
                  info is an additional dictionary, empty by default.
        """
        super().reset(seed=seed)
        self.last_reward = 0.0

        if self.has_state_offset:
            position_offset = self.np_random.uniform(
                -self.max_offset_pos, self.max_offset_pos
            )
            status_offset = self.np_random.uniform(
                -self.max_offset_status, self.max_offset_status
            )
            prize_offset = self.np_random.uniform(
                -self.max_offset_prize,
                self.max_offset_prize,
                size=self.num_prize_indicators,
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
            prize_indicators=np.zeros(self.num_prize_indicators),
            offset=self.offset,
            show_status_ind=self.show_status_ind,
            show_prev_status_ind=self.show_prev_status_ind,
        )

        if self.render_mode == "human":
            self.render()

        return self.state.get_observation(), {}

    def step(
        self, action: int
    ) -> Tuple[Dict[str, np.ndarray], float, bool, bool, Dict[str, Any]]:
        """Runs one timestep of the environment's dynamics.

        The agent takes an action (0=LEFT, 1=RIGHT), which updates the position and
        status indicators, and potentially collects or shifts the prize indicators.

        Args:
            action (int): The action taken by the agent: 0 (LEFT) or 1 (RIGHT).

        Returns:
            observation (Dict[str, np.ndarray]):
                Updated observation after the step.
            reward (float): Immediate reward from this step.
            terminated (bool): Whether this episode has ended (False by default here).
            truncated (bool): Whether this episode was truncated (False by default).
            info (Dict[str, Any]): Additional info dictionary, empty by default.
        """
        if self.state is None:
            raise ValueError("State has not been initialized. Call reset() first.")

        (position, previous_status, current_status, *prize_indicators) = (
            self.state.get_state()
        )

        prize_indicators = np.array(prize_indicators)

        next_pos = self._compute_next_position(action=action, position=position)
        next_status = self._compute_next_status(
            previous_status=previous_status, current_status=current_status
        )
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
        self.last_reward = reward

        self.state.set_state(
            position=next_pos,
            previous_status_indicator=current_status,
            current_status_indicator=next_status,
            prize_indicators=next_prize_indicators,
        )

        if self.render_mode == "human":
            self.render()

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

    def _compute_next_status(self, previous_status: int, current_status: int) -> int:
        """Computes the next status following the second-order markov dynamics.

        Args:
            previous_status (int): Previous status intensity.
            current_status (int): Current status intensity.

        Returns:
            int: Next status intensity.
        """
        return STATUS_TRANSITION.get((previous_status, current_status), 0)

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

    def render(self) -> None | str | np.ndarray:
        """Renders the environment depending on self.render_mode.

        - None: do nothing (warn user if they call render)
        - "ansi": ASCII text
        - "human": Pygame window
        - "rgb_array": returns image array (also uses Pygame, but returns np array)
        """
        if self.render_mode is None:
            if self.spec is not None:
                gym.logger.warn(
                    "You are calling render() without specifying any render_mode. "
                    f'Use e.g. gym.make("{self.spec.id}", render_mode="human").'
                )
            return

        if self.render_mode == "ansi":
            return self._render_ascii()
        elif self.render_mode in {"human", "rgb_array"}:
            return self._render_pygame(self.render_mode)
        else:
            raise NotImplementedError(f"Render mode {self.render_mode} not supported.")

    def _render_ascii(self) -> None | str:
        """ASCII-based rendering that prints to console and returns a string."""
        if self.state is None:
            return "Environment not initialized."

        position = self.state.position
        prev_stat = self.state.previous_status_indicator
        curr_stat = self.state.current_status_indicator
        prizes = self.state.prize_indicators

        cells = ["-"] * self.env_length
        cells[position] = "A"
        grid_str = "".join(cells)

        out = (
            "\n"
            + "=" * 60
            + f"\nGrid:           {grid_str}"
            + f"\nPosition:       {position}"
            + f"\nPrev Status:    {prev_stat}"
            + f"\nCurr Status:    {curr_stat}"
            + f"\nPrizes:         {prizes.tolist()}"
            + f"\nLast Reward:    {self.last_reward}"
            + "\n"
            + "=" * 60
            + "\n"
        )
        print(out)
        return out

    def _render_pygame(self, mode: str) -> Optional[np.ndarray]:
        """Render via pygame in a more 'video game' style.

        Args:
            mode (str): Either 'human' (pygame window) or 'rgb_array' (returns image array).

        Returns:
            Optional[np.ndarray]: The RGB array if mode='rgb_array', else None.
        """
        # Lazy import pygame to keep it optional.
        try:
            import pygame
        except ImportError as e:
            raise DependencyNotInstalled(
                "pygame is not installed. Install via `pip install git+https://github.com/cruz-lucas/goright.git#egg=goright[extra]` if using pip, see README."
            ) from e

        if self.window is None:
            pygame.init()
            if mode == "human":
                pygame.display.init()
                pygame.display.set_caption("GoRight-v0 (Video Game UI)")
                self.window = pygame.display.set_mode(self.window_size)
            else:
                self.window = pygame.Surface(self.window_size)
            self.clock = pygame.time.Clock()

        assert self.window is not None
        self.window.fill((180, 180, 180))

        if self.state is not None:
            position = self.state.position
            prev_stat = self.state.previous_status_indicator
            curr_stat = self.state.current_status_indicator
            prizes = self.state.prize_indicators
        else:
            position = 0
            prev_stat = 0
            curr_stat = 0
            prizes = np.zeros(self.num_prize_indicators)

        self._draw_grid(position, pygame)

        self._draw_ui_bar(position, prev_stat, curr_stat, prizes, pygame)

        if mode == "human":
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
            return None
        else:  # mode == 'rgb_array'
            return self._surface_to_array(self.window)

    def _draw_grid(self, position: int, pygame_module) -> None:
        """Draws a row of rectangles to represent the 1D grid.

        Args:
            position (int): Agent's position.
            pygame_module (_type_): Pygame module.
        """
        cell_width = self.window_size[0] / self.env_length
        cell_height = self.window_size[1] * 0.4  # 40% of window height
        y_offset = self.window_size[1] - cell_height - 10

        for i in range(self.env_length):
            x = i * cell_width + 2
            rect = pygame_module.Rect(x, y_offset, cell_width - 4, cell_height - 4)
            color = (0, 128, 255) if i == position else (200, 200, 200)
            pygame_module.draw.rect(self.window, color, rect)

    def _draw_ui_bar(
        self,
        position: int,
        prev_status: int,
        curr_status: int,
        prizes: np.ndarray,
        pygame_module,
    ) -> None:
        """Draws a top bar with text about position, statuses, prizes, and reward.

        Args:
            position (int): Agent's position.
            prev_status (int): Previous status intensity.
            curr_status (int): Current status intensity.
            prizes (np.ndarray): Prize indicators array.
            pygame_module (_type_): Pygame module.
        """
        ui_bar_height = 60
        ui_rect = pygame_module.Rect(0, 0, self.window_size[0], ui_bar_height)
        pygame_module.draw.rect(self.window, (50, 50, 50), ui_rect)

        font = pygame_module.font.SysFont("Arial", 22)

        prize_str = "".join(["1" if p >= 0.5 else "0" for p in prizes])

        info_line_1 = f"Position: {position}   Prev.Status: {prev_status}   Curr.Status: {curr_status}"
        info_line_2 = f"Prizes: {prize_str}    Reward: {self.last_reward:.2f}"

        text_color = (230, 230, 230)
        line1_surf = font.render(info_line_1, True, text_color)
        line2_surf = font.render(info_line_2, True, text_color)

        self.window.blit(line1_surf, (10, 10))
        self.window.blit(line2_surf, (10, 30))

    def render_with_bbox(
        self,
        agent_position: int,
        low_position: int,
        high_position: int,
        predicted_position: int,
        mode: str = "human",
    ) -> None:
        """Render the grid with overlays for planning range and predicted state.

        Args:
            agent_position (int): The actual agent position (blue).
            low_position (int): Lower bound of planned range.
            high_position (int): Upper bound of planned range.
            predicted_position (int): Predicted position.
            mode (str): Render mode: "human".
        """
        import pygame

        if self.window is None:
            self._render_pygame(mode)
        assert self.window is not None

        # Clear base frame
        self.window.fill((180, 180, 180))
        self._draw_ui_bar(
            self.state.position,
            self.state.previous_status_indicator,
            self.state.current_status_indicator,
            self.state.prize_indicators,
            pygame,
        )

        # Draw grid
        cell_width = self.window_size[0] / self.env_length
        cell_height = self.window_size[1] * 0.4
        y_offset = self.window_size[1] - cell_height - 10

        for i in range(self.env_length):
            x = i * cell_width + 2
            rect = pygame.Rect(x, y_offset, cell_width - 4, cell_height - 4)

            if i == agent_position:
                pygame.draw.rect(self.window, (0, 128, 255), rect)
            else:
                pygame.draw.rect(self.window, (200, 200, 200), rect)

        for i in range(self.env_length):
            x = i * cell_width + 2

            if low_position <= i <= high_position:
                rect = pygame.Rect(x, y_offset - 25, cell_width - 4, 15)
                color = (255, 255, 0)  # yellow box range
                pygame.draw.rect(self.window, color, rect)

        # Predicted (red), slightly above
        x = predicted_position * cell_width + 2
        rect = pygame.Rect(x, y_offset - 50, cell_width - 4, 15)
        pygame.draw.rect(self.window, (255, 0, 0), rect)

        pygame.display.update()
        self.clock.tick(self.metadata["render_fps"])

    @staticmethod
    def _surface_to_array(surface) -> np.ndarray:
        """Convert a pygame surface to a numpy array for 'rgb_array' rendering."""
        import pygame.surfarray

        arr = pygame.surfarray.array3d(surface)
        # Pygame surfaces are (width, height), swap to (height, width, 3)
        return np.transpose(arr, (1, 0, 2))

    def close(self):
        """Closes pygame window."""
        if self.window is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
        super().close()
