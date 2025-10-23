"""Go-Right environment implemented with the Gymnasium functional JAX API."""

from __future__ import annotations

from typing import NamedTuple, TypeAlias

import jax
import jax.numpy as jnp
import jax.random as jrng
import numpy as np
import math
from flax import struct
from gymnasium import spaces
from gymnasium.envs.functional_jax_env import FunctionalJaxEnv
from gymnasium.experimental.functional import FuncEnv
from gymnasium.utils import EzPickle

PRNGKey: TypeAlias = jax.Array

DEFAULT_STATUS_MAPPING = jnp.asarray(
    [
        [1, 0, 1],
        [2, 2, 2],
        [0, 1, 0],
    ],
    dtype=jnp.int32,
)

class EnvState(NamedTuple):
    """Stateless Go-Right environment state."""

    position: jax.Array
    previous_status: jax.Array
    status: jax.Array
    prize_inds: jax.Array
    rng: PRNGKey



@struct.dataclass
class EnvParams:
    """Static parameters for the Go-Right environment."""

    length: int = 21
    num_indicators: int = 2
    num_actions: int = 2
    first_checkpoint: int = 10
    first_reward: float = 3.0
    second_checkpoint: int = 20
    second_reward: float = 6.0
    is_partially_obs: bool = False
    mapping: str = 'default'


class GoRightFunctional(
    FuncEnv[
        EnvState,
        jax.Array,
        jax.Array,
        jax.Array,
        jax.Array,
        None,
        EnvParams,
    ]
):
    """Go-Right environment expressed through the functional Gymnasium API."""

    metadata = {
        "render_modes": ["rgb_array"],
        "render_fps": 30,
    }

    def __init__(self, params: EnvParams | None = None):
        """Initialise the functional environment."""
        self.default_params = self.get_default_params()
        self.params = self.default_params if params is None else params

        self._status_mapping = DEFAULT_STATUS_MAPPING


        self._fully_obs_shape = (
            self.params.length,
            3,
            3,
            2,
            2,
        )

        self._partially_obs_shape = (
                self.params.length,
                3,
                2,
                2,
            )

        self.obs_shape = self._partially_obs_shape if self.params.is_partially_obs else self._fully_obs_shape
        self.action_space = spaces.Discrete(self.params.num_actions)
        self.observation_space = spaces.Discrete(int(math.prod(self.obs_shape)))

    

    def initial(self, rng: PRNGKey, params: EnvParams | None = None) -> EnvState:
        """Sample an initial state."""
        next_rng, key_status, key_prev_status = jrng.split(rng, 3)

        status_levels = jnp.arange(3, dtype=jnp.int32)
        status = jrng.choice(key_status, status_levels)
        previous_status = jrng.choice(key_prev_status, status_levels)
        prize_inds = jnp.zeros((self.params.num_indicators,), dtype=jnp.int32)

        return EnvState(
            position=jnp.asarray(0, dtype=jnp.int32),
            previous_status=previous_status,
            status=status,
            prize_inds=prize_inds,
            rng=next_rng,
        )

    def observation(
        self, state: EnvState, params: EnvParams | None = None
    ) -> jax.Array:
        """Return the discrete observation index."""
        fully_obs = jnp.ravel_multi_index(
            jnp.array(
                [
                    state.position,
                    state.previous_status,
                    state.status,
                    state.prize_inds[0],
                    state.prize_inds[1],
                ]
            ), dims=self._fully_obs_shape, mode='clip'
        )

        partially_obs = jnp.ravel_multi_index(
            jnp.array(
                [
                    state.position,
                    state.status,
                    state.prize_inds[0],
                    state.prize_inds[1],
                ]
            ), dims=self._partially_obs_shape, mode='clip'
        )

        return (1 - self.params.is_partially_obs) * fully_obs + self.params.is_partially_obs * partially_obs

    def state_info(
        self, state: EnvState, params: EnvParams | None = None
    ) -> dict[str, jax.Array]:
        """Diagnostic information for the given state."""
        return {
            "position": jnp.asarray(state.position, dtype=jnp.int32),
            "previous_status": jnp.asarray(state.previous_status, dtype=jnp.int32),
            "status": jnp.asarray(state.status, dtype=jnp.int32),
            "prize_inds": jnp.asarray(state.prize_inds, dtype=jnp.int32),
        }

    def _next_status(self, state: EnvState) -> jax.Array:
        """Compute the next status deterministically."""
        prev_status = jnp.asarray(state.previous_status, dtype=jnp.int32)
        curr_status = jnp.asarray(state.status, dtype=jnp.int32)
        row = jnp.take(self._status_mapping, prev_status, axis=0, mode="clip")
        return jnp.take(row, curr_status, axis=0, mode="clip")

    @staticmethod
    def _shift_prizes(prize_inds: jax.Array, params: EnvParams) -> jax.Array:
        """Shift the prize indicator vector according to the environment rules."""
        pad = jnp.where(jnp.any(prize_inds != 0), 0, 1)
        padded = jnp.concatenate(
            [jnp.asarray(prize_inds, dtype=jnp.int32), jnp.asarray([pad], dtype=jnp.int32)],
            axis=0,
        )
        rolled = jnp.roll(padded, 1, axis=0)
        return rolled[: params.num_indicators]

    def _next_prize_inds(
        self,
        state: EnvState,
        action: jax.Array,
        next_status: jax.Array,
        params: EnvParams,
    ) -> jax.Array:
        """Compute the next prize indicator values."""
        prize = jnp.asarray(state.prize_inds, dtype=jnp.int32)
        is_jackpot = jnp.equal(next_status, 2)
        is_right = jnp.equal(jnp.asarray(action, dtype=jnp.int32), 1)
        triggered_now = jnp.all(prize != 0)

        position = jnp.asarray(state.position, dtype=jnp.int32)
        first_cp = params.first_checkpoint
        second_cp = params.second_checkpoint
        last_pos = params.length - 1

        is_before_first = jnp.equal(position, first_cp - 1)
        is_before_second = jnp.equal(position, second_cp - 1)
        is_before_pos = jnp.logical_or(is_before_first, is_before_second)

        is_first = jnp.equal(position, first_cp)
        is_second = jnp.equal(position, second_cp)
        is_prize_pos = jnp.logical_or(is_first, is_second)

        is_entering = jnp.logical_and(is_right, jnp.logical_and(is_before_pos, is_jackpot))
        is_continuing = jnp.logical_and(
            triggered_now,
            jnp.logical_and(is_right, is_prize_pos),
        )
        is_triggering = jnp.logical_or(is_entering, is_continuing)

        is_at_end = jnp.equal(position, last_pos)
        needs_shifting = jnp.logical_and(
            jnp.logical_and(is_at_end, is_right), jnp.logical_not(is_triggering)
        )

        shifted = self._shift_prizes(prize, params)
        ones_like = jnp.ones_like(prize, dtype=jnp.int32)
        zeros_like = jnp.zeros_like(prize, dtype=jnp.int32)

        return jnp.where(
            is_triggering,
            ones_like,
            jnp.where(needs_shifting, shifted, zeros_like),
        )

    def transition(
        self,
        state: EnvState,
        action: jax.Array,
        params: EnvParams | None = None,
    ) -> EnvState:
        """Deterministically transition to the next state."""
        next_rng, step_key = jrng.split(state.rng, 2)
        action = jnp.asarray(action, dtype=jnp.int32)
        position = jnp.asarray(state.position, dtype=jnp.int32)
        step = jnp.where(jnp.equal(action, 1), 1, -1)
        next_position = jnp.clip(position + step, 0, self.params.length - 1)

        next_status = self._next_status(state)
        next_prize = self._next_prize_inds(state, action, next_status, self.params)

        continue_triggering = jnp.logical_and(
            jnp.all(next_prize != 0),
            jnp.all(jnp.asarray(state.prize_inds, dtype=jnp.int32) != 0),
        )
        freeze_pos = jnp.logical_and(
            jnp.equal(position, self.params.first_checkpoint),
            continue_triggering,
        )
        next_position = jnp.where(freeze_pos, position, next_position)

        return EnvState(
            position=jnp.asarray(next_position, dtype=jnp.int32),
            previous_status=jnp.asarray(state.status, dtype=jnp.int32),
            status=jnp.asarray(next_status, dtype=jnp.int32),
            prize_inds=next_prize,
            rng=next_rng
        )

    def reward(
        self,
        state: EnvState,
        action: jax.Array,
        next_state: EnvState,
        params: EnvParams | None = None,
    ) -> jax.Array:
        """Compute the reward for a transition."""
        action = jnp.asarray(action, dtype=jnp.int32)
        position = jnp.asarray(state.position, dtype=jnp.int32)
        prize = jnp.asarray(state.prize_inds, dtype=jnp.int32)
        next_prize = jnp.asarray(next_state.prize_inds, dtype=jnp.int32)

        is_right = jnp.equal(action, 1)
        penalty = jnp.where(is_right, -1.0, 0.0)

        is_first_cp = jnp.equal(position, self.params.first_checkpoint)
        is_second_cp = jnp.equal(position, self.params.second_checkpoint)
        is_cp = jnp.logical_or(is_first_cp, is_second_cp)

        triggered = jnp.logical_and(
            is_cp,
            jnp.logical_and(jnp.all(prize != 0), jnp.all(next_prize != 0)),
        )

        first_reward = jnp.where(is_first_cp, self.params.first_reward, 0.0)
        second_reward = jnp.where(is_second_cp, self.params.second_reward, 0.0)

        reward = jnp.where(triggered, first_reward + second_reward, penalty)
        return jnp.asarray(reward, dtype=jnp.float32)

    def terminal(self, state: EnvState, params: EnvParams | None = None) -> jax.Array:
        """GoRight has no terminal states."""
        return jnp.asarray(False, dtype=jnp.bool_)
    
    def get_default_params(self, **kwargs) -> EnvParams:
        """Get the default params."""
        return EnvParams(**kwargs)


class GoRightJaxEnv(EzPickle):
    """Jax-friendly API around the functional RiverSwim environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30, "jax": True}

    def __init__(
        self, params: EnvParams | None = None, render_mode: str | None = None, **kwargs
    ):
        """Wraps functional environment."""
        EzPickle.__init__(self, render_mode=render_mode, **kwargs)
        env = GoRightFunctional(params=params)
        env.transform(jax.jit)
        self.env = env

    def reset(self, rng: PRNGKey):
        """Resets the environment using the seed."""
        initial_state = self.env.initial(rng=rng)
        return initial_state, self.env.observation(initial_state)

    def step(self, state: EnvState, action: jax.Array):
        """Steps through the environment using the action."""
        next_state = self.env.transition(state, action)
        observation = self.env.observation(next_state)
        reward = self.env.reward(state, action, next_state)
        terminated = self.env.terminal(next_state)
        info = self.env.transition_info(state, action, next_state)

        return (
            next_state,
            observation,
            jnp.array(reward, dtype=float),
            jnp.array(terminated, dtype=bool),
            False,
            info,
        )


if __name__ == "__main__":
    env = GoRightJaxEnv()

    rng = jrng.key(0)
    state, obs = env.reset(rng=rng)
    print(state)

    terminal = False
    while not terminal:
        action = jnp.array(input("Please input an action\n"), dtype=int)

        state, obs, reward, terminal, truncated, info = env.step(state, action)
        print(state, obs, reward, terminal, truncated, info)

    exit()
