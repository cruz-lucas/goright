"""Registers the internal gym envs then loads the env plugins for module using the entry point."""

from gymnasium.envs.registration import register


register(
    id="GoRight-v0",
    entry_point="goright.goright:GoRight",
)

register(
    id="GoRight10-v0",
    entry_point="goright.goright:GoRight",
    kwargs={"num_prize_indicators": 10}
)