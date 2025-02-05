"""Script to interactively play the GoRight-v0 environment.

This script demonstrates how to:
  - Create and reset the GoRight environment.
  - Optionally render it in 'human' mode (pygame window) or 'ansi' (ASCII text).
  - Interactively control the agent via console inputs.

Example:
    python play_goright.py --render_mode ansi
    python play_goright.py --render_mode human
"""

import argparse

import goright  # noqa: F401
import gymnasium as gym


LEFT = 0
RIGHT = 1


def main(render_mode: str = "ansi") -> None:
    """Plays an interactive session of the GoRight-v0 environment.

    Args:
        render_mode (str): The rendering mode to use. 'ansi' for ASCII console output,
            'human' for a pygame window (requires pygame installed),
            or 'rgb_array' to capture rendered frames as numpy arrays.

    Returns:
        None
    """
    print(f"Initializing GoRight-v0 with render_mode='{render_mode}'...")
    env = gym.make("GoRight-v0", render_mode=render_mode)

    observation, info = env.reset()
    total_reward = 0.0
    step_count = 0

    print("Welcome to GoRight-v0!")
    print("Controls: [A] Move Left, [D] Move Right, [Q] Quit.\n")

    while True:
        _ = env.render()
        print(f"Total Reward so far: {total_reward}")

        user_input = input("Enter action (A/D/Q): ").strip().lower()
        if user_input == "q":
            print("Exiting the game.")
            break
        elif user_input == "a":
            action = LEFT
        elif user_input == "d":
            action = RIGHT
        else:
            print("Invalid input. Please press 'A', 'D', or 'Q'.")
            continue

        obs, reward, done, truncated, info = env.step(action)
        total_reward += float(reward)
        step_count += 1

        print(f"Step: {step_count}, Action: {action}, Reward: {reward}")

        # By default, GoRight-v0 doesn't set done=True, so the loop continues until user quits.

    print("Final total reward:", total_reward)
    env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Play GoRight-v0 interactively.")
    parser.add_argument(
        "--render_mode",
        type=str,
        default="human",
        choices=["ansi", "human", "rgb_array", None],
        help="Rendering mode: 'ansi' for ASCII, 'human' for a pygame window, or 'rgb_array' to get image arrays.",
    )
    args = parser.parse_args()

    main(render_mode=args.render_mode)
