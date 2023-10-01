import os
import sys
from itertools import count
import torch
from stable_baselines3.dqn.dqn import DQN


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment  # type: ignore


def run_model_on_trajectory():
    env = SumoEnvironment(
        net_file="data/single-intersection.net.xml",
        route_file="data/single-intersection-vhvh.rou.xml",
        out_csv_name="outputs/2way-single-intersection/dqn",
        single_agent=True,
        use_gui=False,
        num_seconds=3000,
    )

    model = DQN.load('saved_models/DQN_model', env=env, print_system_info=True)
    
    obs, info = env.reset()

    episode_reward = 0
    for _ in count():
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        if terminated or truncated or info.get("is_success", False):
            break

    return episode_reward


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    result = run_model_on_trajectory()
    print(f'Episode reward: {result}')
