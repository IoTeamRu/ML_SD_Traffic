import os
import sys
from itertools import count
import torch
import argparse
import numpy as np
from model import Model
from data.utils import load_config


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment  # type: ignore


def run_model_on_trajectory(config):
    env_cfg = config.deploy.environment

    env = SumoEnvironment(
        net_file=os.path.join(config.data.source_path, config.train.scenario, env_cfg.net_file),
        route_file=os.path.join(config.data.source_path, config.train.scenario, env_cfg.route_file),
        out_csv_name=env_cfg.agent_result,
        single_agent=env_cfg.single_agent,
        use_gui=env_cfg.gui,
        num_seconds=int(env_cfg.num_steps),
    )

    evaluate_reward = 0
    rewards = []
    model = Model(env.observation_space.shape[0], env.action_space.n)
    try:
        model.load_state_dict(torch.load(os.path.join(config.deploy.model_path, config.deploy.model_file)))
    except:
        raise RuntimeError(f'Parameters you are trying to load are not compatible with the current model') 
    model.eval()
    for _ in range(config.deploy.evaluate_times):
        state = env.reset()
        done = False
        episode_reward = 0
        while not done:
            with torch.no_grad():
                out = model(torch.from_numpy(state).to(device, torch.float32))
            action = np.argmax(out[0])
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            state = next_state
        evaluate_reward += episode_reward
        rewards.append(evaluate_reward)
    evaluate_reward = np.mean(rewards)
    if config.deploy.verbose:
        print(f"Test was successfully passed,  mean_reward:{evaluate_reward}") 

    return evaluate_reward


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config_path=args.config

    config = load_config(config_path)

    device = torch.device(f'cuda:{int(config.train.device_id)}' if torch.cuda.is_available() else "cpu")

    result = run_model_on_trajectory(config, device)
    print(f'Episode reward: {result}')