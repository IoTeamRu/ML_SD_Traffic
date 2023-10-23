import os
import sys
import argparse
import torch
import time
import numpy as np
from collections import deque

from data.utils import load_config
from learner import Learner
from model import Model


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment  # type: ignore


def train(config, device):
    env_cfg = config.train.environment
    alg_cfg = config.train.hyper

    start = time.time()
    
    env = SumoEnvironment(
        net_file=os.path.join(config.data.source_path, config.train.scenario, env_cfg.net_file),
        route_file=os.path.join(config.data.source_path, config.train.scenario, env_cfg.route_file),
        out_csv_name=env_cfg.agent_result,
        single_agent=env_cfg.single_agent,
        use_gui=env_cfg.gui,
        num_seconds=int(env_cfg.num_steps),
    )
    learner = Learner(env, device)

    episode_reward = 0
    mean_rew = deque(maxlen=20)
    observation = env.reset()[0]

    for step in range(alg_cfg.steps):
        action = learner.act(observation, step)
        next_obs, reward, done, trunc, _ = env.step(action)
        learner.memory.add(observation, action, reward, next_obs, done)
        episode_reward += reward
        observation = next_obs

        learner.learn(step)

        if done or trunc:
            observation = env.reset()[0]
            mean_rew.append(episode_reward)
            episode_reward = 0

        if step == alg_cfg.steps -1:
            test_env = SumoEnvironment(
                net_file=os.path.join(config.data.source_path, config.train.scenario, env_cfg.net_file),
                route_file=os.path.join(config.data.source_path, config.train.scenario, env_cfg.route_file),
                out_csv_name=env_cfg.agent_result,
                single_agent=env_cfg.single_agent,
                use_gui=env_cfg.gui,
                num_seconds=int(env_cfg.num_steps),
            )
            test_reward = evaluate(test_env, learner, 3, step)
            learner.save(os.path.join(config.train.model_path, config.train.model_file))

    end = time.time() - start
    print(f'Training is completed after {alg_cfg.steps} steps with mean reward {np.mean(mean_rew):.3f} / {test_reward:.3f}. It took {end/60:.2f} mins.')


def evaluate(env, learner, evaluate_times, step, verbose=True):
        evaluate_reward = 0
        evaluate_rewards = []
        learner.model.eval()
        for _ in range(evaluate_times):
            state = env.reset()
            done = False
            episode_reward = 0
            while not done:
                action = learner.act(state, step=0)
                next_state, reward, done, trunc, _ = env.step(action)
                episode_reward += reward
                state = next_state
                if trunc:
                    break
            evaluate_reward += episode_reward
        learner.model.train()
        evaluate_reward /= evaluate_times
        evaluate_rewards.append(evaluate_reward)
        if verbose:
            print(f"After {step} steps evaluate_reward:{np.mean(evaluate_rewards)}") 
        return np.mean(evaluate_rewards)


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config_path=args.config
    #print(f'is sumo dir exist {os.path.isdir("/usr/share/sumo")}')
    #print(f'ENV variables: {os.environ}')
    config = load_config(config_path)
    #print(f'CUDA is available {torch.cuda.is_available()}')
    device = torch.device(f'cuda:{int(config.train.device_id)}' if torch.cuda.is_available() else "cpu")

    train(config, device)
