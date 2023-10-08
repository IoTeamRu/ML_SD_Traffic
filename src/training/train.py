import os
import sys
from data.utils import load_config
import argparse
import torch
from stable_baselines3.dqn.dqn import DQN


if "SUMO_HOME" in os.environ:
    tools = os.path.join(os.environ["SUMO_HOME"], "tools")
    sys.path.append(tools)
else:
    sys.exit("Please declare the environment variable 'SUMO_HOME'")

from sumo_rl import SumoEnvironment  # type: ignore


if __name__ == "__main__":
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()
    config_path=args.config
    #print(f'is sumo dir exist {os.path.isdir("/usr/share/sumo")}')
    #print(f'ENV variables: {os.environ}')
    config = load_config(config_path)
    env_cfg = config.train.environment
    alg_cfg = config.train.hyper
    #print(f'CUDA is available {torch.cuda.is_available()}')
    device = torch.device(f'cuda:{int(config.train.device_id)}' if torch.cuda.is_available() else "cpu")
    #print(f'Launch agent training on {device}')
    env = SumoEnvironment(
        net_file=os.path.join(config.data.source_path, config.train.scenario, env_cfg.net_file),
        route_file=os.path.join(config.data.source_path, config.train.scenario, env_cfg.route_file),
        out_csv_name=env_cfg.agent_result,
        single_agent=env_cfg.single_agent,
        use_gui=env_cfg.gui,
        num_seconds=int(env_cfg.num_steps),
    )

    model = DQN(
        env=env,
        device=device,
        policy=alg_cfg.policy,
        learning_rate=float(alg_cfg.lr),
        learning_starts=int(alg_cfg.start_learning),
        train_freq=int(alg_cfg.train_freq),
        target_update_interval=int(alg_cfg.target_update),
        exploration_initial_eps=float(alg_cfg.eps_start),
        exploration_final_eps=float(alg_cfg.eps_stop),
        verbose=int(alg_cfg.verbose),
    )

    model.learn(total_timesteps=int(float(alg_cfg.total_steps)))
    model.save(os.path.join(config.train.model_path, config.train.model_name))
