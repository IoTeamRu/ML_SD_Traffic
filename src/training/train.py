import os
import sys
from data.utils import load_config
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

    config = load_config(config_path)
    env_cfg = config.train.environment
    alg_cfg = config.train.hyper

    device = torch.device(f'cuda:{config.train.device_id}' if torch.cuda.is_available() else "cpu")

    env = SumoEnvironment(
        net_file=env_cfg.net_file,
        route_file=env_cfg.route_file,
        out_csv_name=env_cfg.agent_result,
        single_agent=env_cfg.single_agent,
        use_gui=env_cfg.gui,
        num_seconds=env_cfg.num_steps,
    )

    model = DQN(
        env=env,
        device=device,
        policy=alg_cfg.policy,
        learning_rate=alg_cfg.lr,
        learning_starts=alg_cfg.start_learning,
        train_freq=alg_cfg.train_freq,
        target_update_interval=alg_cfg.target_update,
        exploration_initial_eps=alg_cfg.eps_start,
        exploration_final_eps=alg_cfg.eps_stop,
        verbose=alg_cfg.verbose,
    )

    model.learn(total_timesteps=int(1e6))
    model.save('saved_models/DQN_model')
