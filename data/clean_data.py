import os
import shutil
import argparse
from utils import load_config


def clean_data(config_path):
    config = load_config(config_path)

    for root, dirs, files in os.walk(config.data.temp_path):
        directory = root.split('/')[-1]
        if directory not in config.data.scenarios:
            if root != config.data.temp_path:
                shutil.rmtree(root)
        for file in files:
            try:
                f, e = os.path.splitext(os.path.splitext(file)[0])
                if e != '.net' and e != '.rou':
                    os.remove(os.path.join(root, file))
            except Exception as e:
                pass

    os.rename(config.data.temp_path, config.data.source_path)


if __name__=='__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--config', dest='config', required=True)
    args = args_parser.parse_args()

    clean_data(config_path=args.config)
