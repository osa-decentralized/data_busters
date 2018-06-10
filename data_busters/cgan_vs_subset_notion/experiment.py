"""
This script runs experiment with sets generation.

Author: Nikolay Lysenko
"""


import argparse
import os
from typing import Dict, Any

import yaml

from data_busters.cgan_vs_subset_notion import cgan_model, training_set


def parse_settings(path_to_config: str) -> Dict[str, Any]:
    """
    Extract settings from YAML config.

    :param path_to_config:
        full path to config of model's training
    :return:
        parsed settings
    """
    with open(path_to_config) as in_file:
        settings = yaml.load(in_file)
    return settings


def parse_cli_args() -> argparse.Namespace:
    """
    Parse arguments passed via Command Line Interface (CLI).

    :return:
        namespace with arguments
    """
    parser = argparse.ArgumentParser(description='Transactions tweaking')
    parser.add_argument(
        '-c', '--path_to_config', type=str, default=None,
        help='full path to YAML config, default value may suit you'
    )
    cli_args = parser.parse_args()
    cli_args.path_to_config = cli_args.path_to_config or os.path.join(
        os.path.dirname(__file__), 'config.yml'
    )
    return cli_args


def main():
    cli_args = parse_cli_args()
    setup = parse_settings(cli_args.path_to_config)

    dataset = training_set.generate_data(
        setup['data']['size'], setup['data']['n_items']
    )
    cgan_model.train(
        dataset,
        setup['data']['n_items'],
        setup['generator']['z_dim'],
        setup['discriminator']['layer_sizes'],
        setup['generator']['layer_sizes'],
        setup['optimization']['n_epochs'],
        setup['optimization']['batch_size'],
        setup['optimization']['learning_rate'],
        setup['optimization']['beta_one']
    )


if __name__ == '__main__':
    main()
