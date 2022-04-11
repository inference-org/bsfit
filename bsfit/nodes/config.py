# note: the doctsring code below within
# """ is converted to a restructuredText
# .rst file by sphinx to automatically
# generate the api's documentation
#
# docstring style used: Google style
"""
    Pipeline configuration module

    Copyright 2022 by Steeve Laquitaine, GNU license 
"""

import argparse


def parametrize_pipe():
    """get terminal args to parametrize the pipeline

    Returns:
        (dict): parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="This runs analytical pipelines."
    )
    parser.add_argument(
        "--model",
        choices=["standard_bayes", "cardinal_bayes"],
        type=str,
        help="choose a model",
    )
    parser.add_argument(
        "--analysis",
        choices=["fit", "simulate_data"],
        type=str,
        help="choose an analysis",
    )
    args = parser.parse_args()
    return args
