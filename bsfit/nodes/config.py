
import argparse


def parametrize_pipe():
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
