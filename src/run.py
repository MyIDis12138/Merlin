# src/run.py
import argparse
import logging.config
import os

import yaml


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch model.")

    parser.add_argument(
        "config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--model_name", type=str, help="Override the model name.")

    parser.add_argument(
        "--hidden_size", type=int, help="Override the model hidden size."
    )
    parser.add_argument(
        "--batch_size", type=int, help="Override the training batch size."
    )
    parser.add_argument(
        "--learning_rate", type=float, help="Override the learning rate."
    )
    parser.add_argument(
        "--epochs", type=int, help="Override the number of training epochs."
    )

    parser.add_argument(
        "--train_path", type=str, help="Override the training data path."
    )
    parser.add_argument(
        "--val_path", type=str, help="Override the validation data path."
    )
    parser.add_argument(
        "--output_dir", type=str, default="output", help="Path to save the outputs"
    )

    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (overrides config).",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if args.log_level:
        log_level = args.log_level.upper()
    elif (
        "logging" in config
        and "root" in config["logging"]
        and "level" in config["logging"]["root"]
    ):
        log_level = config["logging"]["root"]["level"]
    else:
        log_level = "INFO"

    if "logging" in config:
        logging.config.dictConfig(config["logging"])
        logger = logging.getLogger("minidl")
        logger.setLevel(log_level)
    else:
        logging.basicConfig(level=log_level)
        logger = logging.getLogger(__name__)
        logger.warning("No logging configuration found. Using basicConfig.")

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # train_path = os.path.join(project_root, config["data"]["train_path"])
    # val_path = os.path.join(project_root, config["data"]["val_path"])
    output_dir = os.path.join(project_root, args.output_dir)

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
