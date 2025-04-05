import argparse
import logging
import logging.config
import os
from datetime import datetime

import torch
import yaml

from minidl.runner import RunnerBuilder


def get_log_level(level_str):
    """Convert a log level string to the corresponding logging level.

    Args:
        level_str: Log level string (e.g., 'INFO', 'DEBUG')

    Returns:
        int: Corresponding logging level
    """
    level_str = level_str.upper()
    level_map = {
        "CRITICAL": logging.CRITICAL,
        "ERROR": logging.ERROR,
        "WARNING": logging.WARNING,
        "INFO": logging.INFO,
        "DEBUG": logging.DEBUG,
        "NOTSET": logging.NOTSET,
    }
    return level_map.get(level_str, logging.INFO)


def deep_merge(dict1, dict2):
    """
    Deep merge two dictionaries. dict2 values will override dict1 values for the same keys.
    """
    merged = dict1.copy()
    for key, value in dict2.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path):
    """Load and merge configuration files."""

    # Add a constructor for scientific notation
    def scientific_constructor(loader, node):
        value = loader.construct_scalar(node)
        try:
            return float(value)
        except ValueError:
            return value

    yaml.SafeLoader.add_constructor("tag:yaml.org,2002:str", scientific_constructor)

    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.SafeLoader) or {}

    # Get the directory of the main config file
    config_dir = os.path.dirname(config_path)

    # Load and merge imported configs
    if "imports" in config:
        for import_path in config["imports"]:
            import_path = os.path.join(config_dir, import_path)
            try:
                with open(import_path, "r") as f:
                    imported_config = yaml.load(f, Loader=yaml.SafeLoader) or {}
                    config = deep_merge(config, imported_config)
            except FileNotFoundError:
                print(f"Warning: Could not find config file {import_path}")
                continue

        del config["imports"]

    return config


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch model.")

    parser.add_argument(
        "config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default="BreastMRI",
        help="Experiment ID to save the outputs",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (overrides config).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to run on (e.g., 'cuda:0', 'cpu')",
    )
    parser.add_argument(
        "--runner",
        type=str,
        default=None,
        help="Runner type to use (overrides config)",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )

    args = parser.parse_args()

    # Load and merge configurations
    config = load_config(args.config)

    # Setup output directory
    if args.work_dir:
        work_dir = args.work_dir
    elif config.get("experiment", {}).get("work_dir"):
        work_dir = config["experiment"]["work_dir"]
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        work_dir = os.path.join(project_root, "output", args.experiment_id)

    os.makedirs(work_dir, exist_ok=True)

    config["work_dir"] = work_dir

    # Configure logging
    logs_dir = os.path.join(work_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logging_section = config.get("logging", {})

    if args.log_level:
        log_level = args.log_level.upper()
    elif logging_section.get("root", {}).get("level"):
        log_level = logging_section["root"]["level"]
    else:
        log_level = "INFO"

    log_level_int = get_log_level(log_level)

    has_logging_config = all(key in logging_section for key in ["version", "formatters", "handlers"])

    if has_logging_config:
        # Create a copy of the logging config
        logging_config = {
            "version": logging_section["version"],
            "disable_existing_loggers": logging_section.get("disable_existing_loggers", False),
            "formatters": logging_section["formatters"],
            "handlers": logging_section["handlers"].copy(),
            "loggers": logging_section.get("loggers", {}),
            "root": logging_section.get("root", {"level": log_level, "handlers": ["console"]}),
        }

        # Add timestamp to log filename if file handler exists
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if "file" in logging_config["handlers"]:
            file_handler = logging_config["handlers"]["file"]
            if "filename" in file_handler:
                filename = file_handler["filename"]
                # If it's a relative path, make it relative to logs_dir
                if not os.path.isabs(filename):
                    filename = os.path.join(logs_dir, f"{args.experiment_id}_{timestamp}.log")
                file_handler["filename"] = filename

        # Ensure minidl logger exists
        if "minidl" not in logging_config.get("loggers", {}):
            if "loggers" not in logging_config:
                logging_config["loggers"] = {}
            logging_config["loggers"]["minidl"] = {"level": log_level, "handlers": list(logging_config["handlers"].keys()), "propagate": False}

        logging.config.dictConfig(logging_config)
        logger = logging.getLogger("minidl")
    else:
        # Basic logging configuration
        logging.basicConfig(
            level=log_level_int,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(os.path.join(logs_dir, f"{args.experiment_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
            ],
        )
        logger = logging.getLogger("minidl")
        logger.warning("No complete logging configuration found. Using basicConfig.")

    logger.info(f"Output directory: {work_dir}")
    logger.info(f"Logs directory: {logs_dir}")

    # Set runner
    if args.runner:
        if "runner" not in config:
            config["runner"] = {}
        config["runner"]["name"] = args.runner

    if "runner" not in config:
        config["runner"] = {"name": "epoch_based_runner"}
    elif "name" not in config["runner"]:
        config["runner"]["name"] = "epoch_based_runner"

    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    elif config.get("environment", {}).get("device"):
        device = torch.device(config["environment"]["device"])

    try:
        logger.info("Initializing runner...")

        # Create runner using factory
        runner = RunnerBuilder.build_runner(config, device)

        runner.call_hooks("before_run")

        # Run experiment
        logger.info(f"Starting experiment with runner: {config['runner']['name']}")
        runner.run()

        runner.call_hooks("after_run")

        logger.info("Experiment completed successfully.")

    except Exception as e:
        logger.error(f"Experiment failed: {e}")
        raise


if __name__ == "__main__":
    main()
