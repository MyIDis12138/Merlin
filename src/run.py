import argparse
import logging.config
import os

import yaml

from minidl.dataset.dataset_builder import DatasetBuilder


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
    with open(config_path, "r") as f:
        config = yaml.safe_load(f) or {}

    # Get the directory of the main config file
    config_dir = os.path.dirname(config_path)

    # Load and merge imported configs
    if "imports" in config:
        for import_path in config["imports"]:
            import_path = os.path.join(config_dir, import_path)
            try:
                with open(import_path, "r") as f:
                    imported_config = yaml.safe_load(f) or {}
                    # Deep merge imported config with main config
                    config = deep_merge(config, imported_config)
            except FileNotFoundError:
                print(f"Warning: Could not find config file {import_path}")
                continue

        # Remove imports key after processing
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

    args = parser.parse_args()

    # Load and merge configurations
    config = load_config(args.config)

    # Configure logging
    if args.log_level:
        log_level = args.log_level.upper()
    elif "root" in config and "level" in config["root"]:
        log_level = config["root"]["level"]
    else:
        log_level = "INFO"

    # Create logs directory if it doesn't exist
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # Check if we have all required logging configuration keys
    has_logging_config = all(key in config for key in ["version", "formatters", "handlers", "loggers", "root"])

    if has_logging_config:
        # Create a copy of the logging config
        logging_config = {
            "version": config["version"],
            "disable_existing_loggers": config.get("disable_existing_loggers", False),
            "formatters": config["formatters"],
            "handlers": config["handlers"],
            "loggers": config["loggers"],
            "root": config["root"],
        }

        # Add timestamp to log filename
        from datetime import datetime

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        logging_config["handlers"]["file"]["filename"] = os.path.join(logs_dir, f"{args.experiment_id}_{timestamp}.log")

        logging.config.dictConfig(logging_config)
        logger = logging.getLogger("minidl")
        logger.setLevel(log_level)
    else:
        logging.basicConfig(level=log_level)
        logger = logging.getLogger("minidl")
        logger.warning("No logging configuration found. Using basicConfig.")

    # Setup output directory
    output_dir = os.path.join(project_root, "output", args.experiment_id)
    os.makedirs(output_dir, exist_ok=True)

    # Build datasets
    try:
        logger.info("Building datasets...")
        datasets = {}
        # for split in ["train", "val", "test"]:
        for split in ["train"]:
            datasets[split] = DatasetBuilder.build_dataset(config, split)
            logger.info(f"Successfully built {split} dataset with {len(datasets[split])} samples")

            # Log example from each split
            sample = datasets[split][0]
            logger.debug(f"\nExample from {split} split:")
            logger.debug(f"Sample images shape: {sample['images'].shape}")
            logger.debug(f"Patient ID: {sample['patient_id']}")
            logger.debug(f"Molecular subtype: {sample['molecular_subtype']}")
            logger.debug(f"Available clinical features: {list(sample['clinical_features'].keys())}")

        # TODO: Add model training code here
        # model = build_model(config)
        # train_model(model, datasets['train'], datasets['val'], config)

    except Exception as e:
        logger.error(f"Failed to build datasets: {e}")
        raise

    sample = datasets["train"][0]
    print(datasets["train"][0])
    print(sample["images"].shape)
    logger.info("Training complete.")


if __name__ == "__main__":
    main()
