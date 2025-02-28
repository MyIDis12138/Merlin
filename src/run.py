# src/run.py
import argparse
import logging.config
import os

import yaml

from minidl.dataset.dataset_builder import DatasetBuilder


def main():
    parser = argparse.ArgumentParser(description="Train a PyTorch model.")

    parser.add_argument(
        "config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument("--model_name", type=str, help="Override the model name.")
    parser.add_argument("--hidden_size", type=int, help="Override the model hidden size.")
    parser.add_argument("--batch_size", type=int, help="Override the training batch size.")
    parser.add_argument("--learning_rate", type=float, help="Override the learning rate.")
    parser.add_argument("--epochs", type=int, help="Override the number of training epochs.")

    # Dataset related arguments
    parser.add_argument("--root_dir", type=str, help="Override the dataset root directory")
    parser.add_argument("--clinical_data_path", type=str, help="Override the clinical data file path")

    parser.add_argument("--output_dir", type=str, default="output", help="Path to save the outputs")

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

    # Override config with command line arguments
    if args.model_name:
        config["model"]["name"] = args.model_name
    if args.hidden_size:
        config["model"]["hidden_size"] = args.hidden_size
    if args.batch_size:
        config["training"]["batch_size"] = args.batch_size
    if args.learning_rate:
        config["training"]["learning_rate"] = args.learning_rate
    if args.epochs:
        config["training"]["epochs"] = args.epochs
    if args.root_dir:
        config["data"]["dataset"]["params"]["root_dir"] = args.root_dir
    if args.clinical_data_path:
        config["data"]["dataset"]["params"]["clinical_data_path"] = args.clinical_data_path

    # Configure logging
    if args.log_level:
        log_level = args.log_level.upper()
    elif "logging" in config and "root" in config["logging"] and "level" in config["logging"]["root"]:
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

    # Setup paths
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_dir = os.path.join(project_root, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    # Build datasets
    try:
        logger.info("Building datasets...")
        datasets = {}
        for split in ["train", "val", "test"]:
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

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
