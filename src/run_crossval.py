import argparse
import logging
import logging.config
import os
import sys
from datetime import datetime
import copy

import torch
import yaml

# Import from your existing code
from minidl.runner import RunnerBuilder
from minidl.utils.seed import set_seed

# Import necessary functions from run.py
from run import get_log_level, deep_merge, load_config

CONFIG_FILES = [
    "configs/folds/fold1.yaml",
    "configs/folds/fold2.yaml",
    "configs/folds/fold3.yaml",
    "configs/folds/fold4.yaml",
    "configs/folds/fold5.yaml",
]

def run_single_fold(config, fold_config_path, args, work_dir_base, experiment_id_base):
    """Run a single experiment with the specified fold configuration."""
    
    # Make a deep copy of the config to avoid modifying the original
    fold_config = copy.deepcopy(config)
    
    # Load the fold-specific dataset config
    try:
        with open(fold_config_path, "r") as f:
            dataset_config = yaml.load(f, Loader=yaml.SafeLoader) or {}
            fold_config = deep_merge(fold_config, dataset_config)
    except FileNotFoundError:
        print(f"Warning: Could not find fold config file {fold_config_path}")
        return False
        
    # Extract fold name for naming
    fold_name = os.path.splitext(os.path.basename(fold_config_path))[0]
    
    # Update work_dir and experiment name for this fold
    if args.work_dir:
        work_dir = os.path.join(args.work_dir, fold_name)
    elif fold_config.get("experiment", {}).get("work_dir"):
        original_work_dir = fold_config["experiment"]["work_dir"]
        work_dir = f"{original_work_dir}_{fold_name}"
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        work_dir = os.path.join(project_root, "output", f"{experiment_id_base}_{fold_name}")
    
    # Update the experiment name if it exists
    if "experiment" in fold_config and "name" in fold_config["experiment"]:
        fold_config["experiment"]["name"] = f"{fold_config['experiment']['name']}_{fold_name}"
    
    # Set the work_dir in the config
    fold_config["work_dir"] = work_dir
    os.makedirs(work_dir, exist_ok=True)
    
    # Configure logging
    logs_dir = os.path.join(work_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)
    
    # Set up logging similar to the original script
    logging_section = fold_config.get("logging", {})
    
    if args.log_level:
        log_level = args.log_level.upper()
    elif logging_section.get("root", {}).get("level"):
        log_level = logging_section["root"]["level"]
    else:
        log_level = "INFO"
    
    log_level_int = get_log_level(log_level)
    has_logging_config = all(key in logging_section for key in ["version", "formatters", "handlers"])
    
    # Configure fold-specific logger
    fold_experiment_id = f"{experiment_id_base}_{fold_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
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
        if "file" in logging_config["handlers"]:
            file_handler = logging_config["handlers"]["file"]
            if "filename" in file_handler:
                filename = file_handler["filename"]
                # If it's a relative path, make it relative to logs_dir
                if not os.path.isabs(filename):
                    filename = os.path.join(logs_dir, f"{fold_experiment_id}_{timestamp}.log")
                file_handler["filename"] = filename
        
        # Create fold-specific logger
        if "loggers" not in logging_config:
            logging_config["loggers"] = {}
        
        log_name = f"minidl.{fold_name}"
        logging_config["loggers"][log_name] = {
            "level": log_level, 
            "handlers": list(logging_config["handlers"].keys()), 
            "propagate": False
        }
        
        logging.config.dictConfig(logging_config)
        logger = logging.getLogger(log_name)
    else:
        # Basic logging configuration
        fold_log_handler = logging.FileHandler(
            os.path.join(logs_dir, f"{fold_experiment_id}_{timestamp}.log")
        )
        fold_log_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        
        log_name = f"minidl.{fold_name}"
        logger = logging.getLogger(log_name)
        logger.setLevel(log_level_int)
        logger.addHandler(fold_log_handler)
        
        # Also add a console handler for this fold's logs
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        )
        logger.addHandler(console_handler)
        
        logger.warning(f"[{fold_name}] No complete logging configuration found. Using basic config.")
    
    logger.info(f"[{fold_name}] Output directory: {work_dir}")
    logger.info(f"[{fold_name}] Logs directory: {logs_dir}")
    
    
    if "runner" not in fold_config:
        fold_config["runner"] = {"name": "epoch_based_runner"}
    elif "name" not in fold_config["runner"]:
        fold_config["runner"]["name"] = "epoch_based_runner"
    
    # Set device
    device = None
    if args.device:
        device = torch.device(args.device)
    elif fold_config.get("environment", {}).get("device"):
        device = torch.device(fold_config["environment"]["device"])
    
    seed = fold_config.get("environment", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"[{fold_name}] Set random seed to {seed}")
    
    try:
        logger.info(f"[{fold_name}] Initializing runner...")
        
        # Create runner using factory
        runner = RunnerBuilder.build_runner(fold_config, device)
        
        runner.call_hooks("before_run")
        
        # Run experiment
        logger.info(f"[{fold_name}] Starting experiment with runner: {fold_config['runner']['name']}")
        runner.run()
        
        runner.call_hooks("after_run")
        
        logger.info(f"[{fold_name}] Experiment completed successfully.")
        return True
        
    except Exception as e:
        logger.error(f"[{fold_name}] Experiment failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch models using multiple dataset configurations.")
    
    parser.add_argument(
        "config",
        type=str,
        default="config.yaml",
        help="Path to the base YAML configuration file.",
    )
    parser.add_argument(
        "--work_dir",
        type=str,
        default=None,
        help="Base output directory (overrides config)",
    )
    parser.add_argument(
        "--experiment_id",
        type=str,
        default="BreastMRI",
        help="Base experiment ID to save the outputs",
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
        "--sequential",
        action="store_true",
        help="Run folds sequentially instead of stopping after first failure",
    )
    
    args = parser.parse_args()
    
    # Load the base configuration
    base_config = load_config(args.config)
    
    # Create a master logger
    master_log_level = get_log_level(args.log_level if args.log_level else "INFO")
    master_logger = logging.getLogger("minidl.master")
    master_logger.setLevel(master_log_level)
    
    # Add a console handler for the master logger
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    master_logger.addHandler(console_handler)
    
    # Track successes and failures
    results = {}
    
    # Process each fold configuration
    for fold_config_path in CONFIG_FILES:
        fold_name = os.path.splitext(os.path.basename(fold_config_path))[0]
        master_logger.info(f"Starting experiment for fold: {fold_name}")
        
        # Run the experiment for this fold
        success = run_single_fold(
            base_config, fold_config_path, args, args.work_dir, args.experiment_id
        )
        
        results[fold_name] = "SUCCESS" if success else "FAILURE"
        
        # If sequential is not set, stop after first failure
        if not success and not args.sequential:
            master_logger.error(f"Fold {fold_name} failed. Stopping further processing.")
            break
    
    # Report results
    master_logger.info("== Multi-fold Experiment Results ==")
    for fold, result in results.items():
        master_logger.info(f"Fold {fold}: {result}")
    
    # Overall status
    if all(result == "SUCCESS" for result in results.values()):
        master_logger.info("All folds completed successfully!")
        return 0
    else:
        master_logger.error("One or more folds failed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())