import argparse
import copy
import logging
import logging.config
import os
import pprint
import sys
from datetime import datetime

import torch
import yaml

from minidl.runner import RunnerBuilder
from minidl.utils.seed import set_seed
from run import deep_merge, get_log_level, load_config

# CONFIG_FILES = [
#     "configs/folds/fold1.yaml",
#     "configs/folds/fold2.yaml",
#     "configs/folds/fold3.yaml",
#     "configs/folds/fold4.yaml",
#     "configs/folds/fold5.yaml",
# ]

CONFIG_FILES = [
    "configs/folds_1phase/fold1.yaml",
    "configs/folds_1phase/fold2.yaml",
    "configs/folds_1phase/fold3.yaml",
    "configs/folds_1phase/fold4.yaml",
    "configs/folds_1phase/fold5.yaml",
]

# CONFIG_FILES = [
#     "configs/folds_2phase/fold1.yaml",
#     "configs/folds_2phase/fold2.yaml",
#     "configs/folds_2phase/fold3.yaml",
#     "configs/folds_2phase/fold4.yaml",
#     "configs/folds_2phase/fold5.yaml",
# ]


def run_single_fold(config, fold_config_path, args, work_dir_base, experiment_id_base):
    """
    Run a single experiment fold. Modified to return success status and metrics.
    Logging setup remains as per user's original version.

    Returns:
        dict: {'success': bool, 'metrics': dict | None}
    """
    fold_config = copy.deepcopy(config)
    master_logger = logging.getLogger("minidl.master")

    try:
        with open(fold_config_path, "r") as f:
            dataset_config = yaml.load(f, Loader=yaml.SafeLoader) or {}
            fold_config = deep_merge(fold_config, dataset_config)
    except FileNotFoundError:
        if master_logger.hasHandlers():
            master_logger.warning(f"Could not find fold config file {fold_config_path}. Skipping fold.")
        else:
            print(f"Warning: Could not find fold config file {fold_config_path}. Skipping fold.")
        return {"success": False, "metrics": None}
    except Exception as e:
        if master_logger.hasHandlers():
            master_logger.error(f"Error loading fold config {fold_config_path}: {e}")
        else:
            print(f"Error loading fold config {fold_config_path}: {e}")
        return {"success": False, "metrics": None}

    fold_name = os.path.splitext(os.path.basename(fold_config_path))[0]

    if args.work_dir:
        work_dir = os.path.join(work_dir_base, fold_name)
    elif fold_config.get("experiment", {}).get("work_dir"):
        original_work_dir = fold_config["experiment"]["work_dir"]
        work_dir = f"{original_work_dir}_{fold_name}"
    else:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        work_dir = os.path.join(project_root, "output", f"{experiment_id_base}_{fold_name}")

    if "experiment" in fold_config and "name" in fold_config["experiment"]:
        fold_config["experiment"]["name"] = f"{fold_config['experiment']['name']}_{fold_name}"
    elif "experiment" not in fold_config:
        fold_config["experiment"] = {}
        fold_config["experiment"]["name"] = f"{experiment_id_base}_{fold_name}"
    else:
        fold_config["experiment"]["name"] = f"{experiment_id_base}_{fold_name}"

    fold_config["work_dir"] = work_dir
    os.makedirs(work_dir, exist_ok=True)

    logs_dir = os.path.join(work_dir, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    logging_section = fold_config.get("logging", {})

    if args.log_level:
        log_level = args.log_level.upper()
    elif logging_section.get("root", {}).get("level"):
        log_level = logging_section["root"]["level"]
    else:
        log_level = "INFO"

    log_level_int = get_log_level(log_level)
    has_logging_config = all(key in logging_section for key in ["version", "formatters", "handlers"])

    fold_experiment_id = f"{experiment_id_base}_{fold_name}"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if has_logging_config:
        logging_config = {
            "version": logging_section["version"],
            "disable_existing_loggers": logging_section.get("disable_existing_loggers", False),
            "formatters": logging_section["formatters"],
            "handlers": logging_section["handlers"].copy(),
            "loggers": logging_section.get("loggers", {}),
            "root": logging_section.get("root", {"level": log_level, "handlers": ["console"]}),
        }
        if "file" in logging_config["handlers"]:
            file_handler = logging_config["handlers"]["file"]
            if "filename" in file_handler:
                filename = file_handler["filename"]
                if not os.path.isabs(filename):
                    filename = os.path.join(logs_dir, f"{fold_experiment_id}_{timestamp}.log")
                file_handler["filename"] = filename
                os.makedirs(os.path.dirname(filename), exist_ok=True)
        if "loggers" not in logging_config:
            logging_config["loggers"] = {}
        log_name = f"minidl.{fold_name}"
        logging_config["loggers"][log_name] = {"level": log_level, "handlers": list(logging_config["handlers"].keys()), "propagate": False}
        logging.config.dictConfig(logging_config)
        logger = logging.getLogger(log_name)
    else:
        fold_log_handler = logging.FileHandler(os.path.join(logs_dir, f"{fold_experiment_id}_{timestamp}.log"))
        fold_log_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        log_name = f"minidl.{fold_name}"
        logger = logging.getLogger(log_name)
        if logger.hasHandlers():
            logger.handlers.clear()
        logger.setLevel(log_level_int)
        logger.addHandler(fold_log_handler)
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(console_handler)
        logger.propagate = False
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
    if device is None:
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        logger.info(f"[{fold_name}] Using fallback device: {device}")

    seed = fold_config.get("environment", {}).get("seed", 42)
    set_seed(seed)
    logger.info(f"[{fold_name}] Set random seed to {seed}")

    fold_metrics = None
    try:
        logger.info(f"[{fold_name}] Initializing runner...")
        runner = RunnerBuilder.build_runner(fold_config, device)
        runner.call_hooks("before_run")
        logger.info(f"[{fold_name}] Starting experiment with runner: {fold_config['runner']['name']}")

        fold_metrics = runner.run()

        runner.call_hooks("after_run")
        logger.info(f"[{fold_name}] Experiment completed successfully.")

        return {"success": True, "metrics": fold_metrics}

    except Exception as e:
        logger.error(f"[{fold_name}] Experiment failed: {e}", exc_info=True)
        return {"success": False, "metrics": None}


def main():
    parser = argparse.ArgumentParser(description="Train PyTorch models using multiple dataset configurations.")

    parser.add_argument("config", type=str, help="Path to the base YAML configuration file.")
    parser.add_argument("--work_dir", type=str, default=None, help="Base output directory (overrides config). Folds will be in subdirs.")
    parser.add_argument("--experiment_id", type=str, default="BreastMRI", help="Base experiment ID to save the outputs")
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Set the logging level (overrides config).",
    )
    parser.add_argument("--device", type=str, default=None, help="Device to run on (e.g., 'cuda:0', 'cpu')")
    parser.add_argument("--sequential", action="store_true", help="Run folds sequentially instead of stopping after first failure")

    args = parser.parse_args()

    # Load the base configuration
    try:
        base_config = load_config(args.config)
    except FileNotFoundError:
        print(f"Error: Base configuration file not found at {args.config}")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading base configuration {args.config}: {e}")
        sys.exit(1)

    master_log_level = get_log_level(args.log_level)
    master_logger = logging.getLogger("minidl.master")
    if master_logger.hasHandlers():
        master_logger.handlers.clear()
    master_logger.setLevel(master_log_level)
    master_logger.propagate = False
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    master_logger.addHandler(console_handler)

    if args.work_dir:
        summary_base_dir = args.work_dir
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        summary_base_dir = os.path.join(script_dir, "output", f"{args.experiment_id}_summary")
    os.makedirs(summary_base_dir, exist_ok=True)
    summary_log_path = os.path.join(summary_base_dir, "cross_validation_summary.log")
    master_logger.info(f"Summary log will be written to: {summary_log_path}")

    results_with_metrics = {}

    for fold_config_path in CONFIG_FILES:
        fold_name = os.path.splitext(os.path.basename(fold_config_path))[0]
        master_logger.info(f"Starting experiment for fold: {fold_name}")

        fold_result = run_single_fold(base_config, fold_config_path, args, args.work_dir, args.experiment_id)

        results_with_metrics[fold_name] = fold_result

        status_str = "SUCCESS" if fold_result["success"] else "FAILURE"
        master_logger.info(f"Fold {fold_name} finished with status: {status_str}")
        if fold_result["success"] and fold_result["metrics"]:
            master_logger.info(f"Fold {fold_name} Metrics:\n{pprint.pformat(fold_result['metrics'])}")

        if not fold_result["success"] and not args.sequential:
            master_logger.error(f"Fold {fold_name} failed. Stopping further processing.")
            break

    # --- Report and Log Summary Results ---
    master_logger.info("========== Multi-fold Experiment Results ==========")
    all_succeeded = True
    num_processed = len(results_with_metrics)

    results_report = {}
    for fold, result in results_with_metrics.items():
        status_str = "SUCCESS" if result["success"] else "FAILURE"
        results_report[fold] = status_str
        master_logger.info(f"Fold {fold}: {status_str}")
        if not result["success"]:
            all_succeeded = False

    try:
        with open(summary_log_path, "a") as f:
            f.write(f"========== CV Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ==========\n")
            f.write(f"Base Config: {args.config}\n")
            f.write(f"Sequential Mode: {args.sequential}\n")
            f.write(f"Processed Folds: {num_processed}/{len(CONFIG_FILES)}\n")
            f.write("--- Results ---\n")
            for fold, result in results_with_metrics.items():
                status_str = "SUCCESS" if result["success"] else "FAILURE"
                f.write(f"Fold {fold}: {status_str}\n")
                if result["success"] and result["metrics"]:
                    f.write("  Metrics:\n")
                    f.write(pprint.pformat(result["metrics"], indent=4))
                    f.write("\n")
                elif result["success"]:
                    f.write("  Metrics: None reported.\n")
            f.write("--- Overall ---\n")
            # Determine overall status string for file
            if num_processed == 0:
                overall_status = "No folds processed."
            elif all_succeeded and num_processed == len(CONFIG_FILES):
                overall_status = "All folds completed successfully!"
            elif all_succeeded and num_processed < len(CONFIG_FILES):
                overall_status = f"Run stopped early, but all {num_processed} processed folds succeeded."
            else:
                failed_folds = [f for f, r in results_with_metrics.items() if not r["success"]]
                overall_status = f"One or more folds failed: {', '.join(failed_folds)}"
            f.write(f"{overall_status}\n")
            f.write("=" * 60 + "\n\n")
        master_logger.info(f"Appended summary to {summary_log_path}")
    except Exception as e:
        master_logger.error(f"Failed to write summary log to {summary_log_path}: {e}")

    if not results_report:
        master_logger.error("No folds were processed.")
        return 1
    elif all(result == "SUCCESS" for result in results_report.values()):
        if len(results_report) == len(CONFIG_FILES):
            master_logger.info("All folds completed successfully!")
        else:
            master_logger.warning(f"Run stopped early, but all {len(results_report)} processed folds succeeded.")
        return 0
    else:
        master_logger.error("One or more folds failed.")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
