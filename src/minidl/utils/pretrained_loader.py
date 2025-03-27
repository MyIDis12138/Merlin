import logging
import os

import requests
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Dictionary of available pretrained models with their download URLs
# These are examples - replace with actual sources
PRETRAINED_MODELS = {
    # Med3D models available through Hugging Face
    "resnet3d_10_med3d": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/resolve/main/resnet_10.pth",
    "resnet3d_18_med3d": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18.pth",
    "resnet3d_34_med3d": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34/resolve/main/resnet_34.pth",
    "resnet3d_50_med3d": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50.pth",
    # Med3D models with 23 datasets (larger dataset)
    "resnet3d_10_med3d_23ds": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet10/resolve/main/resnet_10_23dataset.pth",
    "resnet3d_18_med3d_23ds": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet18/resolve/main/resnet_18_23dataset.pth",
    "resnet3d_34_med3d_23ds": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet34/resolve/main/resnet_34_23dataset.pth",
    "resnet3d_50_med3d_23ds": "https://huggingface.co/TencentMedicalNet/MedicalNet-Resnet50/resolve/main/resnet_50_23dataset.pth",
    # Kaggle versions (as fallbacks)
    "resnet3d_18_med3d_kaggle": "https://www.kaggle.com/datasets/jinuahn/medicalnet-pretrained-weights/download/resnet_18.pth",
    "resnet3d_50_med3d_kaggle": "https://www.kaggle.com/datasets/jinuahn/medicalnet-pretrained-weights/download/resnet_50.pth",
    # Models from other repositories
    "resnet3d_18_kinetics": "https://github.com/kenshohara/3D-ResNets-PyTorch/raw/master/pretrained/resnet-18-kinetics.pth",
}


def download_file(url: str, destination: str) -> None:
    """Download a file from a URL to a local destination with progress bar."""
    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))
    block_size = 8192  # 8KB chunks

    with open(destination, "wb") as f, tqdm(total=total_size, unit="B", unit_scale=True, desc=os.path.basename(destination)) as progress_bar:
        for chunk in response.iter_content(chunk_size=block_size):
            if chunk:
                f.write(chunk)
                progress_bar.update(len(chunk))


def get_pretrained_weights(model_name: str, cache_dir: str | None = None, force_download: bool = False) -> str:
    """
    Get pretrained weights file path, downloading if necessary.

    Args:
        model_name: Name of the pretrained model
        cache_dir: Directory to cache the downloaded weights
        force_download: Whether to force download even if the file exists

    Returns:
        Path to the weights file
    """
    if model_name not in PRETRAINED_MODELS:
        raise ValueError(f"Unknown pretrained model: {model_name}. " f"Available models: {list(PRETRAINED_MODELS.keys())}")

    if cache_dir is None:
        cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "minidl", "pretrained")

    os.makedirs(cache_dir, exist_ok=True)

    weights_url = PRETRAINED_MODELS[model_name]
    filename = os.path.basename(weights_url)
    weights_path = os.path.join(cache_dir, filename)

    if not os.path.exists(weights_path) or force_download:
        if weights_url.startswith(("http://", "https://")):
            logger.info(f"Downloading pretrained weights for {model_name}...")
            try:
                download_file(weights_url, weights_path)
                logger.info(f"Successfully downloaded weights to {weights_path}")
            except Exception as e:
                logger.error(f"Failed to download weights: {e}")
                raise
        else:
            if not os.path.exists(weights_url):
                raise FileNotFoundError(f"Pretrained weights file not found: {weights_url}")
            weights_path = weights_url

    return weights_path


def convert_med3d_weights(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """
    Convert weights from MedicalNet (Med3D) format to our model structure.

    Args:
        state_dict: Original state dictionary

    Returns:
        Converted state dictionary
    """
    converted_dict = {}

    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[7:]

        converted_dict[key] = value

    return converted_dict


def load_pretrained_weights(
    model: torch.nn.Module, pretrained: str | bool = True, model_name: str | None = None, cache_dir: str | None = None, force_download: bool = False
) -> torch.nn.Module:
    """
    Load pretrained weights into the model.

    Args:
        model: Model to load weights into
        pretrained: If True, load default weights for the model type;
                   if string, treat as a path to weights file
        model_name: Name of the pretrained model (if pretrained is True)
        cache_dir: Directory to cache downloaded weights
        force_download: Whether to force download even if the file exists

    Returns:
        Model with loaded weights
    """
    if isinstance(pretrained, str) and os.path.exists(pretrained):
        weights_path = pretrained
        logger.info(f"Loading weights from specified path: {weights_path}")

    elif pretrained is True:
        if model_name is None:
            raise ValueError("model_name must be specified when pretrained=True")

        weights_path = get_pretrained_weights(model_name=model_name, cache_dir=cache_dir, force_download=force_download)
        logger.info(f"Loading pretrained weights for {model_name} from {weights_path}")

    else:
        logger.info("No pretrained weights requested, using random initialization")
        return model

    try:
        state_dict = torch.load(weights_path, map_location="cpu")

        if isinstance(state_dict, dict):
            if "state_dict" in state_dict:
                state_dict = state_dict["state_dict"]
            elif "model" in state_dict:
                state_dict = state_dict["model"]

        if model_name and "med3d" in model_name:
            state_dict = convert_med3d_weights(state_dict)

        missing_keys, unexpected_keys = [], []

        try:
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            logger.warning(f"Could not load state dict directly: {e}")
            logger.info("Trying to load weights with key filtering...")

            model_dict = model.state_dict()
            filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict}
            missing_keys = [k for k in model_dict.keys() if k not in filtered_dict]
            unexpected_keys = [k for k in state_dict.keys() if k not in model_dict]

            model_dict.update(filtered_dict)
            model.load_state_dict(model_dict)

        logger.info(f"Loaded pretrained weights from {weights_path}")
        if missing_keys:
            logger.warning(f"Missing keys: {missing_keys}")
        if unexpected_keys:
            logger.warning(f"Unexpected keys: {unexpected_keys}")

    except Exception as e:
        logger.error(f"Failed to load pretrained weights: {e}")
        raise

    return model
