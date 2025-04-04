import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from minidl.layers.attention_layers import MultiHeadAttention
from minidl.layers.resnet3d import resnet3d18, resnet3d34, resnet3d50
from minidl.model.model_registry import ModelRegistry
from minidl.utils.pretrained_loader import load_pretrained_weights

logger = logging.getLogger(__name__)


@ModelRegistry.register("resnet3d_model")
class ResNet3DMRIModel(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        input_channels: int = 1,
        n_classes: int = 2,
        pretrained: bool = False,
        pretrained_model: str = "",
        out_dropout: float = 0.3,
        **kwargs,
    ):
        super().__init__()
        self.backbone = backbone

        if backbone == "resnet18":
            self.backbone = resnet3d18(input_channels=input_channels, num_classes=n_classes)
        elif backbone == "resnet34":
            self.backbone = resnet3d34(input_channels=input_channels, num_classes=n_classes)
        elif backbone == "resnet50":
            self.backbone = resnet3d50(input_channels=input_channels, num_classes=n_classes)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        if backbone in ["resnet18", "resnet34"]:
            feature_dim = 512
        else:
            feature_dim = 2048

        if pretrained:
            self.backbone = load_pretrained_weights(model=self.backbone, pretrained=pretrained, model_name=pretrained_model)
            logger.info(f"Loaded pretrained weights for {backbone}")

        self.backbone.fc = nn.Identity()

        self.shared_extractor = self.backbone

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 3, feature_dim * 2), nn.GELU(), nn.Dropout(out_dropout), nn.Linear(feature_dim * 2, n_classes)
        )

    def _load_pretrained_weights(self, pretrained_path: str):
        """Load pretrained weights from a file"""
        try:
            pretrained_dict = torch.load(pretrained_path, map_location="cpu")

            if "state_dict" in pretrained_dict:
                pretrained_dict = pretrained_dict["state_dict"]

            model_dict = self.backbone.state_dict()

            if "conv1.weight" in pretrained_dict and pretrained_dict["conv1.weight"].shape[1] != self.backbone.conv1.weight.shape[1]:
                if self.backbone.conv1.weight.shape[1] == 1 and pretrained_dict["conv1.weight"].shape[1] == 3:
                    conv1_weight = pretrained_dict["conv1.weight"]
                    pretrained_dict["conv1.weight"] = conv1_weight.mean(dim=1, keepdim=True)
                elif self.backbone.conv1.weight.shape[1] > pretrained_dict["conv1.weight"].shape[1]:
                    conv1_weight = pretrained_dict["conv1.weight"]
                    pretrained_dict["conv1.weight"] = conv1_weight.repeat(1, self.backbone.conv1.weight.shape[1] // conv1_weight.shape[1], 1, 1, 1)

            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and "fc" not in k}

            model_dict.update(pretrained_dict)
            self.backbone.load_state_dict(model_dict)
            print(f"Successfully loaded pretrained weights from {pretrained_path}")
        except Exception as e:
            print(f"Failed to load pretrained weights: {e}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: batch of 3 MRI images [B, 3, D, H, W]
        features = []

        for i in range(3):
            # Extract individual MRI sequence
            mri = x[:, i : i + 1]  # [B, 1, D, H, W]

            feat = self.shared_extractor.forward_features(mri)

            features.append(feat)

        combined_features = torch.cat(features, dim=1)  # [B, d_model*3]

        combined_features = self.avgpool(combined_features)
        combined_features = torch.flatten(combined_features, 1)
        logits = self.classifier(combined_features)

        return logits


@ModelRegistry.register("mri_resnet3d")
class MRI_ResNet3D(nn.Module):
    """
    ResNet3D-based model for MRI analysis using pre-trained weights.
    This model serves as a drop-in replacement for the existing MRI_baseline model,
    but uses a 3D ResNet backbone with optional pre-trained weights.

    Args:
        n_classes: Number of output classes
        d_model: Dimension of the model features
        out_dropout: Dropout rate for the output layer
        backbone: type of ResNet backbone ('resnet18', 'resnet34', 'resnet50')
        pretrained: Whether to use pretrained weights or path to weights file
        pretrained_model: Name of the pretrained model to use
    """

    def __init__(
        self,
        n_classes: int = 2,
        d_model: int = 256,
        out_dropout: float = 0.3,
        backbone: str = "resnet18",
        pretrained: bool | str = False,
        pretrained_model: str | None = None,
    ):
        super().__init__()

        self.d_model = d_model
        self.backbone = backbone

        # Determine the feature dimension based on backbone type
        if backbone in ["resnet18", "resnet34"]:
            feature_dim = 512
        else:  # resnet50, resnet101, resnet152
            feature_dim = 2048

        if backbone == "resnet18":
            self.backbone = resnet3d18(input_channels=1)
        elif backbone == "resnet34":
            self.backbone = resnet3d34(input_channels=1)
        elif backbone == "resnet50":
            self.backbone = resnet3d50(input_channels=1)
        else:
            raise ValueError(f"Unsupported backbone type: {backbone}")

        if pretrained:
            self.backbone = load_pretrained_weights(model=self.backbone, pretrained=pretrained, model_name=pretrained_model)
            logger.info(f"Loaded pretrained weights for {backbone}")

        self.backbone.fc = nn.Identity()

        self.shared_extractor = self.backbone

        self.feature_adapter = nn.Sequential(
            nn.Conv3d(feature_dim, d_model, kernel_size=1, bias=False), nn.BatchNorm3d(d_model), nn.ReLU(inplace=True)
        )

        self.mri_adapters = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv3d(d_model, d_model, kernel_size=3, padding=1),
                    nn.BatchNorm3d(d_model),
                    nn.ReLU(inplace=True),
                    nn.Conv3d(d_model, d_model, kernel_size=3, padding=1),
                    nn.BatchNorm3d(d_model),
                )
                for _ in range(3)
            ]
        )

        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Conv3d(d_model, d_model // 2, 1), nn.GELU(), nn.Conv3d(d_model // 2, d_model, 1), nn.Sigmoid()
        )

        self.attention = MultiHeadAttention(d_model, 4)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model * 3, d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(d_model)

        self.classifier = nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(out_dropout), nn.Linear(d_model * 2, n_classes))

        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize weights for the non-pretrained parts of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d) and m not in self.backbone.modules():
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) and m not in self.backbone.modules():
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear) and m not in self.backbone.modules():
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features using the backbone model.

        Args:
            x: Input tensor of shape [B, C, D, H, W]

        Returns:
            Feature tensor
        """
        features = self.backbone.forward_features(x)

        features = self.feature_adapter(features)
        return features

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape [B, 3, D, H, W] containing 3 MRI sequences

        Returns:
            Output tensor of shape [B, n_classes]
        """
        # x: tensor of 3 MRI images, shaped [B, 3, D, H, W]
        features = []

        for i in range(3):
            mri = x[:, i : i + 1]  # [B, 1, D, H, W]

            feat = self.forward_features(mri)

            feat = self.mri_adapters[i](feat)

            attn = self.spatial_attention(feat)
            feat = feat * attn

            feat_gap = F.adaptive_avg_pool3d(feat, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, d_model]
            features.append(feat_gap)

        V = torch.stack(features, dim=1)  # [B, 3, d_model]

        attn_output, _ = self.attention(V, V, V)

        x = self.feed_forward(attn_output.view(-1, 3 * self.d_model))

        res_x = torch.mean(attn_output, dim=1)  # [B, d_model]

        x = self.layer_norm(x + res_x)

        logits = self.classifier(x)

        return logits
