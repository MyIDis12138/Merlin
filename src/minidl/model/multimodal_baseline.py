import logging

import torch
import torch.nn as nn
import torch.nn.functional as F

from minidl.layers.attention_layers import MultiHeadAttention
from minidl.layers.positional_embeddings import FactorizedPositionalEmbedding3D
from minidl.layers.resnet3d import resnet3d18, resnet3d34, resnet3d50
from minidl.model.model_registry import ModelRegistry
from minidl.utils.pretrained_loader import load_pretrained_weights

logger = logging.getLogger(__name__)


@ModelRegistry.register("multimodal_resnet3d")
class MultiModal_ResNet3D(nn.Module):
    """
    ResNet3D-based model for MRI analysis and clinical features using pre-trained weights.
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
        d_clinical: int = 86,
        clinical_dropout: float = 0.1,
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
            nn.AdaptiveAvgPool3d(1), nn.Conv3d(d_model * 3, d_model, 1), nn.GELU(), nn.Conv3d(d_model, d_model * 3, 1), nn.Sigmoid()
        )

        self.position_embeddings = FactorizedPositionalEmbedding3D(self.d_model, 3, 5, 5)
        self.attention = MultiHeadAttention(d_model * 3, 8)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 12),
            nn.GELU(),
            nn.Linear(d_model * 12, d_model * 6),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(d_model * 6)

        self.classifier = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 12), 
            nn.GELU(), 
            nn.Dropout(out_dropout), 
            nn.Linear(d_model * 12, d_model * 12), 
            nn.GELU(), 
            nn.Dropout(out_dropout), 
            nn.Linear(d_model * 12, n_classes))

        self.clinlical_mlp = nn.Sequential(
            nn.Linear(d_clinical, d_model * 8),
            nn.GELU(),
            nn.Dropout(clinical_dropout),
            nn.Linear(d_model * 8, d_model * 8),
            nn.GELU(),
            nn.Dropout(clinical_dropout),
            nn.Linear(d_model * 8, d_model * 3),
            nn.GELU(),
        )

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
        clinical_feat = self.clinlical_mlp(x["clinical_features"])

        x = x["images"]
        features = []

        for i in range(3):
            mri = x[:, i : i + 1]  # [B, 1, D, H, W]

            feat = self.forward_features(mri)  # [B, 255, 3, 5, 5] for half size

            feat = self.mri_adapters[i](feat)  # [B, d_model, D, H, W]
            features.append(feat)

        V = torch.cat(features, dim=1)  # [B, 3 * d_model, D, H, W]
        B, C, D, H, W = V.shape  # C = 3 * d_model

        attn = self.spatial_attention(V)  # [B, 3 * d_model, 1, 1, 1]
        V = V * attn  # [B,  3 * d_model, D, H, W]

        V = V.view(B, 3 * self.d_model, -1).transpose(1, 2)

        V = self.position_embeddings.add_to_input(V, D, H, W)

        clinical_k = clinical_feat.unsqueeze(1)  # (B, 1, C_cat)
        clinical_v = clinical_feat.unsqueeze(1)  # (B, 1, C_cat)
        attn_output, _ = self.attention(V, clinical_k, clinical_v)  # [B, D * H * W, 3 * d_model]

        attn_output = attn_output.transpose(1, 2).view(B, C, D, H, W)
        x_avg = F.adaptive_avg_pool3d(attn_output, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 3 * d_model]
        x_max = F.adaptive_max_pool3d(attn_output, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 3 * d_model]

        x_prev = torch.cat([x_avg, x_max], dim=1)
        x = self.feed_forward(x_prev)  # [B, 6 * d_model]

        x = self.layer_norm(x + x_prev)

        logits = self.classifier(x)

        return logits

@ModelRegistry.register("multimodal_resnet3d_v2")
class MultiModal_ResNet3D_V2(nn.Module):
    """
    ResNet3D-based model for MRI analysis and clinical features using pre-trained weights.
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
        d_clinical: int = 86,
        clinical_dropout: float = 0.1,
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
            nn.AdaptiveAvgPool3d(1), nn.Conv3d(d_model * 3, d_model, 1), nn.GELU(), nn.Conv3d(d_model, d_model * 3, 1), nn.Sigmoid()
        )

        self.position_embeddings = FactorizedPositionalEmbedding3D(self.d_model, 3, 5, 5)
        self.attention = MultiHeadAttention(d_model * 3, 8)

        self.feed_forward = nn.Sequential(
            nn.Linear(d_model * 6, d_model * 12),
            nn.GELU(),
            nn.Linear(d_model * 12, d_model * 6),
            nn.GELU(),
        )
        self.layer_norm = nn.LayerNorm(d_model * 6)

        self.classifier = nn.Sequential(nn.Linear(d_model * 6, d_model * 12), nn.GELU(), nn.Dropout(out_dropout), nn.Linear(d_model * 12, n_classes))

        self.clinlical_mlp = nn.Sequential(
            nn.Linear(d_clinical, d_model * 8),
            nn.GELU(),
            nn.Dropout(clinical_dropout),
            nn.Linear(d_model * 8, d_model * 8),
            nn.GELU(),
            nn.Dropout(clinical_dropout),
            nn.Linear(d_model * 8, d_model * 3),
            nn.GELU(),
        )

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
        clinical_feat = self.clinlical_mlp(x["clinical_features"])

        x = x["images"]
        features = []

        for i in range(3):
            mri = x[:, i : i + 1]  # [B, 1, D, H, W]

            feat = self.forward_features(mri)  # [B, 255, 3, 5, 5] for half size

            feat = self.mri_adapters[i](feat)  # [B, d_model, D, H, W]
            features.append(feat)

        V = torch.cat(features, dim=1)  # [B, 3 * d_model, D, H, W]
        B, C, D, H, W = V.shape  # C = 3 * d_model

        attn = self.spatial_attention(V)  # [B, 3 * d_model, 1, 1, 1]
        V = V * attn  # [B,  3 * d_model, D, H, W]
        x_avg = F.adaptive_avg_pool3d(V, 1).squeeze(-1).squeeze(-1).squeeze(-1)

        V = V.view(B, 3 * self.d_model, -1).transpose(1, 2)

        V = self.position_embeddings.add_to_input(V, D, H, W)

        clinical_k = clinical_feat.unsqueeze(1)  # (B, 1, C_cat)
        clinical_v = clinical_feat.unsqueeze(1)  # (B, 1, C_cat)
        attn_output, _ = self.attention(V, clinical_k, clinical_v)  # [B, D * H * W, 3 * d_model]

        attn_output = attn_output.transpose(1, 2).view(B, C, D, H, W)
        x_avg = F.adaptive_avg_pool3d(attn_output, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 3 * d_model]
        x_max = F.adaptive_max_pool3d(attn_output, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, 3 * d_model]

        x_prev = torch.cat([x_avg, x_max], dim=1)
        x = self.feed_forward(x_prev)  # [B, 6 * d_model]

        x = self.layer_norm(x + x_prev)

        logits = self.classifier(x)

        return logits
