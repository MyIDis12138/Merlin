import torch
import torch.nn as nn
import torch.nn.functional as F

from minidl.layers.attention_layers import MultiHeadAttention

from .model_registry import ModelRegistry


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, use_residual=True):
        super().__init__()
        self.use_residual = use_residual and (in_channels == out_channels)

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding="same")
        self.bn = nn.BatchNorm3d(out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv(x)
        out = self.bn(out)

        if self.use_residual:
            out += identity

        out = self.activation(out)
        return out


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = ConvBlock(in_channels, out_channels)
        self.conv2 = ConvBlock(out_channels, out_channels)
        self.pool = nn.MaxPool3d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.pool(x)
        return x


@ModelRegistry.register("mri_baseline")
class MRI_baseline(nn.Module):
    def __init__(self, n_classes, d_model, out_dropout=0.3):
        super().__init__()

        assert d_model % 8 == 0, "d_model must be divisible by 8"
        self.d_model = d_model

        self.shared_extractor = nn.Sequential(
            DownBlock(1, d_model // 8),  # 1 -> d_model/8
            DownBlock(d_model // 8, d_model // 4),  # d_model/8 -> d_model/4
            DownBlock(d_model // 4, d_model // 2),  # d_model/4 -> d_model/2
            DownBlock(d_model // 2, d_model),  # d_model/2 -> d_model
        )

        self.mri_adapters = nn.ModuleList([nn.Sequential(ConvBlock(d_model, d_model), ConvBlock(d_model, d_model)) for _ in range(5)])

        self.spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool3d(1), nn.Conv3d(d_model, d_model // 2, 1), nn.ReLU(inplace=True), nn.Conv3d(d_model // 2, d_model, 1), nn.Sigmoid()
        )

        self.attention = MultiHeadAttention(d_model, 4)

        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.ReLU(inplace=True), nn.Dropout(out_dropout), nn.Linear(d_model // 2, n_classes)
        )

    def forward(self, x):
        # x: tensor of 5 MRI images, shaped [B, 5, D, H, W]
        features = []

        for i in range(5):
            mri = x[:, i, :, :, :].unsqueeze(1)  # [B, 1, D, H, W]

            feat = self.shared_extractor(mri)  # [B, d_model, D', H', W']

            feat = self.mri_adapters[i](feat)  # [B, d_model, D', H', W']

            attn = self.spatial_attention(feat)
            feat = feat * attn

            feat_gap = F.adaptive_avg_pool3d(feat, 1).squeeze(-1).squeeze(-1).squeeze(-1)  # [B, d_model]
            features.append(feat_gap)

        V = torch.stack(features, dim=1)  # [B, 5, d_model]

        attn_output, _ = self.attention(V, V, V)
        mean_output = torch.mean(attn_output, dim=1)  # [B, d_model]

        logits = self.classifier(mean_output)  # [B, n_classes]

        return logits


if __name__ == "__main__":
    model = MRI_baseline(4, 32).to("cuda")
    x = torch.randn(4, 5, 158, 512, 512).to("cuda")
    y = model(x)
    print(y.shape)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
