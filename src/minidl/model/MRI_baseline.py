import torch
import torch.nn as nn
import torch.nn.functional as F

from minidl.layers.attention_layers import MultiHeadAttention


class Conv3dBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation):
        super().__init__()

        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size, stride, padding="same")
        self.activation = activation

        self.pool = nn.MaxPool3d(kernel_size=kernel_size, stride=2)
        self.bn = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.activation(x)
        x = self.pool(x)
        x = self.bn(x)

        return x


class MRI_baseline(torch.nn.Module):
    def __init__(self, n_classes, d_model):
        super().__init__()

        assert d_model % 8 == 0, "d_model must be divisible by 8"
        self.d_model = d_model

        self.mri_extrator = nn.ModuleList()

        for _ in range(5):
            conv1 = Conv3dBlock(1, d_model // 4, (32, 32, 16), 1, nn.ReLU())
            conv2 = Conv3dBlock(d_model // 4, d_model // 2, (16, 16, 8), 1, nn.ReLU())
            conv3 = Conv3dBlock(d_model // 2, d_model, (8, 8, 4), 1, nn.ReLU())

            convs = nn.Sequential(conv1, conv2, conv3)
            self.mri_extrator.append(convs)

        self.attention = MultiHeadAttention(d_model, 8)

        self.fc1 = nn.Linear(d_model, d_model * 2)
        self.fc2 = nn.Linear(d_model * 2, d_model * 2)
        self.fc3 = nn.Linear(d_model * 2, n_classes)

    def forward(self, x):
        B = x.shape[0]
        x = [conv(x[:, i, :].unsqueeze(1)) for i, conv in enumerate(self.mri_extrator)]
        x = torch.cat(x, dim=1).reshape(B, -1, self.d_model)

        x, _ = self.attention(x)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


if __name__ == "__main__":
    model = MRI_baseline(4, 32).to("cuda")
    x = torch.randn(4, 5, 158, 512, 512).to("cuda")
    y = model(x)
    print(y.shape)
