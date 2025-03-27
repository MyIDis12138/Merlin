"""
3D ResNet implementation following MedicalNet architecture.

This module implements 3D ResNet architectures compatible with MedicalNet
pretrained weights (https://github.com/Tencent/MedicalNet).

References:
- Chen, S., Ma, K., & Zheng, Y. (2019). Med3D: Transfer learning for 3D medical
  image analysis. arXiv preprint arXiv:1904.00625.
- He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image
  recognition. In Proceedings of the IEEE conference on computer vision and
  pattern recognition (pp. 770-778).
"""

import logging

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def conv3x3x3(in_planes: int, out_planes: int, stride: int | tuple = 1) -> nn.Conv3d:
    """3x3x3 convolution with padding"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1x1(in_planes: int, out_planes: int, stride: int | tuple = 1) -> nn.Conv3d:
    """1x1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(self, inplanes: int, planes: int, stride: int | tuple = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion: int = 4

    def __init__(self, inplanes: int, planes: int, stride: int | tuple = 1, downsample: nn.Module | None = None) -> None:
        super().__init__()
        self.conv1 = conv1x1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet3D(nn.Module):
    def __init__(
        self,
        block: type[BasicBlock | Bottleneck],
        layers: list[int],
        input_channels: int = 1,
        num_classes: int = 1000,
        zero_init_residual: bool = False,
    ) -> None:
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv3d(input_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block: type[BasicBlock | Bottleneck], planes: int, blocks: int, stride: int | tuple = 1) -> nn.Sequential:
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))

        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


def _resnet3d(block: type[BasicBlock | Bottleneck], layers: list[int], input_channels: int = 1, **kwargs) -> ResNet3D:
    model = ResNet3D(block, layers, input_channels=input_channels, **kwargs)
    return model


def resnet3d10(input_channels: int = 1, **kwargs) -> ResNet3D:
    """ResNet-10 3D model"""
    return _resnet3d(BasicBlock, [1, 1, 1, 1], input_channels, **kwargs)


def resnet3d18(input_channels: int = 1, **kwargs) -> ResNet3D:
    """ResNet-18 3D model"""
    return _resnet3d(BasicBlock, [2, 2, 2, 2], input_channels, **kwargs)


def resnet3d34(input_channels: int = 1, **kwargs) -> ResNet3D:
    """ResNet-34 3D model"""
    return _resnet3d(BasicBlock, [3, 4, 6, 3], input_channels, **kwargs)


def resnet3d50(input_channels: int = 1, **kwargs) -> ResNet3D:
    """ResNet-50 3D model"""
    return _resnet3d(Bottleneck, [3, 4, 6, 3], input_channels, **kwargs)


def resnet3d101(input_channels: int = 1, **kwargs) -> ResNet3D:
    """ResNet-101 3D model"""
    return _resnet3d(Bottleneck, [3, 4, 23, 3], input_channels, **kwargs)


def resnet3d152(input_channels: int = 1, **kwargs) -> ResNet3D:
    """ResNet-152 3D model"""
    return _resnet3d(Bottleneck, [3, 8, 36, 3], input_channels, **kwargs)
