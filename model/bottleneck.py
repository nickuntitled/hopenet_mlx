import mlx.nn as nn

def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    # Based on PyTorch
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample = None,
                 base_width=64, dilation=1, groups=16):
        super(Bottleneck, self).__init__()

        width = int(planes * (base_width / 64.))
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.GroupNorm(width if groups is None else groups, width)
        self.conv2 = conv3x3(width, width, stride, dilation)
        self.bn2 = nn.GroupNorm(width if groups is None else groups, width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.GroupNorm(planes * self.expansion if groups is None else groups, planes * self.expansion)
        self.relu = nn.ReLU()
        self.stride = stride

        if downsample is not None:
            self.downsample_conv = downsample['conv']
            self.downsample_bn   = downsample['bn']
        else:
            self.downsample_conv = None
            self.downsample_bn   = None

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample_conv is not None:
            identity = self.downsample_bn(self.downsample_conv(identity))

        out += identity
        out = self.relu(out)

        return out
    
class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample = None,
        groups: int = 1
    ) -> None:
        super().__init__()

        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.GroupNorm(planes if groups is None else groups, planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.GroupNorm(planes if groups is None else groups, planes)
        self.stride = stride

        if downsample is not None:
            self.downsample_conv = downsample['conv']
            self.downsample_bn   = downsample['bn']
        else:
            self.downsample_conv = None
            self.downsample_bn   = None

    def __call__(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample_bn(self.downsample_conv(identity))

        out += identity
        out = self.relu(out)

        return out