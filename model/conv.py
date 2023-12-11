import math
import mlx.core as mx
import mlx.nn as nn
from typing import Union
from model import BatchNorm2d

class Convolution(nn.Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple],
        stride: Union[int, tuple] = 1,
        padding: Union[int, tuple] = 0,
        bias: bool = True,
    ):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, bias)

        kernel_size, stride, padding = map(
            lambda x: (x, x) if isinstance(x, int) else x,
            (kernel_size, stride, padding),
        )

        scale = math.sqrt(2. / (in_channels * kernel_size[0] * kernel_size[1]))
        self.weight = mx.random.uniform(
            low=-scale,
            high=scale,
            shape=(out_channels, *kernel_size, in_channels),
        )

        if bias:
            self.bias = mx.zeros((out_channels,))

        self.padding = padding
        self.stride = stride

class ConvBN(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, kernel_size: int = 3, 
            stride: int = 1, padding: int = 1, groups: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding, bias = False)
        self.bn =   nn.GroupNorm(groups, output_channel)
        self.relu = nn.ReLU()

    def __call__(self, x):
        return self.relu(self.bn(self.conv(x)))
    
class ConvDW(nn.Module):
    def __init__(self, input_channel: int, output_channel: int, kernel_size: int = 3, 
            stride: int = 1, padding: int = 1, groups: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channel, input_channel, kernel_size, stride, padding, bias = False)
        self.bn1   = nn.GroupNorm(groups, input_channel)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(input_channel, output_channel, 1, 1, 0, bias = False)
        self.bn2   = nn.GroupNorm(groups, output_channel)
        self.relu2 = nn.ReLU()
        
    def __call__(self,x):
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        return x