import torch
import torch.nn as nn

def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


class SmallConvNet(nn.Module):
    def __init__(self, channels=3):
        super(SmallConvNet, self).__init__()
        hidden_channels = 16
        self.conv1 = conv3x3(channels,hidden_channels)
        self.conv2 = conv3x3(hidden_channels,hidden_channels)
        self.conv3 = conv3x3(hidden_channels,channels)
    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

class LearnedInput(nn.Module):
    def __init__(self, dimensions):
        super(LearnedInput, self).__init__()
        input = torch.rand(dimensions)*0.1
        self.learned_input = nn.parameter.Parameter(data=input)
    def forward(self,x):
        return self.learned_input
