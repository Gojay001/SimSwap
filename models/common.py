import torch
import torch.nn as nn


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):
        """
            @notice: avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3
        """
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x) # or x ** 2
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp

#-------------------------------------------

class DepthConv(nn.Module):
    """
    DepthwiseConv + Conv2
    if use BN: Depthwise(Conv + BN), Pointwise(Conv + BN + ReLU)
    if use IN: Depthwise(Conv), Pointwise(Conv + IN)

    in_channels:   input channel
    out_channels:  output channel
    kernel_size:   depthwise kernel size
    stride:        conv1 stride
    padding:       depthwise padding
    dilation:      depthwise dilation
    norm_layer:    normalization layer
    activation:    activation function
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, **args):
        super(DepthConv, self).__init__()

        self.norm_layer = args.get('norm_layer', nn.BatchNorm2d)
        self.activation = args.get('activation', nn.ReLU(True))

        depthwise = [nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride, groups=in_channels, padding=padding, dilation=dilation)]
        if self.norm_layer == nn.BatchNorm2d:
            depthwise.append(self.norm_layer(in_channels))
        self.depthwise = nn.Sequential(*depthwise)

        pointwise = [nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
                     self.norm_layer(out_channels)]
        if self.norm_layer == nn.BatchNorm2d:
            pointwise.append(self.activation)
        self.pointwise = nn.Sequential(*pointwise)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x


class ConvDepthConv(nn.Module):
    """
    Conv1 + DepthwiseConv + Conv2

    in_channels:   input channel
    mid_channels:  mid channel
    out_channels:  output channel
    kernel_size:   depthwise kernel size
    stride:        conv1 stride
    padding:       depthwise padding
    dilation:      depthwise dilation
    norm_layer:    normalization layer
    activation:    activation function
    """
    def __init__(self, in_channels, mid_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, **args):

        super(ConvDepthConv, self).__init__()

        self.norm_layer = args.get('norm_layer', nn.BatchNorm2d)
        self.activation = args.get('activation', nn.ReLU(True))

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=1, stride=stride, padding=0),
            self.norm_layer(mid_channels),
            self.activation)

        self.depthwise = nn.Sequential(
            nn.Conv2d(mid_channels, mid_channels, kernel_size=kernel_size, stride=1, groups=mid_channels, padding=padding, dilation=dilation),
            self.norm_layer(mid_channels))

        self.conv2 = nn.Sequential(
            nn.Conv2d(mid_channels, out_channels, kernel_size=1, stride=1, padding=0),
            self.norm_layer(out_channels),
            self.activation)

    def forward(self, x):
        out = self.conv1(x)
        out = self.depthwise(out)
        out = self.conv2(out)
        return out
