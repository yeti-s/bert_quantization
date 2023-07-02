import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, groups=1):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=kernel_size // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = self.stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1))
        layers.extend([
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV3(nn.Module):
    def __init__(self, max_length, word_dimension, input_size=224, width_mult=1.0):
        super(MobileNetV3, self).__init__()
        # Setting of inverted residual blocks
        self.cfgs = [
            # t, c, n, s
            [1, 16, 1, 1],
            [4, 24, 2, 2],
            [4, 24, 3, 1],
            [4, 40, 3, 2],
            [4, 40, 3, 1],
            [4, 80, 2, 2],
            [4, 80, 3, 1],
            [4, 80, 3, 1],
            [4, 112, 3, 1],
            [4, 112, 3, 1],
            [4, 160, 1, 2],
            [4, 160, 2, 1],
            [4, 160, 2, 1],
            [4, 320, 1, 1],
        ]

        self.max_length = max_length
        self.word_dimension = word_dimension

        input_channel = int(16 * width_mult) if width_mult > 1.0 else 16
        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        num_stride = 1
        self.features = [ConvBNReLU(1, input_channel, kernel_size=3, stride=2)]
        for t, c, n, s in self.cfgs:
            if s > 1:
                num_stride += 1
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                self.features.append(InvertedResidual(input_channel, output_channel, stride, expand_ratio=t))
                print(f'in {input_channel} out {output_channel} stride {stride}')
                input_channel = output_channel

        self.features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        self.features = nn.Sequential(*self.features)

        self.classifier = nn.Sequential(
            # nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(self.last_channel, max_length, kernel_size=1, bias=True),
        )

        self.fc = nn.Linear(int(max_length * word_dimension / (2 ** (num_stride * 2))), word_dimension)



    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        x = x.view(x.size(0), self.max_length, -1)
        x = self.fc(x)
        x = x.view(x.size(0), self.max_length, self.word_dimension)

        return x
