import torch
import torch.nn as nn

def conv_bn(in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False),
        nn.BatchNorm2d(out_channels),
    )


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class SEBlock(nn.Module):
    def __init__(self, in_channels, squeeze_ratio=0.25):
        super(SEBlock, self).__init__()
        squeeze_channels = max(1, int(in_channels * squeeze_ratio))
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, squeeze_channels, kernel_size=1, stride=1, bias=True)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(squeeze_channels, in_channels, kernel_size=1, stride=1, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        return x * out


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand_ratio, stride, kernel_size, dropout_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        self.use_residual = self.stride == 1 and in_channels == out_channels
        hidden_channels = in_channels * expand_ratio
        self.expand = in_channels != hidden_channels
        self.squeeze_excitation = SEBlock(hidden_channels)

        layers = []
        if self.expand:
            layers.append(conv_bn(in_channels, hidden_channels, kernel_size=1))
            layers.append(nn.ReLU(inplace=True))

        layers.extend([
            conv_bn(hidden_channels, hidden_channels, kernel_size, stride, padding=kernel_size // 2, groups=hidden_channels),
            nn.ReLU(inplace=True),
            self.squeeze_excitation,
            nn.Conv2d(hidden_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(dropout_rate) if dropout_rate > 0 else nn.Identity(),
        ])
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        out = self.layers(x)
        if self.use_residual:
            out += x
        return out


class EfficientNetLiteForEncoder(nn.Module):
    def __init__(self, max_length, word_dimension, width_multiplier=1.0, dropout_rate=0.2):
        super(EfficientNetLiteForEncoder, self).__init__()
        settings = [
            # t, c, n, s, k
            [1, 24, 2, 1, 3],
            [4, 48, 4, 2, 3],
            [4, 64, 4, 2, 5],
            [4, 128, 6, 2, 3],
            [6, 160, 9, 1, 5],
            [6, 256, 15, 2, 5],
            [6, 512, 15, 2, 5],
            [6, 640, 15, 2, 3],
        ]

        self.max_length = max_length
        self.word_dimension = word_dimension

        input_channels = int(32 * width_multiplier)
        self.stem = nn.Sequential(
            conv_bn(1, input_channels, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        blocks = []
        for t, c, n, s, k in settings:
            output_channels = int(c * width_multiplier)
            for i in range(n):
                stride = s if i == 0 else 1
                blocks.append(MBConvBlock(input_channels, output_channels, expand_ratio=t, stride=stride,
                                          kernel_size=k, dropout_rate=dropout_rate))
                input_channels = output_channels

        self.blocks = nn.Sequential(*blocks)

        self.head = nn.Sequential(
            conv_bn(input_channels, int(1280 * width_multiplier), kernel_size=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1)
        )
        self.fc = nn.Linear(int(1280 * width_multiplier), max_length * word_dimension)

    def forward(self, x):
        batch_size = x.shape[0]
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x.view(batch_size, self.max_length, self.word_dimension)
