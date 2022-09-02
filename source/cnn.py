import torch.nn as nn


class CNN(nn.Module):

    def __init__(self, in_channels, out_channels, base_channels = 16):

        super().__init__()

        self.layers = nn.Sequential(

            nn.Conv2d(in_channels, base_channels, kernel_size=1),

            DownStep(base_channels, base_channels * 2),
            DownStep(base_channels * 2, base_channels * 4),
            DownStep(base_channels * 4, base_channels * 8),

            nn.Flatten(),

            nn.Linear(8192, out_channels)
        )

    def forward(self, x):

        return self.layers(x)


class DownStep(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = PoolBlock(out_channels)

    def forward(self, x):

        x = self.conv(x)
        x = self.pool(x)

        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),

            # Select one...
            # nn.BatchNorm2d(out_channels)
            # nn.InstanceNorm2d(out_channels)
            nn.Identity()
        )

    def forward(self, x):
        
        return self.layers(x)


class PoolBlock(nn.Module):

    def __init__(self, n_channels):

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            # Select one...
            # nn.BatchNorm2d(out_channels)
            # nn.InstanceNorm2d(out_channels)
            nn.Identity()
        )

    def forward(self, x):
        
        return self.layers(x)