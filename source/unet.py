import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, base_channels = 32):

        super().__init__()

        self.conv_in = nn.Conv2d(in_channels, base_channels, kernel_size=1)

        self.down1 = DownStep(base_channels, base_channels * 2)
        self.down2 = DownStep(base_channels * 2, base_channels * 4)
        self.down3 = DownStep(base_channels * 4, base_channels * 8)
        self.down4 = DownStep(base_channels * 8, base_channels * 16)

        self.up1 = UpStep(base_channels * 16, base_channels * 8)
        self.up2 = UpStep(base_channels * 8, base_channels * 4)
        self.up3 = UpStep(base_channels * 4, base_channels * 2)
        self.up4 = UpStep(base_channels * 2, base_channels)

        self.conv_out = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):

        x1 = self.conv_in(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.down4(x4)

        x = self.up1(x, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.conv_out(x)

        return x


class DownStep(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()

        self.conv = ConvBlock(in_channels, out_channels)
        self.pool = PoolBlock(out_channels)

    def forward(self, x):

        x = self.conv(x)
        x = self.pool(x)

        return x


class UpStep(nn.Module):

    def __init__(self, in_channels: int, out_channels: int = None):

        super().__init__()

        out_channels = in_channels // 2  # standard UNet configuration

        self.up = UnpoolBlock(in_channels)
        self.conv = ConvBlock(int(1.5*in_channels), out_channels) # recurrent tensor is concatenated, so that the number of in_channels is increased 1.5-fold

    def forward(self, current_x, recurrent_x):

        current_x = self.up(current_x)

        # current_x max be smaller than recurrent_x; therefore current_x must be padded before concatenation
        difference_dim2 = recurrent_x.size()[2] - current_x.size()[2]  # difference in height
        difference_dim3 = recurrent_x.size()[3] - current_x.size()[3]  # difference in width
        current_x = F.pad(current_x, [
            difference_dim3 // 2, difference_dim3 - (difference_dim3 // 2),  # left side and right side
            difference_dim2 // 2, difference_dim2 - (difference_dim2 // 2)  # top and bottom
        ])
        x = torch.cat([recurrent_x, current_x], dim=1)
        
        x = self.conv(x)

        return x


class ConvBlock(nn.Module):

    def __init__(self, in_channels: int, out_channels: int):

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.LeakyReLU(),

            # Select one
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

            # Select one
            # nn.BatchNorm2d(n_channels)
            # nn.InstanceNorm2d(out_channels)
            nn.Identity()
        )

    def forward(self, x):
        
        return self.layers(x)


class UnpoolBlock(nn.Module):

    def __init__(self, n_channels):

        super().__init__()
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(n_channels, n_channels, kernel_size=2, stride=2),
            nn.LeakyReLU(),

            # Select one
            # nn.BatchNorm2d(n_channels)
            # nn.InstanceNorm2d(out_channels)
            nn.Identity()
        )

    def forward(self, x):
        
        return self.layers(x)