import torch
import torch.nn as nn

# Define the doubleconv unit 
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, Norm=True):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        if Norm:
            self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.InstanceNorm2d(out_channels),
                nn.ReLU(inplace=True))
        else: 
             self.double_conv = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(inplace=True))
        
    def forward(self, x):
        return self.double_conv(x)

# Upsampleing (Decoder block)
class Up(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels, Norm=True):
        super(Up, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.double_conv = DoubleConv(middle_channels, out_channels, Norm=Norm)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.double_conv(x1)
        return x1

# Downsampling (Encoder block)
class Down(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, Norm=True):
        super().__init__()
        self.double_conv = DoubleConv(in_channels, out_channels, Norm=Norm)
        self.max_pool = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.max_pool(x)
        out = self.double_conv(x)
        return out

# Structure of LTRUnet 
class LTRUnet(nn.Module):
    def __init__(self, in_layer, out_layer):
        super().__init__()

        self.down1 = nn.Sequential(
            nn.Conv2d(in_layer, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),) # in_layer * 224 * 224 -> 32 * 224 * 224
        self.down2 = Down(32, 64, Norm=True)  # 32 * 224 * 224 -> 64 * 112 * 112
        self.down3 = Down(64, 128, Norm=True) # 64 * 112 * 112 -> 128 * 56 * 56
        self.down4 = Down(128, 256, Norm=True) # 128 * 56 * 56 -> 256 * 28 * 28
        self.down5 = Down(256, 512, Norm=True) # 256 * 28 * 28 -> 512 * 14 *14
        self.up4 = Up(512, 256+256, 256, Norm=True) # 512 * 14 *14 -> 256 * 28 * 28
        self.up3 = Up(256, 128+128, 128, Norm=True) # 256 * 28 * 28 -> 128 * 56 * 56
        self.up2 = Up(128, 64+64, 64, Norm=True) # 128 * 56 * 56 -> 64 * 112 * 112
        self.up1 = Up(64, 32+32, 32, Norm=False) # 64 * 112 * 112 -> 32 * 2224 *224

        self.conv_last = nn.Conv2d(32, out_layer, kernel_size=3, padding=1)

    def forward(self, x):
        e1 = self.down1(x) 
        e2 = self.down2(e1) 
        e3 = self.down3(e2) 
        e4 = self.down4(e3) 

        f = self.down5(e4) 

        d4 = self.up4(f, e4)
        d3 = self.up3(d4, e3)
        d2 = self.up2(d3, e2) 
        d1 = self.up1(d2, e1) 
        out = self.conv_last(d1) 
        return out

