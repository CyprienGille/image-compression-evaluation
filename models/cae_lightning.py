import torch
import torch.nn as nn


class EncoderLayer(torch.nn.Module):
    def __init__(self):

        super().__init__()

        # 64x64x64
        self.conv_1 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=3, out_channels=64, kernel_size=(5, 5), stride=(2, 2)
            ),
            nn.LeakyReLU(),
        )

        # 128x32x32
        self.conv_2 = nn.Sequential(
            nn.ZeroPad2d((1, 2, 1, 2)),
            nn.Conv2d(
                in_channels=64, out_channels=128, kernel_size=(5, 5), stride=(2, 2)
            ),
            nn.LeakyReLU(),
        )

        # 128x32x32
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 128x32x32
        self.block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 128x32x32
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 32x32x32
        self.conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=128,
                out_channels=32,
                kernel_size=(5, 5),
                stride=(1, 1),
                padding=(2, 2),
            ),
            nn.Tanh(),
        )

    def forward(self, x):

        ec1 = self.conv_1(x)
        ec2 = self.conv_2(ec1)
        eblock1 = self.block_1(ec2) + ec2
        eblock2 = self.block_2(eblock1) + eblock1
        eblock3 = self.block_3(eblock2) + eblock2
        encoded = self.conv_3(eblock3)  # in [-1, 1] from tanh activation

        return encoded


class DecoderLayer(torch.nn.Module):
    def __init__(self):

        super().__init__()

        # 128x64x64
        self.up_conv_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=64, out_channels=128, kernel_size=(2, 2), stride=(2, 2)
            ),
        )

        # 128x64x64
        self.block_1 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 128x64x64
        self.block_2 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 128x64x64
        self.block_3 = nn.Sequential(
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.Conv2d(
                in_channels=128, out_channels=128, kernel_size=(3, 3), stride=(1, 1)
            ),
        )

        # 256x128x128
        self.up_conv_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=128, out_channels=32, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ZeroPad2d((1, 1, 1, 1)),
            nn.ConvTranspose2d(
                in_channels=32, out_channels=256, kernel_size=(2, 2), stride=(2, 2)
            ),
        )

        # 3x128x128
        self.up_conv_3 = nn.Sequential(
            nn.Conv2d(
                in_channels=256, out_channels=16, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.LeakyReLU(),
            nn.ReflectionPad2d((2, 2, 2, 2)),
            nn.Conv2d(
                in_channels=16, out_channels=3, kernel_size=(3, 3), stride=(1, 1)
            ),
            nn.Tanh(),
        )

    def forward(self, x):

        uc1 = self.up_conv_1(x)
        dblock1 = self.block_1(uc1) + uc1
        dblock2 = self.block_2(dblock1) + dblock1
        dblock3 = self.block_3(dblock2) + dblock2
        uc2 = self.up_conv_2(dblock3)
        dec = self.up_conv_3(uc2)

        return dec


class LightningCAE(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = EncoderLayer()
        self.decoder = DecoderLayer()

    def forward(self, x):

        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return (encoded, decoded)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, encoded):
        return self.decoder(encoded)

