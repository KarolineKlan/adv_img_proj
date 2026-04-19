import torch
import torch.nn as nn
import torch.nn.functional as f


class UnetModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = 64
        self.in_channels = 1

        # Contracting path
        self.con1 = self.contracting_block(self.in_channels, self.features)
        self.con2 = self.contracting_block(self.features, 2 * self.features)
        self.con3 = self.contracting_block(2 * self.features, 4 * self.features)
        self.con4 = self.contracting_block(4 * self.features, 8 * self.features)

        # bottom layer
        self.bot = self.bottom_layer(8 * self.features)

        # Expanding path
        self.exp1 = self.expanding_block(16 * self.features)
        self.exp2 = self.expanding_block(8 * self.features)
        self.exp3 = self.expanding_block(4 * self.features)

        # final layer
        self.final = self.final_layer(2 * self.features, 2)

    def forward(self, x):
        # assuming input images are 64 x 64 x 1
        """ Preforms forward propagation for the neural network """
        x = self.con1(x)
        x = self.con2(x)
        x = self.con3(x)
        x = self.con4(x)
        x = self.bot(x)
        x = self.exp1(x)
        x = self.exp2(x)
        x = self.exp3(x)
        x = self.final(x)
        # x = f.softmax(x, dim=1)
        # We use BCEWithLogitsLoss, that takes the raw network output (no activation or softmax needed)

        return x

    def contracting_block(self, channels_in, channels_out=None):
        """ Creates a contracting block sequence defined by:
            conv (3x3) -> conv (3x3) -> max pool (2x2)
        """
        # if we contract then the number of out_channels double
        channels_out = int(2 * channels_in) if channels_out is None else channels_out

        # create a sequence of layers that can be re-used
        x = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        return x

    def expanding_block(self, channels_in, channels_out=None):
        """ Creates an expanding block sequence defined by:
            conv (3x3) -> conv (3x3) -> up conv (3x3)
        """
        # if we expand the number of out_channels half
        channels_out = channels_in // 2 if channels_out is None else channels_out

        x = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=3, stride=2, padding=1, output_padding=1)
        )

        return x

    def bottom_layer(self, channels_in, channels_out=None):
        """ Creates the bottleneck (layer) of the UNet structure, defined by:
            conv (3x3) -> conv (3x3) -> up conv (3x3)
        """
        # double the amount of channels
        channels_out = int(2 * channels_in) if channels_out is None else channels_out

        x = nn.Sequential(
            nn.Conv2d(channels_in, channels_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(),
            nn.ConvTranspose2d(channels_out, channels_out, kernel_size=3, stride=2, padding=1, output_padding=1),
        )
        return x

    def final_layer(self, channels_in, channels_out):
        """ Creates the final block structure, defined by:
            conv (3x3) -> conv (3x3) -> conv (1x1)
        """
        channels_mid = channels_in // 2

        x = nn.Sequential(
            nn.Conv2d(channels_in, channels_mid, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_mid),
            nn.ReLU(),
            nn.Conv2d(channels_mid, channels_mid, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels_mid),
            nn.ReLU(),
            nn.Conv2d(channels_mid, channels_out, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(channels_out)
        )
        return x

if __name__ == "__main__":
    model = UnetModel()
    x = torch.rand(1)
    print(f"Output shape of model: {model(x).shape}")
