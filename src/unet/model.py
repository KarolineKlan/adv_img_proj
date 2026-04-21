import torch
import torch.nn as nn


class UnetModel(nn.Module):
    """U-Net for EM membrane segmentation.

    Input:  (B, 1, 256, 256)  — grayscale patches
    Output: (B, 2, 256, 256)  — logits for [background, membrane]

    Use with CrossEntropyLoss (no softmax needed in forward).
    """

    def __init__(self):
        super().__init__()
        f = 64  # base number of features

        # Contracting path (encoder)
        self.enc1 = self.conv_block(1,      f)      # (B,  64, 256, 256) -> pool -> (B,  64, 128, 128)
        self.enc2 = self.conv_block(f,    2*f)      # (B, 128, 128, 128) -> pool -> (B, 128,  64,  64)
        self.enc3 = self.conv_block(2*f,  4*f)      # (B, 256,  64,  64) -> pool -> (B, 256,  32,  32)
        self.enc4 = self.conv_block(4*f,  8*f)      # (B, 512,  32,  32) -> pool -> (B, 512,  16,  16)
        self.pool = nn.MaxPool2d(2, 2)

        # Bottleneck
        self.bottleneck = self.conv_block(8*f, 16*f)  # (B, 1024, 16, 16)

        # Expanding path (decoder) — each step: upsample, cat skip, conv_block
        self.up4    = nn.ConvTranspose2d(16*f, 8*f, kernel_size=2, stride=2)  # -> (B, 512, 32, 32)
        self.dec4   = self.conv_block(16*f, 8*f)   # 16*f because skip is concatenated

        self.up3    = nn.ConvTranspose2d(8*f, 4*f, kernel_size=2, stride=2)   # -> (B, 256, 64, 64)
        self.dec3   = self.conv_block(8*f,  4*f)

        self.up2    = nn.ConvTranspose2d(4*f, 2*f, kernel_size=2, stride=2)   # -> (B, 128, 128, 128)
        self.dec2   = self.conv_block(4*f,  2*f)

        self.up1    = nn.ConvTranspose2d(2*f, f,   kernel_size=2, stride=2)   # -> (B, 64, 256, 256)
        self.dec1   = self.conv_block(2*f,  f)

        # Final 1x1 conv to get class logits
        self.final  = nn.Conv2d(f, 2, kernel_size=1)

    def conv_block(self, channels_in, channels_out):
        """Two 3x3 convolutions with BatchNorm and ReLU. padding=1 keeps spatial size."""
        return nn.Sequential(
            nn.Conv2d(channels_in,  channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels_out, channels_out, kernel_size=3, padding=1),
            nn.BatchNorm2d(channels_out),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder — save outputs for skip connections
        s1 = self.enc1(x)           # (B,  64, 256, 256)
        s2 = self.enc2(self.pool(s1))  # (B, 128, 128, 128)
        s3 = self.enc3(self.pool(s2))  # (B, 256,  64,  64)
        s4 = self.enc4(self.pool(s3))  # (B, 512,  32,  32)

        # Bottleneck
        x = self.bottleneck(self.pool(s4))  # (B, 1024, 16, 16)

        # Decoder — upsample, concatenate skip, convolve
        x = self.dec4(torch.cat([self.up4(x), s4], dim=1))  # (B, 512, 32, 32)
        x = self.dec3(torch.cat([self.up3(x), s3], dim=1))  # (B, 256, 64, 64)
        x = self.dec2(torch.cat([self.up2(x), s2], dim=1))  # (B, 128, 128, 128)
        x = self.dec1(torch.cat([self.up1(x), s1], dim=1))  # (B,  64, 256, 256)

        return self.final(x)  # (B, 2, 256, 256)


if __name__ == "__main__":
    model = UnetModel()

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {n_params:,}")

    # Check output shape matches input shape
    x = torch.rand(2, 1, 256, 256)  # batch of 2 grayscale patches
    out = model(x)
    print(f"Input shape:  {x.shape}")
    print(f"Output shape: {out.shape}  (expect [2, 2, 256, 256])")
    assert out.shape == torch.Size([2, 2, 256, 256]), "Shape mismatch!"
    print("Shape check passed.")