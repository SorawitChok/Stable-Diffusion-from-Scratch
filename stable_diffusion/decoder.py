import torch
from torch import nn
import torch.nn.functional as F

from attention import SelfAttention

class VAE_AttentionBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, channels)
        self.attention = SelfAttention(1, channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (Batch size, channels, Height, Width)
        """
        residue = input

        n, c, h, w = input.shape

        # (Batch size, channels, Height, Width) -> (Batch size, channels, Height*Width)
        x = input.view(n, c, h*w)

        # (Batch size, channels, Height*Width) -> (Batch size, Height*Width, channels)
        x = x.transpose(-1, -2)

        # (Batch size, Height*Width, channels) -> (Batch size, Height*Width, channels)
        x = self.attention(x)

        # (Batch size, Height*Width, channels) -> (Batch size, channels, Height*Width) 
        x = x.transpose(-1, -2)

        # (Batch size, channels, Height*Width) -> (Batch size, channels, Height, Width)
        x = x.view(n, c, h, w)

        return x + residue

class VAE_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm_1 = nn.GroupNorm(32, in_channels)
        self.conv_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.groupnorm_2 = nn.GroupNorm(32, out_channels)
        self.conv_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else: 
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (Batch size, In Channel, Height, Width)
        """
        residue = input

        x = self.groupnorm_1(input)
        x = F.silu(x)
        x = self.conv_1(x)

        x = self.groupnorm_2(x)
        x = F.silu(x)
        x = self.conv_2(x)

        return x + self.residual_layer(residue)

