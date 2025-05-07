import torch
from torch import nn
import torch.nn.functional as F

from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, n_embed: int):
        super().__init__()
        self.linear_1 = nn.Linear(n_embed, 4*n_embed)
        self.linear_2 = nn.Linear(4*n_embed, 4*n_embed)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (1, 320)
        """

        x = self.linear_1(input)

        x = F.silu(x)

        x = self.linear_2(x)

        # (1, 1280)
        return x

class SwitchSequential(nn.Sequential):

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                latent = layer(latent, context)
            elif isinstance(layer, UNET_ResidualBlock):
                latent = layer(latent, time)
            else:
                latent = layer(latent)
        return latent

class UNET(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.ModuleList([
            # (Batch size, 4, ~Height/8, ~Width/8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),

            # (Batch size, 320, ~Height/8, ~Width/8) -> (Batch size, 320, ~Height/16, ~Width/16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),

            # (Batch size, 640, ~Height/16, ~Width/16) -> (Batch size, 640, ~Height/32, ~Width/32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),

            # (Batch size, 1280, ~Height/32, ~Width/32) -> (Batch size, 1280, ~Height/64, ~Width/64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280)),

            SwitchSequential(UNET_ResidualBlock(1280, 1280))
        ])

        self.bottleneck = SwitchSequential(
            UNET_ResidualBlock(1280, 1280),

            UNET_AttentionBlock(8, 160),

            UNET_ResidualBlock(1280, 1280),
        )

        self.decoder = nn.ModuleList([
            # (Batch size, 2560, ~Height/64, ~Width/64) -> (Batch size, 1280, ~Height/64, ~Width/64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),

            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),

            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),

            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

class Diffusion(nn.Module):
    def __init__(self):
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.output_layer = UNET_OutputLayer(320, 4)

    def forward(self, latent: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        """
        latent: (Batch size, 4, ~Height/8, ~Width/8)
        context: (Batch size, Seq len, d_model)
        time: (1, 320)
        """

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)

        # (Batch size, 4, ~Height/8, ~Width/8) -> (Batch size, 320, ~Height/8, ~Width/8)
        output = self.unet(latent, context, time)

        # (Batch size, 320, ~Height/8, ~Width/8) -> (Batch size, 4, ~Height/8, ~Width/8)
        output = self.output_layer(output)

        return output
