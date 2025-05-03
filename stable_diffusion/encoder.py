import torch
import torch.nn as nn
import torch.nn.functional as F

from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    super().__init__(
        # (Batch size, Channel, Height, Width) -> (Batch size, 128, Height, Width)
        nn.Conv2d(3, 128, kernel_size=3, padding=1),

        # (Batch size, 128, Height, Width) -> (Batch size, 128, Height, Width)
        VAE_ResidualBlock(128, 128), 

        # (Batch size, 128, Height, Width) -> (Batch size, 128, Height, Width)
        VAE_ResidualBlock(128, 128), 

        # (Batch size, 128, Height, Width) -> (Batch size, 128, ~Height/2, ~Width/2)
        # The exact calculation of ~Height/2 is floor([Height - kernel_size + 2*padding] /stride) + 1
        # The exact calculation of ~Weidth/2 is floor([Width - kernel_size + 2*padding] /stride) + 1
        nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

        # (Batch size, 128, ~Height/2, ~Width/2) -> (Batch size, 256, ~Height/2, ~Width/2)
        VAE_ResidualBlock(128, 256),

        # (Batch size, 256, ~Height/2, ~Width/2) -> (Batch size, 256, ~Height/2, ~Width/2)
        VAE_ResidualBlock(256, 256),

        # (Batch size, 256, ~Height/2, ~Width/2) -> (Batch size, 256, ~Height/4, ~Width/4)
        nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0),

        # (Batch size, 256, ~Height/4, ~Width/4) -> (Batch size, 512, ~Height/4, ~Width/4)
        VAE_ResidualBlock(256, 512),

        # (Batch size, 512, ~Height/4, ~Width/4) -> (Batch size, 512, ~Height/4, ~Width/4)
        VAE_ResidualBlock(512, 512),
        
        # (Batch size, 512, ~Height/8, ~Width/8) -> (Batch size, 512, ~Height/8, ~Width/8)
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),

        VAE_ResidualBlock(512, 512),

        VAE_ResidualBlock(512, 512),

        # (Batch size, 512, ~Height/8, ~Width/8) -> (Batch size, 512, ~Height/8, ~Width/8)
        VAE_ResidualBlock(512, 512),

        # (Batch size, 512, ~Height/8, ~Width/8) -> (Batch size, 512, ~Height/8, ~Width/8)
        VAE_AttentionBlock(512),

        # (Batch size, 512, ~Height/8, ~Width/8) -> (Batch size, 512, ~Height/8, ~Width/8)
        VAE_ResidualBlock(512, 512),

        # (Batch size, 512, ~Height/8, ~Width/8) -> (Batch size, 512, ~Height/8, ~Width/8)
        nn.GroupNorm(32, 512), 

        # (Batch size, 512, ~Height/8, ~Width/8) -> (Batch size, 512, ~Height/8, ~Width/8)
        nn.SiLU(),

        # (Batch size, 512, ~Height/8, ~Width/8) -> (Batch size, 8, ~Height/8, ~Width/8)
        nn.Conv2d(512, 8, kernel_size=3, padding=1),

        # (Batch size, 8, ~Height/8, ~Width/8) -> (Batch size, 8, ~Height/8, ~Width/8)
        nn.Conv2d(8, 8, kernel_size=1, padding=0)
    )