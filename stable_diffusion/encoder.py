import torch
from torch import nn
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
        
        # (Batch size, 512, ~Height/4, ~Width/4) -> (Batch size, 512, ~Height/8, ~Width/8)
        nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0),
        
        # (Batch_size, 512, ~Height/8, ~Width/8) --> (Batch_size, 512, ~Height/8, ~Width/8)
        VAE_ResidualBlock(512, 512),
        
        # (Batch_size, 512, ~Height/8, ~Width/8) --> (Batch_size, 512, ~Height/8, ~Width/8)
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

    def forward(self, input: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """ 
        input: (Batcvh size, Channel, Height, Width)
        noise: (Batch size, Out channel, ~Height/8, ~ Width/8)
        """

        for module in self:
            if getattr(module, 'stride', None) == (2,2):
                # (Padding Left, Padding Right, Padding Top, Padding Bottom)
                input = F.pad(input,(0, 1, 0, 1))
            input = module(input)
        
        # (Batch size, 8, ~Height/8, ~Width/8) -> two tensor of shape (Batch size, 4, ~Height/8, ~Width/8)
        mean, log_variance = torch.chunk(input, 2, dims=1)

        # (Batch size, 4, ~Height/8, ~Width/8) -> (Batch size, 4, ~Height/8, ~Width/8)
        log_variance = torch.clamp(log_variance, -30, 20)
        
        # (Batch size, 4, ~Height/8, ~Width/8) -> (Batch size, 4, ~Height/8, ~Width/8)
        variance = log_variance.exp()

        # (Batch size, 4, ~Height/8, ~Width/8) -> (Batch size, 4, ~Height/8, ~Width/8)
        std = variance.sqrt()

        # Z = N(0, 1) [standard normal distribution with mean = 0 and variance = 1] -> X = N(mean, variance)
        # X = mean + std * Z
        x = mean + std * noise

        # Scale the output by a constant
        x *= 0.18215

        return x

