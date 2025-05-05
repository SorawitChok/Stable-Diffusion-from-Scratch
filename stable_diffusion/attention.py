import torch
from torch import nn
import torch.nn.functional as F

import math

class SelfAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()

        self.in_proj = nn.Linear(d_model, 3*d_model, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        
    def forward(self, input: torch.Tensor, casual_mask=False):
        """
        input: (Batch size, Seq len, d_model)
        """

        input_shape = input.shape
        batch_size, seq_len, d_model = input_shape

        intermediate_shape = (batch_size, seq_len, self.n_heads, self.d_head)

        # (Batch size, Seq len, d_model) -> (Batch size, Seq len, d_model*3) -> three tensor of shape (Batch size, Seq len, d_model)
        q, k, v = self.in_proj(input).chunk(3, d_model=-1)

        # Dk =  d_model/n_heads
        # (Batch size, Seq len, d_model) -> (Batch size, Seq len, n_heads, Dk) -> (Batch size, n_heads, Seq len, Dk)
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        # (Batch size, n_heads, Seq len, Seq len)
        weight = q @ k.transpose(-1,-2)

        if casual_mask:
            # Mask where the upper triangle (above principal diagonal) is made up of 1 
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_fill_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, d_model=-1)
        
        # (Batch size, n_heads, Seq len, Seq len) @ (Batch size, n_heads, Seq len, Dk) -> (Batch size, n_heads, Seq len, Dk)
        output = weight @ v

        # (Batch size, n_heads, Seq len, Dk) -> (Batch size, Seq len, n_heads, Dk)
        output = output.transpose(1, 2)

        # (Batch size, Seq len, d_model)
        output = output.reshape(input_shape)

        output = self.out_proj(output)
        
        return output

