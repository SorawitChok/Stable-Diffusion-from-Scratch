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
        q, k, v = self.in_proj(input).chunk(3, dim=-1)

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
        weight = F.softmax(weight, dim=-1)
        
        # (Batch size, n_heads, Seq len, Seq len) @ (Batch size, n_heads, Seq len, Dk) -> (Batch size, n_heads, Seq len, Dk)
        output = weight @ v

        # (Batch size, n_heads, Seq len, Dk) -> (Batch size, Seq len, n_heads, Dk)
        output = output.transpose(1, 2)

        # (Batch size, Seq len, d_model)
        output = output.reshape(input_shape)

        output = self.out_proj(output)
        
        return output
    
class CrossAttention(nn.Module):
    def __init__(self, n_heads: int, d_model: int, d_cross: int, in_proj_bias=True, out_proj_bias=True):
        super().__init__()
        self.q_proj = nn.Linear(d_model, d_model, bias=in_proj_bias)
        self.k_proj = nn.Linear(d_cross, d_model, bias=in_proj_bias)
        self.v_proj = nn.Linear(d_cross, d_model, bias=in_proj_bias)
        self.out_proj = nn.Linear(d_model, d_model, bias=out_proj_bias)
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

    def forward(self, latent: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        latent: (Batch size, Seq_len_Q, d_model)
        context: (Batch size, Seq_len_KV, d_cross) = (Batch size, 77, 768)
        """

        input_shape = latent.shape
        batch_size, sequence_length, d_model = input_shape

        intermediate_shape = (batch_size, -1, self.n_heads, self.d_head)

        # Multiply q, k, v with Wq, Wk, Wv
        q = self.q_proj(latent)
        k = self.k_proj(context)
        v = self.v_proj(context)

        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        weight = q @ k.transpose(-1 ,-2)

        weight /= math.sqrt(self.d_head)

        weight = F.softmax(weight, dim=-1)

        output = weight @ v

        output = output.transpose(1, 2).contiguous()

        output = output.view(input_shape)

        output = self.out_proj(output)

        return output



