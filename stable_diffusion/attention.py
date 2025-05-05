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

    def forward(self, input: torch.Tensor, causal_mask=False):
        # input: (Batch_size, Seq_len, d_model)

        input_shape = input.shape
        batch_size, sequence_length, d_model = input_shape

        intermediate_shape = (batch_size, sequence_length, self.n_heads, self.d_head)

        # (Batch_size, Seq_len, d_model) -> (Batch_size, Seq_len, d_model*3) -> 3 tensors of shape (Batch_size, Seq_len, d_model)
        q, k, v = self.in_proj(input).chunk(3, d_model=-1)

        # dk = d_model / n_heads
        # (Batch_size, Seq_len, d_model) -> (Batch_size, Seq_len, n_heads, dk) -> (Batch_size, n_heads, Seq_len, dk)
        q = q.view(intermediate_shape).transpose(1, 2)
        k = k.view(intermediate_shape).transpose(1, 2)
        v = v.view(intermediate_shape).transpose(1, 2)

        # (Batch_size, n_heads, Seq_len, Seq_len)
        weight = q @ k.transpose(-1, -2)

        if causal_mask:
            # Mask where the upper triangle (above principle diagonal) is made up of 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            weight.masked_filled_(mask, -torch.inf)

        weight /= math.sqrt(self.d_head)
        weight = F.softmax(weight, d_model=-1)

        # (Batch_size, n_heads, Seq_len, Seq_len) @ (Batch_size, n_heads, Seq_len, dk) -> (Batch_size, n_heads, Seq_len, dk)
        output = weight @ v

        # (Batch_size, n_heads, Seq_len, dk) -> (Batch_size, Seq_len, n_heads, dk)
        output = output.transpose(1, 2)

        # (Batch_size, Seq_len, d_model)
        output = output.reshape(input_shape)

        output = self.out_proj(output)

        # (Batch_size, Seq_len, d_model)
        return output