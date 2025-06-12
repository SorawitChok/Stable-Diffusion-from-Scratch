import torch
from torch import nn
import torch.nn.functional as F

from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, n_vocab: int, d_model: int, n_tokens: int):
        super().__init__()

        self.token_embedding = nn.Embedding(n_vocab, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(n_tokens, d_model))

    def forward(self, tokens):
        # (Batch size, Seq len) -> (Batch size, Seq len, Dim)
        x = self.token_embedding(tokens)
        x += self.positional_encoding

        return x

class CLIPLayer(nn.Module):
    def __init__(self, n_head: int, d_model: int):
        super().__init__()

        self.layernorm_1 = nn.LayerNorm(d_model)
        self.attention = SelfAttention(n_head, d_model)
        self.layernorm_2 = nn.LayerNorm(d_model)
        self.linear_1 = nn.Linear(d_model, d_model*4)
        self.linear_2 = nn.Linear(d_model*4, d_model)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        input: (Batch size, Seq len, d_model)
        """
        
        residue = input

        # Self Attention Block
        x = self.layernorm_1(input)
        x = self.attention(x, casual_mask=True)
        x += residue

        residue = x

        # Feedforward Block
        x = self.layernorm_2(x)
        x = self.linear_1(x)
        x = x * torch.sigmoid(1.702*x) # Quick GeLU activation
        x = self.linear_2(x)

        x += residue

        return x 

class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
        ])
        self.layernorm = nn.LayerNorm(768)

    def forward(self, tokens: torch.LongTensor) -> torch.FloatTensor:
        tokens = tokens.type(torch.long)

        # (Batch size, Seq len) -> (Batch size, Seq len, d_model [768])
        state = self.embedding(tokens)

        for layer in self.layers:
            state = layer(state)

        # (Batch size, Seq len, d_model)
        output = self.layernorm(state)

        return output