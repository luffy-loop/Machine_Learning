
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model, num_heads):
        super().__init__()

        assert d_model % num_heads == 0, (
            "d_model must be divisible by num_heads"
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch_size, seq_length, _ = x.shape

        # Generate Query, Key and Value
        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # Split into multiple heads
        Q = Q.view(
            batch_size, seq_length,
            self.num_heads, self.d_k
        ).transpose(1, 2)

        K = K.view(
            batch_size, seq_length,
            self.num_heads, self.d_k
        ).transpose(1, 2)

        V = V.view(
            batch_size, seq_length,
            self.num_heads, self.d_k
        ).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1))
        scores = scores / math.sqrt(self.d_k)

        if mask is not None:
            scores = scores.masked_fill(
                mask == 0, float('-inf')
            )

        attention_weights = F.softmax(scores, dim=-1)
        attention_output = torch.matmul(attention_weights, V)

        # Combine all heads
        attention_output = attention_output.transpose(
            1, 2
        ).contiguous()

        attention_output = attention_output.view(
            batch_size, seq_length, self.d_model
        )

        # Final linear projection
        output = self.W_o(attention_output)

        return output, attention_weights


torch.manual_seed(42)

batch_size = 2
seq_length = 5
d_model = 8
num_heads = 2

x = torch.randn(batch_size, seq_length, d_model)

model = MultiHeadAttention(d_model, num_heads)
output, weights = model(x)

print("Input shape:", x.shape)
print("Attention weights shape:", weights.shape)
print("Final output shape:", output.shape)

print("\nFinal output:")
print(output)
