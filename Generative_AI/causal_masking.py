
import torch
from multi_head_attention import MultiHeadAttention

torch.manual_seed(42)

batch_size = 2
seq_length = 5
d_model = 8
num_heads = 2

# Create input tensor
x = torch.randn(batch_size, seq_length, d_model)

# Create causal mask
causal_mask = torch.tril(
    torch.ones(seq_length, seq_length)
).view(1, 1, seq_length, seq_length)

print("Causal Mask:")
print(causal_mask[0, 0])

# Initialize Multi-Head Attention
model = MultiHeadAttention(d_model, num_heads)

# Apply causal masking
output, weights = model(x, causal_mask)

print("\nMasked Attention Weights:")
print(weights[0, 0])

print("\nInput shape:", x.shape)
print("Attention weights shape:", weights.shape)
print("Final output shape:", output.shape)

# Verify future tokens are masked
print("\nMasking verified:",
      torch.all(weights.triu(diagonal=1) == 0).item())
