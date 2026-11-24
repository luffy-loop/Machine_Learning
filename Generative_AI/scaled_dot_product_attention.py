
import torch
import torch.nn.functional as F
import math

def scaled_dot_product_attention(Q, K, V, mask=None):
    # Step 1: Calculate attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1))

    # Step 2: Scale the scores
    d_k = Q.size(-1)
    scores = scores / math.sqrt(d_k)

    # Step 3: Apply mask if provided
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))

    # Step 4: Apply softmax
    attention_weights = F.softmax(scores, dim=-1)

    # Step 5: Calculate attention output
    output = torch.matmul(attention_weights, V)

    return output, attention_weights


Q = torch.randn(1, 3, 4)
K = torch.randn(1, 3, 4)
V = torch.randn(1, 3, 4)

output, weights = scaled_dot_product_attention(Q, K, V)

print("Q shape:", Q.shape)
print("K shape:", K.shape)
print("V shape:", V.shape)

print("\nAttention output:")
print(output)

print("\nAttention weights:")
print(weights)

print("\nOutput shape:", output.shape)
