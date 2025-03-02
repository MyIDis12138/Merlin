import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(torch.nn.Module):
    """
    Standard Multi-Head Attention module
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.q_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        attn = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)

        return out, attn

    def forward(self, k, v, q, mask=None):
        B, L, D = k.shape

        k = self.k_proj(k).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        q = self.q_proj(q).view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        out, attn = self.scaled_dot_product_attention(q, k, v, mask)

        out = out.transpose(1, 2).contiguous().view(B, -1, self.d_model)
        out = self.out_proj(out)

        return out, attn
