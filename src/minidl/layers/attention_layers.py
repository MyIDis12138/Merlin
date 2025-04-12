import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadAttention(torch.nn.Module):
    """
    Standard Multi-Head Attention module
    """

    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)

        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, q, k, v, mask=None):
        # q: (B, num_heads, Seq_Len_Q, head_dim)
        # k: (B, num_heads, Seq_Len_K, head_dim)
        # v: (B, num_heads, Seq_Len_V, head_dim)

        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_k**0.5)  # (B, num_heads, Seq_Len_Q, Seq_Len_K)

        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float("-inf"))

        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)

        output = torch.matmul(attn_probs, v)

        return output, attn_probs

    # CHANGE: Use standard forward(query, key, value) order
    def forward(self, query, key, value, mask=None):
        # query: (B, Seq_Len_Q, D)
        # key:   (B, Seq_Len_K, D)
        # value: (B, Seq_Len_V, D)

        B, Seq_Len_Q, _ = query.shape
        B, Seq_Len_K, _ = key.shape
        B, Seq_Len_V, _ = value.shape

        q = self.q_proj(query).view(B, Seq_Len_Q, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, Seq_Len_Q, head_dim)
        k = self.k_proj(key).view(B, Seq_Len_K, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, Seq_Len_K, head_dim)
        v = self.v_proj(value).view(B, Seq_Len_V, self.num_heads, self.d_k).transpose(1, 2)  # (B, num_heads, Seq_Len_V, head_dim)

        attn_output, attn_weights = self.scaled_dot_product_attention(q, k, v, mask)

        attn_output = attn_output.transpose(1, 2).contiguous().view(B, Seq_Len_Q, self.d_model)  # (B, Seq_Len_Q, d_model)
        out = self.out_proj(attn_output)

        return out, attn_weights
