import torch
import torch.nn as nn


class FactorizedPositionalEmbedding3D(nn.Module):
    """
    Factorized 3D positional embeddings for volumetric data.

    This class implements efficient positional embeddings by factorizing
    the 3D positions into separate dimensions, reducing parameter count
    and computational complexity compared to full positional embeddings.

    Args:
        embedding_dim (int): Dimension of the embeddings for each spatial dimension
        max_depth (int): Maximum depth dimension size
        max_height (int): Maximum height dimension size
        max_width (int): Maximum width dimension size
        dropout (float): Dropout rate applied to embeddings
    """

    def __init__(self, embedding_dim: int, max_depth: int = 128, max_height: int = 128, max_width: int = 128, dropout: float = 0.1):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.max_depth = max_depth
        self.max_height = max_height
        self.max_width = max_width

        # Create separate embedding tables for each dimension
        self.d_embeddings = nn.Embedding(max_depth, embedding_dim)
        self.h_embeddings = nn.Embedding(max_height, embedding_dim)
        self.w_embeddings = nn.Embedding(max_width, embedding_dim)

        # Apply dropout to embeddings
        self.dropout = nn.Dropout(dropout)

        # Initialize embeddings
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize embedding weights."""
        nn.init.normal_(self.d_embeddings.weight, mean=0, std=0.02)
        nn.init.normal_(self.h_embeddings.weight, mean=0, std=0.02)
        nn.init.normal_(self.w_embeddings.weight, mean=0, std=0.02)

    def forward(self, depth: int, height: int, width: int, batch_size: int = 1, device=None):
        """
        Generate factorized positional embeddings for a 3D volume.

        Args:
            depth (int): Depth dimension size
            height (int): Height dimension size
            width (int): Width dimension size
            batch_size (int): Batch size
            device: Device to create tensors on (default: None, uses embedding device)

        Returns:
            torch.Tensor: Positional embeddings tensor of shape [batch_size, depth*height*width, embedding_dim*3]
        """
        if device is None:
            device = self.d_embeddings.weight.device

        d_indices = torch.arange(min(depth, self.max_depth), device=device)
        h_indices = torch.arange(min(height, self.max_height), device=device)
        w_indices = torch.arange(min(width, self.max_width), device=device)

        d_grid, h_grid, w_grid = torch.meshgrid(d_indices, h_indices, w_indices, indexing="ij")

        # Get embeddings for all positions in the grid
        d_pos_emb = self.d_embeddings(d_grid.reshape(-1)).reshape(len(d_indices), len(h_indices), len(w_indices), -1)
        h_pos_emb = self.h_embeddings(h_grid.reshape(-1)).reshape(len(d_indices), len(h_indices), len(w_indices), -1)
        w_pos_emb = self.w_embeddings(w_grid.reshape(-1)).reshape(len(d_indices), len(h_indices), len(w_indices), -1)

        # Combine the embeddings
        pos_emb = torch.cat([d_pos_emb, h_pos_emb, w_pos_emb], dim=-1)  # [D, H, W, 3*emb_dim]

        # Reshape to match expected shape [seq_len, emb_dim]
        pos_emb = pos_emb.reshape(-1, pos_emb.size(-1))  # [D*H*W, 3*emb_dim]

        if depth > self.max_depth or height > self.max_height or width > self.max_width:
            full_seq_len = depth * height * width

            if pos_emb.size(0) < full_seq_len:
                padding_len = full_seq_len - pos_emb.size(0)
                avg_emb = pos_emb.mean(dim=0, keepdim=True).expand(padding_len, -1)
                pos_emb = torch.cat([pos_emb, avg_emb], dim=0)

        pos_emb = self.dropout(pos_emb)
        pos_emb = pos_emb.unsqueeze(0).expand(batch_size, -1, -1)

        return pos_emb

    def add_to_input(self, input_tensor: torch.Tensor, depth: int, height: int, width: int):
        """
        Add positional embeddings to an input tensor.

        Args:
            input_tensor (torch.Tensor): Input tensor of shape [batch_size, seq_len, hidden_dim]
            depth (int): Depth dimension size
            height (int): Height dimension size
            width (int): Width dimension size

        Returns:
            torch.Tensor: Input tensor with positional embeddings added
        """
        batch_size, seq_len, hidden_dim = input_tensor.shape

        # Generate positional embeddings
        pos_emb = self.forward(depth, height, width, batch_size, device=input_tensor.device)

        # If dimensions don't match exactly, adjust as needed
        if pos_emb.size(1) != seq_len or pos_emb.size(2) != hidden_dim:
            # Handle sequence length mismatch
            if pos_emb.size(1) > seq_len:
                pos_emb = pos_emb[:, :seq_len, :]
            elif pos_emb.size(1) < seq_len:
                # Pad by repeating the pattern
                pad_len = seq_len - pos_emb.size(1)
                pos_emb = torch.cat([pos_emb, pos_emb[:, :pad_len, :]], dim=1)

            # Handle feature dimension mismatch
            if pos_emb.size(2) > hidden_dim:
                pos_emb = pos_emb[:, :, :hidden_dim]
            elif pos_emb.size(2) < hidden_dim:
                # Pad feature dimension with zeros
                pad_feat = hidden_dim - pos_emb.size(2)
                padding = torch.zeros(batch_size, seq_len, pad_feat, device=pos_emb.device)
                pos_emb = torch.cat([pos_emb, padding], dim=2)

        # Add positional embeddings to input
        return input_tensor + pos_emb
