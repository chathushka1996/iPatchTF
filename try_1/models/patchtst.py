import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
import math

class PatchEmbedding(nn.Module):
    """
    2D Image to Patch Embedding adapted for 1D time series
    """
    def __init__(self, seq_len, patch_size, stride, padding, in_chans, embed_dim, norm_layer=None):
        super().__init__()
        self.seq_len = seq_len
        self.patch_size = patch_size
        self.stride = stride
        self.padding = padding
        
        # Calculate number of patches
        patch_num = int((seq_len + 2 * padding - patch_size) / stride + 1)
        self.patch_num = patch_num
        
        # Patch embedding projection
        self.proj = nn.Conv1d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        """
        x: [B, L, C] -> [B, N, D]
        B: batch size, L: sequence length, C: number of channels
        N: number of patches, D: embedding dimension
        """
        B, L, C = x.shape
        assert L == self.seq_len, f"Input sequence length ({L}) doesn't match expected ({self.seq_len})"
        
        # Convert to [B, C, L] for conv1d
        x = x.transpose(1, 2)  # [B, C, L]
        
        # Apply patch embedding
        x = self.proj(x)  # [B, D, N]
        x = x.transpose(1, 2)  # [B, N, D]
        x = self.norm(x)
        
        return x

class PositionalEncoding(nn.Module):
    """
    Positional encoding for patches
    """
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1), :].transpose(0, 1)

class MultiHeadAttention(nn.Module):
    """
    Multi-head self-attention mechanism
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        
        # Linear transformations and split into heads
        Q = self.w_q(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads and put through final linear layer
        context = context.transpose(1, 2).contiguous().view(
            batch_size, -1, self.d_model
        )
        
        output = self.w_o(context)
        
        return output, attention_weights

class FeedForward(nn.Module):
    """
    Position-wise feed-forward network
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super().__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x):
        return self.linear2(self.dropout(self.activation(self.linear1(x))))

class TransformerEncoderLayer(nn.Module):
    """
    Single transformer encoder layer
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, n_heads, dropout)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Self-attention
        attn_output, _ = self.self_attention(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        return x

class TransformerEncoder(nn.Module):
    """
    Stack of transformer encoder layers
    """
    def __init__(self, layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

class PatchTST(nn.Module):
    """
    PatchTST: A Time Series Worth 64 Words: Long-term Forecasting with Transformers
    Modified for the hybrid model
    """
    def __init__(self, config):
        super(PatchTST, self).__init__()
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.patch_size = config.patch_size
        self.stride = config.stride
        self.d_model = config.d_model
        self.n_heads = config.n_heads
        self.e_layers = config.e_layers
        self.d_ff = config.d_ff
        self.dropout = config.dropout
        
        # Input dimensions
        self.enc_in = len(config.feature_cols) + len(config.time_cols)  # All input features
        
        # Calculate padding for patch embedding
        self.padding = (self.patch_size - self.stride) // 2
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            seq_len=self.seq_len,
            patch_size=self.patch_size,
            stride=self.stride,
            padding=self.padding,
            in_chans=self.enc_in,
            embed_dim=self.d_model
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model)
        
        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            dropout=self.dropout
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, self.e_layers)
        
        # Head for prediction
        self.head = nn.Linear(self.patch_embedding.patch_num * self.d_model, self.pred_len)
        
        # Dropout
        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(self, x):
        """
        x: [B, L, C] where B=batch, L=seq_len, C=features
        Output: [B, pred_len, 1] for univariate prediction
        """
        B, L, C = x.shape
        
        # Patch embedding
        x = self.patch_embedding(x)  # [B, N, D]
        
        # Add positional encoding
        x = self.pos_encoding(x)
        x = self.dropout_layer(x)
        
        # Transformer encoding
        x = self.transformer_encoder(x)  # [B, N, D]
        
        # Flatten and predict
        x = x.reshape(B, -1)  # [B, N*D]
        x = self.head(x)  # [B, pred_len]
        x = x.unsqueeze(-1)  # [B, pred_len, 1]
        
        return x

class RevIN(nn.Module):
    """
    Reversible Instance Normalization for better distribution shift handling
    """
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode='norm'):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

class EnhancedPatchTST(nn.Module):
    """
    Enhanced PatchTST with RevIN and additional improvements
    """
    def __init__(self, config):
        super(EnhancedPatchTST, self).__init__()
        
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        self.enc_in = len(config.feature_cols) + len(config.time_cols)
        
        # RevIN for better normalization
        self.revin = RevIN(self.enc_in)
        
        # Core PatchTST
        self.patchtst = PatchTST(config)
        
        # Additional components for enhanced performance
        self.residual_projection = nn.Linear(self.seq_len, self.pred_len)
        self.output_projection = nn.Linear(2, 1)  # Combine PatchTST and residual
        
    def forward(self, x):
        """
        Enhanced forward pass with residual connections
        """
        # Store original for residual connection
        x_orig = x.clone()
        
        # Apply RevIN normalization
        x = self.revin(x, 'norm')
        
        # PatchTST prediction
        patch_output = self.patchtst(x)  # [B, pred_len, 1]
        
        # Denormalize PatchTST output
        patch_output = self.revin(patch_output, 'denorm')
        
        # Simple residual connection (last value carried forward)
        last_values = x_orig[:, -1:, -1:]  # [B, 1, 1] - last value of target
        residual_output = last_values.repeat(1, self.pred_len, 1)  # [B, pred_len, 1]
        
        # Combine outputs
        combined = torch.cat([patch_output, residual_output], dim=-1)  # [B, pred_len, 2]
        output = self.output_projection(combined)  # [B, pred_len, 1]
        
        return output 