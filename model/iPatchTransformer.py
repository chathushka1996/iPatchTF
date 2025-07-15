import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import DataEmbedding_inverted
import numpy as np
import math


class PatchEmbedding(nn.Module):
    """
    Patch embedding layer that converts time series into patches and embeds them.
    """
    def __init__(self, patch_len, stride, padding_patch, input_len, d_model):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding_patch = padding_patch
        
        # Calculate number of patches
        if padding_patch == 'end':
            self.num_patches = int((input_len - patch_len) / stride + 1)
        else:
            self.num_patches = int((input_len + stride - patch_len) / stride)
        
        # Linear projection for patches
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
    def forward(self, x):
        """
        x: [batch_size, num_variables, input_len]
        output: [batch_size, num_variables, num_patches, d_model]
        """
        batch_size, num_vars, seq_len = x.shape
        
        if self.padding_patch == 'end':
            # Padding at the end
            if seq_len % self.stride != 0:
                length_add = self.stride - (seq_len % self.stride)
                x = torch.cat([x, x[:, :, -1:].repeat(1, 1, length_add)], dim=-1)
        else:
            # No padding
            pass
            
        # Extract patches
        patches = []
        for i in range(0, x.size(-1) - self.patch_len + 1, self.stride):
            patch = x[:, :, i:i + self.patch_len]
            patches.append(patch)
        
        if len(patches) == 0:
            # Handle edge case
            patches = [x[:, :, :self.patch_len]]
            
        patches = torch.stack(patches, dim=2)  # [batch_size, num_vars, num_patches, patch_len]
        
        # Apply embedding
        embedded_patches = self.value_embedding(patches)  # [batch_size, num_vars, num_patches, d_model]
        
        return embedded_patches


class VariablePatchAttention(nn.Module):
    """
    Attention mechanism that operates on variable-patch tokens.
    Variables attend to each other while considering their patch representations.
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(VariablePatchAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_qs = nn.Linear(d_model, d_model, bias=False)
        self.w_ks = nn.Linear(d_model, d_model, bias=False)
        self.w_vs = nn.Linear(d_model, d_model, bias=False)
        
        self.fc = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x, attn_mask=None):
        """
        x: [batch_size, num_variables, num_patches, d_model]
        """
        batch_size, num_vars, num_patches, d_model = x.shape
        
        # Reshape for multi-head attention: treat each patch as a token
        # [batch_size, num_variables * num_patches, d_model]
        x_reshaped = x.view(batch_size, num_vars * num_patches, d_model)
        
        # Multi-head attention
        Q = self.w_qs(x_reshaped).view(batch_size, num_vars * num_patches, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_ks(x_reshaped).view(batch_size, num_vars * num_patches, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_vs(x_reshaped).view(batch_size, num_vars * num_patches, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if attn_mask is not None:
            scores.masked_fill_(attn_mask, -1e9)
            
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        context = torch.matmul(attn_weights, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, num_vars * num_patches, d_model)
        
        # Final linear transformation
        output = self.fc(context)
        
        # Reshape back to [batch_size, num_variables, num_patches, d_model]
        output = output.view(batch_size, num_vars, num_patches, d_model)
        
        return output, attn_weights


class iPatchTransformerBlock(nn.Module):
    """
    iPatchTransformer encoder block with variable-patch attention and feed-forward.
    """
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1, activation='gelu'):
        super(iPatchTransformerBlock, self).__init__()
        
        self.attention = VariablePatchAttention(d_model, n_heads, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Feed forward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU() if activation == 'gelu' else nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x, attn_mask=None):
        # Multi-head attention with residual connection
        attn_output, attn_weights = self.attention(x, attn_mask)
        x = self.norm1(x + attn_output)
        
        # Feed-forward with residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)
        
        return x, attn_weights


class Model(nn.Module):
    """
    iPatchTransformer: Inverted Patch Transformer for Time Series Forecasting
    
    This model combines:
    1. iTransformer's variable-as-tokens approach
    2. PatchTST's patching mechanism for local temporal semantics
    3. Efficient variable-level attention on patch representations
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        
        # Patch configuration
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        self.padding_patch = getattr(configs, 'padding_patch', 'end')
        
        # Model dimensions
        self.d_model = configs.d_model
        self.n_heads = configs.n_heads
        self.e_layers = configs.e_layers
        self.d_ff = configs.d_ff
        self.dropout = configs.dropout
        self.activation = configs.activation
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(
            patch_len=self.patch_len,
            stride=self.stride,
            padding_patch=self.padding_patch,
            input_len=self.seq_len,
            d_model=self.d_model
        )
        
        # Calculate number of patches
        if self.padding_patch == 'end':
            self.num_patches = int((self.seq_len - self.patch_len) / self.stride + 1)
        else:
            self.num_patches = int((self.seq_len + self.stride - self.patch_len) / self.stride)
        
        # Positional encoding for patches
        self.pos_encoding = nn.Parameter(torch.randn(self.num_patches, self.d_model))
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            iPatchTransformerBlock(
                d_model=self.d_model,
                n_heads=self.n_heads,
                d_ff=self.d_ff,
                dropout=self.dropout,
                activation=self.activation
            ) for _ in range(self.e_layers)
        ])
        
        self.norm = nn.LayerNorm(self.d_model)
        
        # Projection head: from patches to prediction
        # We need to project from [num_patches, d_model] to pred_len for each variable
        self.projector = nn.Linear(self.num_patches * self.d_model, self.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        """
        Forward pass for forecasting
        """
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size; L: seq_len; N: number of variates (variables)

        # Step 1: Invert dimensions (iTransformer approach)
        # B L N -> B N L
        x_inverted = x_enc.permute(0, 2, 1)
        
        # Step 2: Apply patch embedding to each variable
        # B N L -> B N num_patches d_model
        patches = self.patch_embedding(x_inverted)
        
        # Step 3: Add positional encoding
        patches = patches + self.pos_encoding.unsqueeze(0).unsqueeze(0)
        
        # Step 4: Apply transformer encoder layers
        enc_out = patches
        attentions = []
        
        for layer in self.encoder_layers:
            enc_out, attn = layer(enc_out)
            if self.output_attention:
                attentions.append(attn)
        
        enc_out = self.norm(enc_out)
        
        # Step 5: Project to prediction
        # Flatten patches: B N num_patches d_model -> B N (num_patches * d_model)
        enc_out_flat = enc_out.view(enc_out.size(0), enc_out.size(1), -1)
        
        # Project to prediction length: B N (num_patches * d_model) -> B N pred_len
        dec_out = self.projector(enc_out_flat)
        
        # Step 6: Reshape to final output format
        # B N pred_len -> B pred_len N
        dec_out = dec_out.permute(0, 2, 1)[:, :, :N]  # Filter out any extra dimensions
        
        # De-normalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out, attentions

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


# Additional utility functions for the model
def create_patch_mask(seq_len, patch_len, stride):
    """
    Create a mask for patch-based attention if needed
    """
    num_patches = int((seq_len - patch_len) / stride + 1)
    mask = torch.zeros(num_patches, num_patches)
    return mask


def calculate_patch_config(seq_len, desired_patches=None, desired_patch_len=None):
    """
    Helper function to calculate optimal patch configuration
    """
    if desired_patches is not None:
        patch_len = seq_len // desired_patches
        stride = patch_len
    elif desired_patch_len is not None:
        patch_len = desired_patch_len
        stride = patch_len // 2  # 50% overlap
    else:
        # Default configuration
        patch_len = 16
        stride = 8
    
    num_patches = int((seq_len - patch_len) / stride + 1)
    
    return {
        'patch_len': patch_len,
        'stride': stride,
        'num_patches': num_patches
    } 