import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, DataEmbedding_inverted, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
import numpy as np
import math


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class BasicPatchEmbedding(nn.Module):
    """Basic patch embedding WITHOUT any enhancements"""
    def __init__(self, n_vars, d_model, patch_len, stride, padding, dropout):
        super(BasicPatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        
        # Basic patch embedding (NO enhanced initialization)
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        # NO Xavier initialization
        
        # NO enhanced global token
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        # NO Xavier initialization for global token
        
        # Basic positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # Standard normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [bs, nvars, seq_len]
        batch_size, n_vars, seq_len = x.shape
        
        # Apply padding
        if self.padding:
            x = F.pad(x, (0, self.padding), mode='replicate')
        
        # Create patches with unfold
        x_patch = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x_patch: [bs, nvars, patch_num, patch_len]
        
        curr_patch_num = x_patch.shape[2]
        
        # Reshape for processing
        x_patch = x_patch.reshape(-1, curr_patch_num, self.patch_len)
        
        # Apply basic patch embedding (NO enhancements)
        x_embedded = self.value_embedding(x_patch)  # [bs*nvars, patch_num, d_model]
        
        # Add basic positional encoding
        pos_embed = self.position_embedding(x_embedded)
        x_embedded = x_embedded + pos_embed
        
        # Reshape back
        x_embedded = x_embedded.reshape(batch_size, n_vars, curr_patch_num, -1)
        
        # Add basic global token (NO enhancements)
        glb = self.glb_token.repeat(batch_size, 1, 1, 1)
        x_embedded = torch.cat([x_embedded, glb], dim=2)  # [bs, nvars, patch_num+1, d_model]
        
        # Reshape for encoder
        x_final = x_embedded.reshape(-1, curr_patch_num + 1, x_embedded.shape[-1])
        
        # Apply standard normalization and dropout
        x_final = self.layer_norm(x_final)
        x_final = self.dropout(x_final)
        
        return x_final, n_vars


class BasicEncoderLayer(nn.Module):
    """Basic encoder layer WITHOUT any enhancements"""
    def __init__(self, self_attention, d_model, d_ff=None,
                 dropout=0.1, activation="gelu"):
        super(BasicEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        
        # NO frequency-enhanced attention
        
        # Basic feed-forward
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Standard normalization (NO adaptive normalization)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        # NO frequency-enhanced attention
        
        # Standard self-attention only
        std_attn_out = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=delta)[0]
        x = x + self.dropout(std_attn_out)
        x = self.norm1(x)
        
        # NO cross-attention (no hybrid functionality)
        
        # Basic feed-forward
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm2(x + y)


class BasicPredictionHead(nn.Module):
    """Basic prediction head WITHOUT enhancements"""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.target_window = target_window
        
        # Basic prediction (NO residual connections)
        self.projection = nn.Linear(nf, target_window)
        
        self.flatten = nn.Flatten(start_dim=-2)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.dropout(x)
        
        # Basic prediction (NO enhancements)
        output = self.projection(x)
        
        return output


class Model(nn.Module):
    """
    Basic Patch Model - Removes ALL PatchXFormer enhancements
    
    Ablation study: Basic patch-based model without any enhancements
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_vars = configs.enc_in
        
        # Basic patch embedding (NO enhancements)
        padding = stride
        self.patch_embedding = BasicPatchEmbedding(
            self.n_vars, configs.d_model, patch_len, stride, padding, configs.dropout)
        
        # Basic exogenous embedding (not used without cross-attention)
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Calculate patch number
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        
        # Basic encoder (NO enhancements)
        self.encoder = nn.ModuleList([
            BasicEncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation
            ) for l in range(configs.e_layers)
        ])
        
        # Final normalization
        self.final_norm = nn.LayerNorm(configs.d_model)
        
        # Basic prediction head
        self.head_nf = configs.d_model * (self.patch_num + 1)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = BasicPredictionHead(configs.enc_in, self.head_nf, configs.pred_len,
                                          head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = BasicPredictionHead(configs.enc_in, self.head_nf, configs.seq_len,
                                          head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Basic normalization
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Basic patch embedding
        x_enc = x_enc.permute(0, 2, 1)  # [bs, nvars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [bs*nvars, patch_num+1, d_model]
        
        # NO exogenous embedding processing
        
        # Basic encoder processing
        for layer in self.encoder:
            enc_out = layer(enc_out)
        
        # Final normalization
        enc_out = self.final_norm(enc_out)
        
        # Reshape for prediction head
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num+1]
        
        # Basic prediction
        dec_out = self.head(enc_out)  # [bs, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs, pred_len, nvars]
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Basic normalization for imputation
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        
        # Same processing as forecast
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        for layer in self.encoder:
            enc_out = layer(enc_out)
        
        enc_out = self.final_norm(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return dec_out

    def anomaly_detection(self, x_enc):
        # Basic enhancement as forecast
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        for layer in self.encoder:
            enc_out = layer(enc_out)
        
        enc_out = self.final_norm(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        for layer in self.encoder:
            enc_out = layer(enc_out)
        
        enc_out = self.final_norm(enc_out)
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        output = self.flatten(enc_out)
        output = self.dropout(output)
        output = output.reshape(output.shape[0], -1)
        output = self.projection(output)
        
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
        if self.task_name == 'imputation':
            dec_out = self.imputation(x_enc, x_mark_enc, x_dec, x_mark_dec, mask)
            return dec_out  # [B, L, D]
        if self.task_name == 'anomaly_detection':
            dec_out = self.anomaly_detection(x_enc)
            return dec_out  # [B, L, D]
        if self.task_name == 'classification':
            dec_out = self.classification(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        return None
