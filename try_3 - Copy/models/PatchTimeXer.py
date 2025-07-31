import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Embed import PatchEmbedding, DataEmbedding_inverted, PositionalEmbedding
from layers.Transformer_EncDec import Encoder, EncoderLayer
import numpy as np


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class FlattenHead(nn.Module):
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(nf, target_window)
        self.dropout = nn.Dropout(head_dropout)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        return x


class HybridPatchEmbedding(nn.Module):
    """Enhanced patch embedding combining PatchTST efficiency with TimeXer's global token"""
    def __init__(self, n_vars, d_model, patch_len, stride, padding, dropout):
        super(HybridPatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.padding = padding
        
        # PatchTST-style patch embedding
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        
        # TimeXer-style global token
        self.glb_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        
        # Enhanced positional embedding
        self.position_embedding = PositionalEmbedding(d_model)
        
        # Layer normalization for better stability
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [bs, nvars, seq_len]
        n_vars = x.shape[1]
        
        # Apply padding
        if self.padding:
            x = F.pad(x, (0, self.padding), mode='replicate')
        
        # Create patches with unfold
        x_patch = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # x_patch: [bs, nvars, patch_num, patch_len]
        
        batch_size, nvars, patch_num, patch_len = x_patch.shape
        
        # Reshape for processing
        x_patch = x_patch.reshape(-1, patch_num, patch_len)
        
        # Apply patch embedding
        x_embedded = self.value_embedding(x_patch)  # [bs*nvars, patch_num, d_model]
        
        # Add positional embedding
        x_embedded = x_embedded + self.position_embedding(x_embedded)
        
        # Reshape back
        x_embedded = x_embedded.reshape(batch_size, nvars, patch_num, -1)
        
        # Add global token
        glb = self.glb_token.repeat(batch_size, 1, 1, 1)
        x_embedded = torch.cat([x_embedded, glb], dim=2)  # [bs, nvars, patch_num+1, d_model]
        
        # Reshape for encoder
        x_final = x_embedded.reshape(-1, patch_num + 1, x_embedded.shape[-1])
        
        # Apply layer norm and dropout
        x_final = self.layer_norm(x_final)
        x_final = self.dropout(x_final)
        
        return x_final, n_vars


class HybridEncoderLayer(nn.Module):
    """Enhanced encoder layer combining self-attention with global-exogenous cross-attention"""
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(HybridEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        
        # Enhanced feed-forward with better activation
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Multiple layer norms for better gradient flow
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Self-attention on patches
        attn_out = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=delta)[0]
        x = x + self.dropout(attn_out)
        x = self.norm1(x)
        
        # Cross-attention between global token and exogenous features (if available)
        if cross is not None:
            B = cross.shape[0]
            L, D = x.shape[1], x.shape[2]
            
            # Extract global token (last token)
            x_glb_ori = x[:, -1, :].unsqueeze(1)  # [bs*nvars, 1, d_model]
            x_glb = x_glb_ori.reshape(B, -1, D)  # [bs, nvars, d_model]
            
            # Cross-attention with exogenous features
            x_glb_attn = self.cross_attention(x_glb, cross, cross, 
                                            attn_mask=cross_mask, tau=tau, delta=delta)[0]
            x_glb_attn = x_glb_attn.reshape(-1, 1, D)  # [bs*nvars, 1, d_model]
            
            # Update global token
            x_glb = x_glb_ori + self.dropout(x_glb_attn)
            x_glb = self.norm2(x_glb)
            
            # Replace global token in sequence
            x = torch.cat([x[:, :-1, :], x_glb], dim=1)
        
        # Feed-forward network
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        
        return self.norm3(x + y)


class HybridEncoder(nn.Module):
    """Enhanced encoder supporting both self and cross attention"""
    def __init__(self, layers, norm_layer=None):
        super(HybridEncoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask, tau=tau, delta=delta)
        
        if self.norm is not None:
            x = self.norm(x)
        
        return x, None


class Model(nn.Module):
    """
    PatchTimeXer: A hybrid model combining the best features of PatchTST and TimeXer
    
    Key Features:
    1. Efficient patching from PatchTST with enhanced embedding
    2. Global token mechanism from TimeXer for capturing global patterns
    3. Dual embedding for both endogenous and exogenous features
    4. Cross-attention between patches and global context
    5. Enhanced normalization and regularization
    6. Multi-scale feature integration
    """

    def __init__(self, configs, patch_len=16, stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.n_vars = configs.enc_in
        
        # Enhanced patching and embedding
        padding = stride
        self.patch_embedding = HybridPatchEmbedding(
            self.n_vars, configs.d_model, patch_len, stride, padding, configs.dropout)
        
        # Exogenous embedding (from TimeXer)
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Calculate patch number
        self.patch_num = int((configs.seq_len - patch_len) / stride + 2)
        
        # Enhanced encoder with hybrid layers
        self.encoder = HybridEncoder(
            [
                HybridEncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    AttentionLayer(
                        FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                      output_attention=False), configs.d_model, configs.n_heads),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model)
        )
        
        # Enhanced prediction head
        self.head_nf = configs.d_model * (self.patch_num + 1)  # +1 for global token
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.pred_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = FlattenHead(configs.enc_in, self.head_nf, configs.seq_len,
                                    head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Enhanced normalization (from Non-stationary Transformer)
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1)  # [bs, nvars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [bs*nvars, patch_num+1, d_model]
        
        # Exogenous embedding
        ex_embed = None
        if x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)  # [bs, seq_len, d_model]
        
        # Encoder with cross-attention
        enc_out, attns = self.encoder(enc_out, ex_embed)
        
        # Reshape for prediction head
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num+1]
        
        # Prediction
        dec_out = self.head(enc_out)  # [bs, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs, pred_len, nvars]
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Normalization for imputation
        means = torch.sum(x_enc, dim=1) / torch.sum(mask == 1, dim=1)
        means = means.unsqueeze(1).detach()
        x_enc = x_enc - means
        x_enc = x_enc.masked_fill(mask == 0, 0)
        stdev = torch.sqrt(torch.sum(x_enc * x_enc, dim=1) / torch.sum(mask == 1, dim=1) + 1e-5)
        stdev = stdev.unsqueeze(1).detach()
        x_enc /= stdev
        
        # Patch embedding
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        # Exogenous embedding
        ex_embed = None
        if x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)
        
        # Encoder
        enc_out, attns = self.encoder(enc_out, ex_embed)
        
        # Reshape and predict
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return dec_out

    def anomaly_detection(self, x_enc):
        # Similar to forecast but for anomaly detection
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        enc_out, attns = self.encoder(enc_out)
        
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)
        
        dec_out = self.head(enc_out)
        dec_out = dec_out.permute(0, 2, 1)
        
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.seq_len, 1))
        
        return dec_out

    def classification(self, x_enc, x_mark_enc):
        # Classification task
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        x_enc = x_enc.permute(0, 2, 1)
        enc_out, n_vars = self.patch_embedding(x_enc)
        
        # Exogenous embedding for classification
        ex_embed = None
        if x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)
        
        enc_out, attns = self.encoder(enc_out, ex_embed)
        
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