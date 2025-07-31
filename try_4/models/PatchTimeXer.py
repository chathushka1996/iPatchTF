import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.SelfAttention_Family import FullAttention, AttentionLayer
from layers.Transformer_EncDec import Encoder, EncoderLayer
from layers.Embed import DataEmbedding_inverted, PositionalEmbedding
from layers.Autoformer_EncDec import series_decomp
import numpy as np


class MultiScalePatchEmbedding(nn.Module):
    """
    Multi-scale patching inspired by PatchTST and TimeMixer
    """
    def __init__(self, d_model, patch_len=16, stride=8, scales=[1, 2, 4], dropout=0.1):
        super(MultiScalePatchEmbedding, self).__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.scales = scales
        self.d_model = d_model
        
        # Multi-scale patch embeddings
        self.patch_embeddings = nn.ModuleList([
            nn.Sequential(
                nn.Linear(patch_len, d_model, bias=False),
                nn.LayerNorm(d_model),
                nn.Dropout(dropout)
            ) for _ in scales
        ])
        
        # Scale mixing layer
        self.scale_mixer = nn.Sequential(
            nn.Linear(d_model * len(scales), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [B, N, T] where B=batch, N=variables, T=time
        B, N, T = x.shape
        
        multi_scale_patches = []
        
        for i, scale in enumerate(self.scales):
            # Downsample for different scales
            if scale > 1:
                x_scaled = F.avg_pool1d(x, kernel_size=scale, stride=scale)
            else:
                x_scaled = x
                
            # Create patches
            patches = x_scaled.unfold(dimension=-1, size=self.patch_len, step=self.stride)
            # Reshape: [B, N, num_patches, patch_len]
            patches = patches.reshape(B * N, patches.shape[2], self.patch_len)
            
            # Embed patches
            patch_emb = self.patch_embeddings[i](patches)
            multi_scale_patches.append(patch_emb)
        
        # Concatenate multi-scale features
        if len(multi_scale_patches) > 1:
            # Align patch dimensions by taking minimum
            min_patches = min(p.shape[1] for p in multi_scale_patches)
            aligned_patches = [p[:, :min_patches, :] for p in multi_scale_patches]
            combined = torch.cat(aligned_patches, dim=-1)
            mixed = self.scale_mixer(combined)
        else:
            mixed = multi_scale_patches[0]
            
        return mixed, N


class SeasonTrendDecomposer(nn.Module):
    """
    Seasonal-trend decomposition inspired by TimeMixer
    """
    def __init__(self, moving_avg=25):
        super(SeasonTrendDecomposer, self).__init__()
        self.decomp = series_decomp(moving_avg)
        
    def forward(self, x):
        seasonal, trend = self.decomp(x)
        return seasonal, trend


class InvertedAttentionLayer(nn.Module):
    """
    Inverted attention mechanism inspired by iTransformer
    """
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(InvertedAttentionLayer, self).__init__()
        self.attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, n_heads
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Apply inverted attention (across variables)
        # x: [B*N, L, D] -> transpose to [B*L, N, D] for variable-wise attention
        B_N, L, D = x.shape
        x_inv = x.view(-1, L, D).transpose(0, 1).contiguous()  # [L, B*N, D]
        x_inv = x_inv.view(L, -1, D)  # [L, B*N, D]
        
        # Self-attention across variables
        attn_out, _ = self.attention(x_inv, x_inv, x_inv)
        x_inv = self.norm1(x_inv + attn_out)
        
        # FFN
        ffn_out = self.ffn(x_inv)
        x_inv = self.norm2(x_inv + ffn_out)
        
        # Reshape back
        x_out = x_inv.view(L, B_N // L, L, D).transpose(0, 1).contiguous()
        x_out = x_out.view(B_N, L, D)
        
        return x_out


class GlobalContextLayer(nn.Module):
    """
    Global context mechanism inspired by TimeXer
    """
    def __init__(self, n_vars, d_model, dropout=0.1):
        super(GlobalContextLayer, self).__init__()
        self.n_vars = n_vars
        self.global_token = nn.Parameter(torch.randn(1, 1, d_model))
        self.cross_attention = AttentionLayer(
            FullAttention(False, attention_dropout=dropout, output_attention=False),
            d_model, 8  # Fixed number of heads for global context
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, patch_emb, ext_emb=None):
        B_N, L, D = patch_emb.shape
        B = B_N // self.n_vars
        
        # Add global token
        global_tokens = self.global_token.repeat(B_N, 1, 1)
        patch_with_global = torch.cat([patch_emb, global_tokens], dim=1)
        
        # Cross-attention with external features if available
        if ext_emb is not None:
            ext_emb_expanded = ext_emb.repeat(self.n_vars, 1, 1)
            global_context, _ = self.cross_attention(
                global_tokens, ext_emb_expanded, ext_emb_expanded
            )
            global_tokens = self.norm(global_tokens + self.dropout(global_context))
            patch_with_global = torch.cat([patch_emb, global_tokens], dim=1)
        
        return patch_with_global


class AdaptiveForecastHead(nn.Module):
    """
    Adaptive forecasting head with multi-scale prediction
    """
    def __init__(self, n_vars, d_model, patch_num, pred_len, dropout=0.1):
        super(AdaptiveForecastHead, self).__init__()
        self.n_vars = n_vars
        self.pred_len = pred_len
        
        # Multi-scale prediction layers
        self.seasonal_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(d_model * (patch_num + 1), pred_len),
            nn.Dropout(dropout)
        )
        
        self.trend_head = nn.Sequential(
            nn.Flatten(start_dim=-2),
            nn.Linear(d_model * (patch_num + 1), pred_len),
            nn.Dropout(dropout)
        )
        
        # Adaptive weighting
        self.weight_net = nn.Sequential(
            nn.Linear(d_model * (patch_num + 1), 64),
            nn.ReLU(),
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, seasonal_emb, trend_emb):
        # seasonal_emb, trend_emb: [B, N, D, L+1] (including global token)
        B, N, D, L_plus = seasonal_emb.shape
        
        # Reshape for processing
        seasonal_flat = seasonal_emb.view(B, N, -1)
        trend_flat = trend_emb.view(B, N, -1)
        
        # Predictions
        seasonal_pred = self.seasonal_head(seasonal_flat)  # [B, N, pred_len]
        trend_pred = self.trend_head(trend_flat)  # [B, N, pred_len]
        
        # Adaptive weighting
        combined_emb = (seasonal_flat + trend_flat) / 2
        weights = self.weight_net(combined_emb)  # [B, N, 2]
        
        # Weighted combination
        final_pred = (weights[:, :, 0:1] * seasonal_pred + 
                     weights[:, :, 1:2] * trend_pred)
        
        return final_pred.transpose(1, 2)  # [B, pred_len, N]


class Model(nn.Module):
    """
    PatchTimeXer: A hybrid model combining the best features of 
    TimeXer, PatchTST, iTransformer, and TimeMixer
    """
    
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.n_vars = configs.enc_in
        self.use_norm = getattr(configs, 'use_norm', True)
        
        # Model hyperparameters
        self.patch_len = getattr(configs, 'patch_len', 16)
        self.stride = getattr(configs, 'stride', 8)
        self.d_model = configs.d_model
        self.scales = getattr(configs, 'scales', [1, 2, 4])
        
        # Calculate patch number
        self.patch_num = (self.seq_len - self.patch_len) // self.stride + 1
        
        # Core components
        self.decomposer = SeasonTrendDecomposer(
            moving_avg=getattr(configs, 'moving_avg', 25)
        )
        
        self.patch_embedding = MultiScalePatchEmbedding(
            d_model=self.d_model,
            patch_len=self.patch_len,
            stride=self.stride,
            scales=self.scales,
            dropout=configs.dropout
        )
        
        # External feature embedding (for exogenous variables)
        self.ext_embedding = DataEmbedding_inverted(
            self.seq_len, self.d_model, configs.embed, configs.freq, configs.dropout
        )
        
        # Inverted attention layers
        self.inv_attention_layers = nn.ModuleList([
            InvertedAttentionLayer(self.d_model, configs.n_heads, configs.dropout)
            for _ in range(configs.e_layers)
        ])
        
        # Global context layers
        self.global_context = GlobalContextLayer(
            self.n_vars, self.d_model, configs.dropout
        )
        
        # Standard transformer encoder for temporal patterns
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, configs.factor, 
                                    attention_dropout=configs.dropout,
                                    output_attention=False),
                        self.d_model, configs.n_heads
                    ),
                    self.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(self.d_model)
        )
        
        # Adaptive forecasting head
        self.forecast_head = AdaptiveForecastHead(
            self.n_vars, self.d_model, self.patch_num, 
            self.pred_len, configs.dropout
        )

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Normalization
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev
        
        # Seasonal-trend decomposition
        seasonal, trend = self.decomposer(x_enc)
        
        # Process seasonal and trend components separately
        seasonal_out = self._process_component(seasonal, x_mark_enc)
        trend_out = self._process_component(trend, x_mark_enc)
        
        # Adaptive forecasting
        dec_out = self.forecast_head(seasonal_out, trend_out)
        
        # De-normalization
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out
    
    def _process_component(self, x, x_mark):
        # Multi-scale patch embedding
        x_patches, n_vars = self.patch_embedding(x.permute(0, 2, 1))
        
        # External feature embedding
        if x_mark is not None:
            ext_emb = self.ext_embedding(x, x_mark)
        else:
            ext_emb = None
        
        # Global context integration
        x_with_global = self.global_context(x_patches, ext_emb)
        
        # Inverted attention for variable relationships
        for inv_layer in self.inv_attention_layers:
            x_with_global = inv_layer(x_with_global)
        
        # Standard transformer encoding for temporal patterns
        enc_out, _ = self.encoder(x_with_global)
        
        # Reshape for forecasting head
        enc_out = enc_out.view(-1, n_vars, enc_out.shape[-2], enc_out.shape[-1])
        enc_out = enc_out.permute(0, 1, 3, 2)  # [B, N, D, L]
        
        return enc_out

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)
            return dec_out[:, -self.pred_len:, :]
        else:
            raise NotImplementedError("Only forecasting tasks are currently supported") 