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


class MultiScalePatchEmbedding(nn.Module):
    """Enhanced multi-scale patch embedding with adaptive patch lengths"""
    def __init__(self, n_vars, d_model, patch_lens=[8, 16, 24], stride=8, padding=8, dropout=0.1):
        super(MultiScalePatchEmbedding, self).__init__()
        self.patch_lens = patch_lens
        self.stride = stride
        self.padding = padding
        self.n_scales = len(patch_lens)
        
        # Multiple patch embeddings for different scales
        self.patch_embeddings = nn.ModuleList([
            nn.Linear(patch_len, d_model // self.n_scales, bias=False) 
            for patch_len in patch_lens
        ])
        
        # Scale-specific global tokens
        self.global_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, n_vars, 1, d_model // self.n_scales))
            for _ in range(self.n_scales)
        ])
        
        # Cross-scale attention for token fusion
        self.scale_fusion = nn.MultiheadAttention(d_model, num_heads=8, dropout=dropout, batch_first=True)
        
        # Enhanced positional encoding with scale awareness
        self.position_embedding = PositionalEmbedding(d_model)
        self.scale_embedding = nn.Embedding(self.n_scales, d_model)
        
        # Adaptive patch length selection
        self.patch_selector = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, self.n_scales),
            nn.Softmax(dim=-1)
        )
        
        # Enhanced normalization
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [bs, nvars, seq_len]
        batch_size, n_vars, seq_len = x.shape
        
        # Apply padding
        if self.padding:
            x = F.pad(x, (0, self.padding), mode='replicate')
        
        scale_embeddings = []
        total_patches = 0
        
        # Process each scale
        for scale_idx, (patch_len, patch_embed) in enumerate(zip(self.patch_lens, self.patch_embeddings)):
            # Create patches for this scale
            x_patch = x.unfold(dimension=-1, size=patch_len, step=self.stride)
            # x_patch: [bs, nvars, patch_num, patch_len]
            
            curr_patch_num = x_patch.shape[2]
            total_patches = max(total_patches, curr_patch_num)
            
            # Reshape and embed
            x_patch = x_patch.reshape(-1, curr_patch_num, patch_len)
            x_embedded = patch_embed(x_patch)  # [bs*nvars, patch_num, d_model//n_scales]
            
            # Add scale embedding
            scale_emb = self.scale_embedding(torch.tensor(scale_idx, device=x.device))
            x_embedded = x_embedded + scale_emb.unsqueeze(0).unsqueeze(0)
            
            # Reshape back
            x_embedded = x_embedded.reshape(batch_size, n_vars, curr_patch_num, -1)
            
            # Add global token for this scale
            global_token = self.global_tokens[scale_idx].repeat(batch_size, 1, 1, 1)
            x_embedded = torch.cat([x_embedded, global_token], dim=2)
            
            # Pad to same length if needed
            if curr_patch_num + 1 < total_patches + 1:
                padding_size = (total_patches + 1) - (curr_patch_num + 1)
                x_embedded = F.pad(x_embedded, (0, 0, 0, padding_size))
            
            scale_embeddings.append(x_embedded)
        
        # Concatenate across scales
        multi_scale_embed = torch.cat(scale_embeddings, dim=-1)  # [bs, nvars, max_patches+1, d_model]
        
        # Reshape for cross-scale attention
        multi_scale_embed = multi_scale_embed.reshape(-1, total_patches + 1, multi_scale_embed.shape[-1])
        
        # Apply cross-scale fusion
        fused_embed, _ = self.scale_fusion(multi_scale_embed, multi_scale_embed, multi_scale_embed)
        
        # Add positional encoding
        fused_embed = fused_embed + self.position_embedding(fused_embed)
        
        # Apply normalization and dropout
        fused_embed = self.layer_norm(fused_embed)
        fused_embed = self.dropout(fused_embed)
        
        return fused_embed, n_vars


class FrequencyEnhancedAttention(nn.Module):
    """Frequency-enhanced attention mechanism inspired by FEDformer"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super(FrequencyEnhancedAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        
        # Standard attention components
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Frequency domain components
        self.freq_proj = nn.Linear(d_model, d_model)
        self.freq_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
        # Learnable frequency filters
        self.freq_filter = nn.Parameter(torch.randn(d_model // 2 + 1))
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)

    def forward(self, x, attn_mask=None):
        batch_size, seq_len, d_model = x.shape
        
        # Standard attention computation
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
            attn_scores = attn_scores.masked_fill(attn_mask == 0, -1e9)
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Standard attention output
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Frequency domain enhancement
        # Convert to frequency domain
        x_freq = torch.fft.rfft(x, dim=1)
        
        # Apply learnable frequency filter
        freq_filter = self.freq_filter.unsqueeze(0).unsqueeze(-1)
        x_freq_filtered = x_freq * freq_filter
        
        # Convert back to time domain
        x_freq_enhanced = torch.fft.irfft(x_freq_filtered, n=seq_len, dim=1)
        
        # Project frequency features
        freq_features = self.freq_proj(x_freq_enhanced)
        
        # Gated fusion of time and frequency features
        gate = self.freq_gate(x)
        enhanced_output = gate * attn_output + (1 - gate) * freq_features
        
        return self.out_proj(enhanced_output), attn_weights


class HierarchicalGlobalToken(nn.Module):
    """Hierarchical global tokens for multi-level global context modeling"""
    def __init__(self, n_vars, d_model, n_levels=3):
        super(HierarchicalGlobalToken, self).__init__()
        self.n_levels = n_levels
        self.n_vars = n_vars
        
        # Hierarchical global tokens
        self.global_tokens = nn.ParameterList([
            nn.Parameter(torch.randn(1, n_vars, 1, d_model))
            for _ in range(n_levels)
        ])
        
        # Level-specific transformations
        self.level_transforms = nn.ModuleList([
            nn.Linear(d_model, d_model) for _ in range(n_levels)
        ])
        
        # Hierarchical aggregation
        self.level_attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        
        # Final projection
        self.output_proj = nn.Linear(d_model * n_levels, d_model)

    def forward(self, x, patches_per_level):
        # x: [bs*nvars, seq_len, d_model]
        batch_size_vars = x.shape[0]
        batch_size = batch_size_vars // self.n_vars
        
        level_outputs = []
        
        for level in range(self.n_levels):
            # Get global token for this level
            global_token = self.global_tokens[level].repeat(batch_size, 1, 1, 1)
            global_token = global_token.reshape(batch_size_vars, 1, -1)
            
            # Apply level-specific transformation
            global_token = self.level_transforms[level](global_token)
            
            # Attend to appropriate patches for this level
            start_idx = sum(patches_per_level[:level]) if level > 0 else 0
            end_idx = start_idx + patches_per_level[level]
            
            level_patches = x[:, start_idx:end_idx, :]
            
            # Cross-attention between global token and level patches
            enhanced_global, _ = self.level_attention(global_token, level_patches, level_patches)
            
            level_outputs.append(enhanced_global)
        
        # Concatenate and project
        hierarchical_output = torch.cat(level_outputs, dim=-1)
        return self.output_proj(hierarchical_output)


class AdaptiveNormalization(nn.Module):
    """Adaptive normalization that learns distribution-specific parameters"""
    def __init__(self, d_model, eps=1e-5):
        super(AdaptiveNormalization, self).__init__()
        self.eps = eps
        self.d_model = d_model
        
        # Learnable parameters for adaptive normalization
        self.alpha = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        
        # Distribution estimation network
        self.dist_estimator = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 2)  # mean and std
        )
        
        # Adaptation network
        self.adaptation = nn.Sequential(
            nn.Linear(2, d_model),
            nn.Tanh()
        )

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        
        # Estimate distribution parameters
        dist_params = self.dist_estimator(x.mean(dim=1))  # [batch, 2]
        adaptation_weights = self.adaptation(dist_params)  # [batch, d_model]
        
        # Standard normalization
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        normalized = (x - mean) / (std + self.eps)
        
        # Apply adaptive parameters
        adaptive_alpha = self.alpha.unsqueeze(0).unsqueeze(0) * (1 + adaptation_weights.unsqueeze(1))
        adaptive_beta = self.beta.unsqueeze(0).unsqueeze(0) + adaptation_weights.unsqueeze(1) * 0.1
        
        return normalized * adaptive_alpha + adaptive_beta


class EnhancedHybridEncoderLayer(nn.Module):
    """Enhanced encoder layer with frequency attention and hierarchical global tokens"""
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="gelu", n_vars=1):
        super(EnhancedHybridEncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.n_vars = n_vars
        
        # Frequency-enhanced attention
        self.freq_attention = FrequencyEnhancedAttention(d_model, n_heads=8, dropout=dropout)
        
        # Hierarchical global token processing
        self.hierarchical_global = HierarchicalGlobalToken(n_vars, d_model)
        
        # Enhanced feed-forward with gating
        self.ffn_gate = nn.Sequential(
            nn.Linear(d_model, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model),
            nn.Sigmoid()
        )
        
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        
        # Advanced normalization
        self.norm1 = AdaptiveNormalization(d_model)
        self.norm2 = AdaptiveNormalization(d_model)
        self.norm3 = AdaptiveNormalization(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.activation = F.gelu if activation == "gelu" else F.relu

    def forward(self, x, cross=None, x_mask=None, cross_mask=None, tau=None, delta=None):
        # Enhanced self-attention with frequency components
        freq_attn_out, _ = self.freq_attention(x, x_mask)
        x = x + self.dropout(freq_attn_out)
        x = self.norm1(x)
        
        # Standard self-attention
        std_attn_out = self.self_attention(x, x, x, attn_mask=x_mask, tau=tau, delta=delta)[0]
        x = x + self.dropout(std_attn_out)
        x = self.norm2(x)
        
        # Cross-attention with exogenous features (if available)
        if cross is not None:
            B = cross.shape[0]
            L, D = x.shape[1], x.shape[2]
            
            # Extract and enhance global token with hierarchical processing
            x_glb_ori = x[:, -1, :].unsqueeze(1)  # [bs*nvars, 1, d_model]
            
            # Apply hierarchical global token processing
            patches_per_level = [L//3, L//3, L - 2*(L//3)]  # Divide patches into 3 levels
            x_glb_enhanced = self.hierarchical_global(x, patches_per_level)
            
            # Reshape for cross-attention
            x_glb = x_glb_ori.reshape(B, -1, D)  # [bs, nvars, d_model]
            
            # Cross-attention with exogenous features
            x_glb_attn = self.cross_attention(x_glb, cross, cross, 
                                            attn_mask=cross_mask, tau=tau, delta=delta)[0]
            x_glb_attn = x_glb_attn.reshape(-1, 1, D)  # [bs*nvars, 1, d_model]
            
            # Combine original, enhanced, and cross-attended global tokens
            x_glb_final = x_glb_ori + self.dropout(x_glb_attn) + 0.1 * x_glb_enhanced
            x_glb_final = self.norm3(x_glb_final)
            
            # Replace global token in sequence
            x = torch.cat([x[:, :-1, :], x_glb_final], dim=1)
        
        # Enhanced feed-forward with gating
        y = x
        gate = self.ffn_gate(y)
        
        y_ff = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y_ff = self.dropout(self.conv2(y_ff).transpose(-1, 1))
        
        # Apply gating
        y_gated = gate * y_ff + (1 - gate) * y
        
        return self.norm4(x + y_gated)


class EnhancedPredictionHead(nn.Module):
    """Enhanced prediction head with multi-path processing"""
    def __init__(self, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        self.n_vars = n_vars
        self.target_window = target_window
        
        # Multi-path processing
        self.global_path = nn.Sequential(
            nn.Linear(nf, nf // 2),
            nn.ReLU(),
            nn.Dropout(head_dropout),
            nn.Linear(nf // 2, target_window)
        )
        
        self.local_path = nn.Sequential(
            nn.Linear(nf, nf // 2),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(nf // 2, target_window)
        )
        
        # Path fusion
        self.path_fusion = nn.Sequential(
            nn.Linear(target_window * 2, target_window),
            nn.Tanh()
        )
        
        # Final refinement
        self.refinement = nn.Sequential(
            nn.Linear(target_window, target_window),
            nn.Dropout(head_dropout)
        )
        
        self.flatten = nn.Flatten(start_dim=-2)

    def forward(self, x):  # x: [bs x nvars x d_model x patch_num]
        x = self.flatten(x)
        
        # Multi-path processing
        global_out = self.global_path(x)
        local_out = self.local_path(x)
        
        # Fuse paths
        fused = torch.cat([global_out, local_out], dim=-1)
        fused = self.path_fusion(fused)
        
        # Final refinement
        output = self.refinement(fused)
        
        return output


class Model(nn.Module):
    """
    Enhanced PatchTimeXer with Advanced Features:
    
    1. Multi-scale patch embedding with adaptive patch lengths
    2. Frequency-enhanced attention mechanisms
    3. Hierarchical global tokens for multi-level context
    4. Adaptive normalization for better stationarity handling
    5. Enhanced prediction head with multi-path processing
    6. Advanced cross-attention mechanisms
    """

    def __init__(self, configs, patch_lens=[8, 16, 24], stride=8):
        super().__init__()
        self.task_name = configs.task_name
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.patch_lens = patch_lens
        self.stride = stride
        self.n_vars = configs.enc_in
        
        # Multi-scale patch embedding
        padding = stride
        self.patch_embedding = MultiScalePatchEmbedding(
            self.n_vars, configs.d_model, patch_lens, stride, padding, configs.dropout)
        
        # Exogenous embedding with enhancement
        self.ex_embedding = DataEmbedding_inverted(
            configs.seq_len, configs.d_model, configs.embed, configs.freq, configs.dropout)
        
        # Calculate effective patch number (using largest patch size)
        self.patch_num = int((configs.seq_len - max(patch_lens)) / stride + 2)
        
        # Enhanced encoder with advanced layers
        self.encoder = nn.ModuleList([
            EnhancedHybridEncoderLayer(
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model, configs.n_heads),
                AttentionLayer(
                    FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                  output_attention=False), configs.d_model, configs.n_heads),
                configs.d_model,
                configs.d_ff,
                dropout=configs.dropout,
                activation=configs.activation,
                n_vars=self.n_vars
            ) for l in range(configs.e_layers)
        ])
        
        # Final normalization
        self.final_norm = AdaptiveNormalization(configs.d_model)
        
        # Enhanced prediction head
        self.head_nf = configs.d_model * (self.patch_num + 1)
        
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            self.head = EnhancedPredictionHead(configs.enc_in, self.head_nf, configs.pred_len,
                                             head_dropout=configs.dropout)
        elif self.task_name == 'imputation' or self.task_name == 'anomaly_detection':
            self.head = EnhancedPredictionHead(configs.enc_in, self.head_nf, configs.seq_len,
                                             head_dropout=configs.dropout)
        elif self.task_name == 'classification':
            self.flatten = nn.Flatten(start_dim=-2)
            self.dropout = nn.Dropout(configs.dropout)
            self.projection = nn.Linear(self.head_nf * configs.enc_in, configs.num_class)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        # Enhanced normalization with adaptive statistics
        means = x_enc.mean(1, keepdim=True).detach()
        x_enc = x_enc - means
        stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= stdev
        
        # Multi-scale patch embedding
        x_enc = x_enc.permute(0, 2, 1)  # [bs, nvars, seq_len]
        enc_out, n_vars = self.patch_embedding(x_enc)  # [bs*nvars, patch_num+1, d_model]
        
        # Enhanced exogenous embedding
        ex_embed = None
        if x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)  # [bs, seq_len, d_model]
        
        # Enhanced encoder processing
        for layer in self.encoder:
            enc_out = layer(enc_out, ex_embed)
        
        # Final normalization
        enc_out = self.final_norm(enc_out)
        
        # Reshape for prediction head
        enc_out = torch.reshape(enc_out, (-1, n_vars, enc_out.shape[-2], enc_out.shape[-1]))
        enc_out = enc_out.permute(0, 1, 3, 2)  # [bs, nvars, d_model, patch_num+1]
        
        # Enhanced prediction
        dec_out = self.head(enc_out)  # [bs, nvars, pred_len]
        dec_out = dec_out.permute(0, 2, 1)  # [bs, pred_len, nvars]
        
        # De-normalization
        dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        
        return dec_out

    def imputation(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask):
        # Enhanced normalization for imputation
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
        
        ex_embed = None
        if x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)
        
        for layer in self.encoder:
            enc_out = layer(enc_out, ex_embed)
        
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
        # Same enhancement as forecast
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
        
        ex_embed = None
        if x_mark_enc is not None:
            ex_embed = self.ex_embedding(x_enc.permute(0, 2, 1), x_mark_enc)
        
        for layer in self.encoder:
            enc_out = layer(enc_out, ex_embed)
        
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