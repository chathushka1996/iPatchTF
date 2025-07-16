import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any
import math

# Import our custom components
from models.autoformer import Autoformer
from models.patchtst import EnhancedPatchTST

class FusionConfig:
    """Configuration for the hybrid model"""
    def __init__(self, base_config):
        # Copy all base configuration attributes
        self.__dict__.update(vars(base_config))
        
        # Autoformer specific
        self.enc_in = len(base_config.feature_cols) + len(base_config.time_cols)
        self.dec_in = len(base_config.feature_cols) + len(base_config.time_cols)
        self.c_out = 1  # Univariate prediction
        self.activation = 'gelu'
        
        # Fusion specific parameters
        self.fusion_hidden_dim = 256
        self.attention_hidden_dim = 128
        self.num_experts = 4
        
        # Additional components
        self.use_wavelet = True
        self.use_residual_connection = True
        self.use_uncertainty_estimation = True

class WaveletDecomposition(nn.Module):
    """
    Simple wavelet-like decomposition using learnable filters
    """
    def __init__(self, seq_len, num_levels=3):
        super().__init__()
        self.seq_len = seq_len
        self.num_levels = num_levels
        
        # Learnable decomposition filters
        self.low_pass_filters = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=4, stride=2, padding=1)
            for _ in range(num_levels)
        ])
        
        self.high_pass_filters = nn.ModuleList([
            nn.Conv1d(1, 1, kernel_size=4, stride=2, padding=1)
            for _ in range(num_levels)
        ])
        
        # Reconstruction layers
        self.reconstruction = nn.ModuleList([
            nn.ConvTranspose1d(1, 1, kernel_size=4, stride=2, padding=1)
            for _ in range(num_levels)
        ])

    def forward(self, x):
        """
        x: [B, L, C] -> decompose last channel (target)
        """
        B, L, C = x.shape
        target = x[:, :, -1:].transpose(1, 2)  # [B, 1, L]
        
        components = []
        current = target
        
        for i in range(self.num_levels):
            # Decompose
            low = self.low_pass_filters[i](current)
            high = self.high_pass_filters[i](current)
            
            components.append(high)
            current = low
        
        components.append(current)  # Final low-frequency component
        
        return components

class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention mechanism for different temporal scales
    """
    def __init__(self, d_model, scales=[1, 2, 4, 8]):
        super().__init__()
        self.scales = scales
        self.d_model = d_model
        
        self.scale_attentions = nn.ModuleList([
            nn.MultiheadAttention(d_model, num_heads=8, batch_first=True)
            for _ in scales
        ])
        
        self.scale_projections = nn.ModuleList([
            nn.Linear(d_model, d_model)
            for _ in scales
        ])
        
        self.fusion = nn.Linear(len(scales) * d_model, d_model)

    def forward(self, x):
        """
        x: [B, L, D]
        """
        B, L, D = x.shape
        scale_outputs = []
        
        for i, scale in enumerate(self.scales):
            # Downsample
            if scale > 1:
                x_scaled = F.avg_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale).transpose(1, 2)
            else:
                x_scaled = x
            
            # Self-attention
            attn_out, _ = self.scale_attentions[i](x_scaled, x_scaled, x_scaled)
            attn_out = self.scale_projections[i](attn_out)
            
            # Upsample back
            if scale > 1:
                attn_out = F.interpolate(attn_out.transpose(1, 2), size=L, mode='linear', align_corners=False).transpose(1, 2)
            
            scale_outputs.append(attn_out)
        
        # Fuse multi-scale features
        fused = torch.cat(scale_outputs, dim=-1)
        output = self.fusion(fused)
        
        return output

class ExpertMixture(nn.Module):
    """
    Mixture of Experts for adaptive fusion
    """
    def __init__(self, input_dim, hidden_dim, num_experts, pred_len):
        super().__init__()
        self.num_experts = num_experts
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_dim, pred_len)
            )
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_experts),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        """
        x: [B, input_dim]
        """
        # Get expert predictions
        expert_outputs = []
        for expert in self.experts:
            expert_outputs.append(expert(x))
        expert_outputs = torch.stack(expert_outputs, dim=-1)  # [B, pred_len, num_experts]
        
        # Get gating weights
        gates = self.gating(x)  # [B, num_experts]
        gates = gates.unsqueeze(1)  # [B, 1, num_experts]
        
        # Weighted combination
        output = torch.sum(expert_outputs * gates, dim=-1)  # [B, pred_len]
        
        return output, gates.squeeze(1)

class UncertaintyEstimator(nn.Module):
    """
    Uncertainty estimation using ensemble-like approach
    """
    def __init__(self, input_dim, pred_len):
        super().__init__()
        self.mean_head = nn.Linear(input_dim, pred_len)
        self.log_var_head = nn.Linear(input_dim, pred_len)

    def forward(self, x):
        mean = self.mean_head(x)
        log_var = self.log_var_head(x)
        std = torch.exp(0.5 * log_var)
        return mean, std

class AdaptiveFusion(nn.Module):
    """
    Adaptive fusion mechanism with attention
    """
    def __init__(self, num_models, d_model):
        super().__init__()
        self.num_models = num_models
        
        # Attention for fusion weights
        self.attention = nn.MultiheadAttention(d_model, num_heads=4, batch_first=True)
        self.fusion_weights = nn.Linear(d_model, num_models)
        self.output_projection = nn.Linear(d_model, 1)

    def forward(self, model_outputs, features):
        """
        model_outputs: List of [B, pred_len, 1] from different models
        features: [B, pred_len, D] contextual features
        """
        B, pred_len, _ = model_outputs[0].shape
        
        # Stack model outputs
        stacked_outputs = torch.cat(model_outputs, dim=-1)  # [B, pred_len, num_models]
        
        # Attention-based fusion
        attn_out, _ = self.attention(features, features, features)
        fusion_weights = torch.softmax(self.fusion_weights(attn_out), dim=-1)  # [B, pred_len, num_models]
        
        # Weighted combination
        fused_output = torch.sum(stacked_outputs * fusion_weights, dim=-1, keepdim=True)  # [B, pred_len, 1]
        
        return fused_output

class HybridSolarModel(nn.Module):
    """
    Hybrid model combining Autoformer, PatchTST, and statistical features
    """
    def __init__(self, config):
        super().__init__()
        self.config = FusionConfig(config)
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
        
        # We'll set enc_in dynamically in forward()
        self.d_model = getattr(self.config, 'd_model', 512)
        self.fusion_hidden_dim = getattr(self.config, 'fusion_hidden_dim', 256)
        self.num_experts = getattr(self.config, 'num_experts', 4)
        self.use_wavelet = getattr(self.config, 'use_wavelet', True)
        self.use_uncertainty = getattr(self.config, 'use_uncertainty_estimation', True)

        # Core models
        self.autoformer = self._create_autoformer()
        self.patchtst = EnhancedPatchTST(config)
        
        # Additional components
        if self.use_wavelet:
            self.wavelet_decomp = WaveletDecomposition(self.seq_len)
            self.wavelet_processor = nn.LSTM(1, 64, batch_first=True)
            self.wavelet_head = nn.Linear(64, self.pred_len)
        
        # Multi-scale attention (d_model will be set after feature_extractor is built)
        self.multiscale_attention = None  # Placeholder
        self.feature_extractor = None     # Will be built on first forward
        self.mixture_of_experts = None    # Will be built on first forward
        self.adaptive_fusion = None       # Will be built on first forward
        self.uncertainty_estimator = None # Will be built on first forward
        self.final_projection = None      # Will be built on first forward
        self.residual_projection = None   # Will be built on first forward

    def _create_autoformer(self):
        """Create Autoformer with proper configuration"""
        # Get dimensions with fallbacks
        enc_in = getattr(self.config, 'enc_in', 52)  # fallback based on your data
        d_model = getattr(self.config, 'd_model', 512)  # fallback value
        
        # Create a simplified Autoformer-like model
        return nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def build_layers(self, enc_in):
        d_model = self.d_model
        fusion_hidden_dim = self.fusion_hidden_dim
        num_experts = self.num_experts
        seq_len = self.seq_len
        pred_len = self.pred_len
        use_wavelet = self.use_wavelet
        use_uncertainty = self.use_uncertainty

        print(f"[HybridSolarModel] Detected input feature dimension: {enc_in}")

        self.feature_extractor = nn.Sequential(
            nn.Linear(enc_in, d_model),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model, d_model)
        )
        self.multiscale_attention = MultiScaleAttention(d_model)
        self.mixture_of_experts = ExpertMixture(
            input_dim=d_model * seq_len,
            hidden_dim=fusion_hidden_dim,
            num_experts=num_experts,
            pred_len=pred_len
        )
        num_base_models = 3 if use_wavelet else 2
        self.adaptive_fusion = AdaptiveFusion(num_base_models, d_model)
        if use_uncertainty:
            self.uncertainty_estimator = UncertaintyEstimator(d_model * seq_len, pred_len)
        self.final_projection = nn.Sequential(
            nn.Linear(pred_len, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(fusion_hidden_dim, pred_len),
        )
        self.residual_projection = nn.Linear(seq_len, pred_len)

    def forward(self, x):
        # Dynamically build layers on first forward pass
        if self.feature_extractor is None:
            enc_in = x.shape[-1]
            self.build_layers(enc_in)
            self.to(x.device)  # Ensure all layers are on the same device as input

        B, L, C = x.shape
        device = x.device
        
        features = self.feature_extractor(x)  # [B, L, d_model]
        features_flat = features.reshape(B, -1)  # [B, L * d_model]
        enhanced_features = self.multiscale_attention(features)

        model_outputs = []
        autoformer_input = x.reshape(B * L, C)
        autoformer_out = self.autoformer(autoformer_input)
        autoformer_out = autoformer_out.reshape(B, L, 1)
        autoformer_pred = F.adaptive_avg_pool1d(
            autoformer_out.transpose(1, 2),
            self.pred_len
        ).transpose(1, 2)
        model_outputs.append(autoformer_pred)
        patchtst_pred = self.patchtst(x)
        model_outputs.append(patchtst_pred)
        if self.use_wavelet:
            wavelet_components = self.wavelet_decomp(x)
            wavelet_features = []
            for component in wavelet_components:
                if component.shape[-1] > 0:
                    component_reshaped = component.transpose(1, 2)
                    lstm_out, _ = self.wavelet_processor(component_reshaped)
                    wavelet_features.append(lstm_out[:, -1, :])
            if wavelet_features:
                wavelet_concat = torch.cat(wavelet_features, dim=-1)
                if wavelet_concat.shape[-1] >= 64:
                    wavelet_pred = self.wavelet_head(wavelet_concat[:, :64]).unsqueeze(-1)
                else:
                    padding = torch.zeros(B, 64 - wavelet_concat.shape[-1], device=device)
                    wavelet_concat = torch.cat([wavelet_concat, padding], dim=-1)
                    wavelet_pred = self.wavelet_head(wavelet_concat).unsqueeze(-1)
                model_outputs.append(wavelet_pred)
        moe_pred, expert_weights = self.mixture_of_experts(features_flat)
        moe_pred = moe_pred.unsqueeze(-1)
        model_outputs.append(moe_pred)
        fusion_features = enhanced_features[:, -self.pred_len:, :]
        if fusion_features.shape[1] < self.pred_len:
            last_feature = fusion_features[:, -1:, :].repeat(1, self.pred_len - fusion_features.shape[1], 1)
            fusion_features = torch.cat([fusion_features, last_feature], dim=1)
        fused_output = self.adaptive_fusion(model_outputs, fusion_features)
        final_output = self.final_projection(fused_output.squeeze(-1)).unsqueeze(-1)
        if self.use_residual_connection:
            target_values = x[:, :, -1]
            residual = self.residual_projection(target_values).unsqueeze(-1)
            final_output = final_output + residual
        uncertainty = None
        if self.use_uncertainty:
            mean_pred, std_pred = self.uncertainty_estimator(features_flat)
            uncertainty = {
                'mean': mean_pred.unsqueeze(-1),
                'std': std_pred.unsqueeze(-1)
            }
        return {
            'prediction': final_output,
            'autoformer_pred': autoformer_pred,
            'patchtst_pred': patchtst_pred,
            'moe_pred': moe_pred,
            'expert_weights': expert_weights,
            'uncertainty': uncertainty
        }

class HybridLoss(nn.Module):
    """
    Combined loss function for the hybrid model
    """
    def __init__(self, alpha=0.7, beta=0.2, gamma=0.1):
        super().__init__()
        self.alpha = alpha  # Main prediction loss weight
        self.beta = beta    # Individual model loss weight
        self.gamma = gamma  # Uncertainty loss weight
        
        self.mse_loss = nn.MSELoss()
        self.mae_loss = nn.L1Loss()

    def forward(self, outputs, targets):
        """
        outputs: dict from HybridSolarModel
        targets: [B, pred_len, 1]
        """
        # Main prediction loss
        main_loss = self.mse_loss(outputs['prediction'], targets)
        
        # Individual model losses (for regularization)
        autoformer_loss = self.mse_loss(outputs['autoformer_pred'], targets)
        patchtst_loss = self.mse_loss(outputs['patchtst_pred'], targets)
        individual_loss = (autoformer_loss + patchtst_loss) / 2
        
        # Uncertainty loss (if available)
        uncertainty_loss = 0
        if outputs['uncertainty'] is not None:
            mean_pred = outputs['uncertainty']['mean']
            std_pred = outputs['uncertainty']['std']
            
            # Negative log-likelihood for Gaussian
            uncertainty_loss = torch.mean(
                0.5 * torch.log(2 * torch.pi * std_pred**2) + 
                0.5 * ((targets - mean_pred)**2) / (std_pred**2)
            )
        
        # Combined loss
        total_loss = (
            self.alpha * main_loss + 
            self.beta * individual_loss + 
            self.gamma * uncertainty_loss
        )
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'individual_loss': individual_loss,
            'uncertainty_loss': uncertainty_loss
        } 