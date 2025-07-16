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
        
        # Clamp log_var to prevent numerical instability
        log_var = torch.clamp(log_var, min=-10, max=3)  # Prevent extreme values
        
        # Safe exponential computation
        std = torch.exp(0.5 * log_var)
        
        # Ensure std is within reasonable bounds
        std = torch.clamp(std, min=1e-6, max=10.0)
        
        # Check for NaN and replace if necessary
        if torch.isnan(mean).any():
            print("Warning: NaN in uncertainty mean, replacing with zeros")
            mean = torch.nan_to_num(mean, nan=0.0)
        
        if torch.isnan(std).any():
            print("Warning: NaN in uncertainty std, replacing with 1.0")
            std = torch.nan_to_num(std, nan=1.0)
        
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
        self.d_model = getattr(self.config, 'd_model', 512)
        self.fusion_hidden_dim = getattr(self.config, 'fusion_hidden_dim', 256)
        self.num_experts = getattr(self.config, 'num_experts', 4)
        self.use_wavelet = getattr(self.config, 'use_wavelet', True)
        self.use_uncertainty = getattr(self.config, 'use_uncertainty_estimation', True)
        self.use_residual_connection = getattr(self.config, 'use_residual_connection', True)

        # Core models
        self.autoformer = None  # Will be built dynamically in build_layers
        self.patchtst = EnhancedPatchTST(config)

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

    def build_layers(self, enc_in):
        d_model = self.d_model
        fusion_hidden_dim = self.fusion_hidden_dim
        num_experts = self.num_experts
        seq_len = self.seq_len
        pred_len = self.pred_len
        use_wavelet = self.use_wavelet
        use_uncertainty = self.use_uncertainty

        print(f"[HybridSolarModel] Detected input feature dimension: {enc_in}")

        # Build autoformer with correct input dim
        self.autoformer = nn.Sequential(
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
        # Base models: autoformer + patchtst + moe = 3, +1 if wavelet is used
        num_base_models = 4 if use_wavelet else 3
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
        
        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights properly"""
        for module in [self.autoformer, self.feature_extractor, self.final_projection, self.residual_projection]:
            if module is not None:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        # Xavier normal initialization for better stability
                        nn.init.xavier_normal_(m.weight)
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input validation - check for NaN/inf values early
        if torch.isnan(x).any():
            print(f"Warning: NaN detected in input tensor, count: {torch.isnan(x).sum()}")
            x = torch.nan_to_num(x, nan=0.0)
        
        if torch.isinf(x).any():
            print(f"Warning: Inf detected in input tensor, count: {torch.isinf(x).sum()}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        # Dynamically build layers on first forward pass
        if self.feature_extractor is None:
            enc_in = x.shape[-1]
            self.build_layers(enc_in)
            self.to(x.device)  # Ensure all layers are on the same device as input

        B, L, C = x.shape
        device = x.device
        
        try:
            features = self.feature_extractor(x)  # [B, L, d_model]
            
            # Check features for NaN
            if torch.isnan(features).any():
                print("Warning: NaN detected in feature extraction, replacing with zeros")
                features = torch.nan_to_num(features, nan=0.0)
            
            features_flat = features.reshape(B, -1)  # [B, L * d_model]
            enhanced_features = self.multiscale_attention(features)
            
            # Check enhanced features for NaN
            if torch.isnan(enhanced_features).any():
                print("Warning: NaN detected in enhanced features, replacing with zeros")
                enhanced_features = torch.nan_to_num(enhanced_features, nan=0.0)

            model_outputs = []
            
            # Autoformer prediction with error handling
            try:
                autoformer_input = x.reshape(B * L, C)
                autoformer_out = self.autoformer(autoformer_input)
                autoformer_out = autoformer_out.reshape(B, L, 1)
                autoformer_pred = F.adaptive_avg_pool1d(
                    autoformer_out.transpose(1, 2),
                    self.pred_len
                ).transpose(1, 2)
                
                # Check for NaN in autoformer output
                if torch.isnan(autoformer_pred).any():
                    print("Warning: NaN detected in autoformer prediction, replacing with zeros")
                    autoformer_pred = torch.nan_to_num(autoformer_pred, nan=0.0)
                
                model_outputs.append(autoformer_pred)
            except Exception as e:
                print(f"Error in autoformer: {e}, using zero prediction")
                autoformer_pred = torch.zeros(B, self.pred_len, 1, device=device)
                model_outputs.append(autoformer_pred)
            
            # PatchTST prediction with error handling
            try:
                patchtst_pred = self.patchtst(x)
                
                # Check for NaN in patchtst output
                if torch.isnan(patchtst_pred).any():
                    print("Warning: NaN detected in patchtst prediction, replacing with zeros")
                    patchtst_pred = torch.nan_to_num(patchtst_pred, nan=0.0)
                
                model_outputs.append(patchtst_pred)
            except Exception as e:
                print(f"Error in patchtst: {e}, using zero prediction")
                patchtst_pred = torch.zeros(B, self.pred_len, 1, device=device)
                model_outputs.append(patchtst_pred)
            
            # Wavelet processing with error handling
            if self.use_wavelet:
                try:
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
                        
                        # Check for NaN in wavelet output
                        if torch.isnan(wavelet_pred).any():
                            print("Warning: NaN detected in wavelet prediction, replacing with zeros")
                            wavelet_pred = torch.nan_to_num(wavelet_pred, nan=0.0)
                        
                        model_outputs.append(wavelet_pred)
                except Exception as e:
                    print(f"Error in wavelet processing: {e}, skipping wavelet component")
            
            # Mixture of experts with error handling
            try:
                moe_pred, expert_weights = self.mixture_of_experts(features_flat)
                moe_pred = moe_pred.unsqueeze(-1)
                
                # Check for NaN in MoE output
                if torch.isnan(moe_pred).any():
                    print("Warning: NaN detected in MoE prediction, replacing with zeros")
                    moe_pred = torch.nan_to_num(moe_pred, nan=0.0)
                
                if torch.isnan(expert_weights).any():
                    print("Warning: NaN detected in expert weights, replacing with uniform weights")
                    expert_weights = torch.ones_like(expert_weights) / expert_weights.shape[-1]
                
                model_outputs.append(moe_pred)
            except Exception as e:
                print(f"Error in mixture of experts: {e}, using zero prediction")
                moe_pred = torch.zeros(B, self.pred_len, 1, device=device)
                expert_weights = torch.ones(B, self.num_experts, device=device) / self.num_experts
                model_outputs.append(moe_pred)
            
            # Adaptive fusion with error handling
            try:
                fusion_features = enhanced_features[:, -self.pred_len:, :]
                if fusion_features.shape[1] < self.pred_len:
                    last_feature = fusion_features[:, -1:, :].repeat(1, self.pred_len - fusion_features.shape[1], 1)
                    fusion_features = torch.cat([fusion_features, last_feature], dim=1)
                fused_output = self.adaptive_fusion(model_outputs, fusion_features)
                
                # Check for NaN in fused output
                if torch.isnan(fused_output).any():
                    print("Warning: NaN detected in fused output, replacing with zeros")
                    fused_output = torch.nan_to_num(fused_output, nan=0.0)
                
            except Exception as e:
                print(f"Error in adaptive fusion: {e}, using simple average")
                fused_output = torch.mean(torch.stack([out for out in model_outputs if out.shape == model_outputs[0].shape]), dim=0)
                if torch.isnan(fused_output).any():
                    fused_output = torch.zeros_like(model_outputs[0])
            
            # Final projection with error handling
            try:
                final_output = self.final_projection(fused_output.squeeze(-1)).unsqueeze(-1)
                
                # Check for NaN in final projection
                if torch.isnan(final_output).any():
                    print("Warning: NaN detected in final projection, replacing with zeros")
                    final_output = torch.nan_to_num(final_output, nan=0.0)
                
            except Exception as e:
                print(f"Error in final projection: {e}, using fused output")
                final_output = fused_output
                if torch.isnan(final_output).any():
                    final_output = torch.zeros(B, self.pred_len, 1, device=device)
            
            # Residual connection with error handling
            if self.use_residual_connection:
                try:
                    target_values = x[:, :, -1]
                    residual = self.residual_projection(target_values).unsqueeze(-1)
                    
                    # Check for NaN in residual
                    if torch.isnan(residual).any():
                        print("Warning: NaN detected in residual, replacing with zeros")
                        residual = torch.nan_to_num(residual, nan=0.0)
                    
                    final_output = final_output + residual
                except Exception as e:
                    print(f"Error in residual connection: {e}, skipping residual")
            
            # Uncertainty estimation with error handling
            uncertainty = None
            if self.use_uncertainty:
                try:
                    mean_pred, std_pred = self.uncertainty_estimator(features_flat)
                    uncertainty = {
                        'mean': mean_pred.unsqueeze(-1),
                        'std': std_pred.unsqueeze(-1)
                    }
                except Exception as e:
                    print(f"Error in uncertainty estimation: {e}, setting uncertainty to None")
                    uncertainty = None
            
            # Final safety check for NaN values
            if torch.isnan(final_output).any():
                print("Warning: NaN detected in final_output, replacing with zeros")
                final_output = torch.nan_to_num(final_output, nan=0.0)
            
        except Exception as e:
            print(f"Critical error in forward pass: {e}")
            print("Returning zero predictions to prevent crash")
            final_output = torch.zeros(B, self.pred_len, 1, device=device)
            autoformer_pred = torch.zeros(B, self.pred_len, 1, device=device)
            patchtst_pred = torch.zeros(B, self.pred_len, 1, device=device)
            moe_pred = torch.zeros(B, self.pred_len, 1, device=device)
            expert_weights = torch.ones(B, self.num_experts, device=device) / self.num_experts
            uncertainty = None
        
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
        # Check for NaN in inputs first
        if torch.isnan(targets).any():
            print("Warning: NaN detected in targets, replacing with zeros")
            targets = torch.nan_to_num(targets, nan=0.0)
        
        if torch.isnan(outputs['prediction']).any():
            print("Warning: NaN detected in predictions, replacing with zeros")
            outputs['prediction'] = torch.nan_to_num(outputs['prediction'], nan=0.0)
        
        # Main prediction loss
        main_loss = self.mse_loss(outputs['prediction'], targets)
        
        # Check for NaN in main loss
        if torch.isnan(main_loss):
            print("Warning: NaN in main_loss, setting to 1.0")
            main_loss = torch.tensor(1.0, device=targets.device, requires_grad=True)
        
        # Individual model losses (for regularization)
        autoformer_loss = self.mse_loss(outputs['autoformer_pred'], targets)
        patchtst_loss = self.mse_loss(outputs['patchtst_pred'], targets)
        
        # Check for NaN in individual losses
        if torch.isnan(autoformer_loss):
            autoformer_loss = torch.tensor(1.0, device=targets.device)
        if torch.isnan(patchtst_loss):
            patchtst_loss = torch.tensor(1.0, device=targets.device)
            
        individual_loss = (autoformer_loss + patchtst_loss) / 2
        
        # Uncertainty loss (if available) with numerical stability
        uncertainty_loss = 0
        if outputs['uncertainty'] is not None:
            mean_pred = outputs['uncertainty']['mean']
            std_pred = outputs['uncertainty']['std']
            
            # Ensure numerical stability
            std_pred = torch.clamp(std_pred, min=1e-6, max=10.0)  # Prevent too small/large values
            mean_pred = torch.nan_to_num(mean_pred, nan=0.0)
            
            # Safe log computation with clipping
            log_term = torch.log(2 * torch.pi * std_pred**2 + 1e-8)  # Add small epsilon
            log_term = torch.clamp(log_term, min=-10, max=10)  # Prevent extreme values
            
            # Safe division for the second term
            diff_squared = (targets - mean_pred)**2
            variance_term = diff_squared / (std_pred**2 + 1e-8)  # Add small epsilon
            variance_term = torch.clamp(variance_term, max=100)  # Prevent explosion
            
            # Combine terms safely
            uncertainty_loss = torch.mean(0.5 * log_term + 0.5 * variance_term)
            
            # Final NaN check
            if torch.isnan(uncertainty_loss):
                print("Warning: NaN in uncertainty_loss, setting to 0")
                uncertainty_loss = torch.tensor(0.0, device=targets.device)
        
        # Combined loss with numerical stability
        total_loss = (
            self.alpha * main_loss + 
            self.beta * individual_loss + 
            self.gamma * uncertainty_loss
        )
        
        # Final NaN check for total loss
        if torch.isnan(total_loss):
            print("Warning: NaN in total_loss, using only main_loss")
            total_loss = main_loss
        
        return {
            'total_loss': total_loss,
            'main_loss': main_loss,
            'individual_loss': individual_loss,
            'uncertainty_loss': uncertainty_loss
        } 