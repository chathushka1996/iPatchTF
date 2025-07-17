import torch

class Config:
    # Data parameters
    data_path = 'data/sl_t/'
    target_col = 'Solar Power Output'
    
    # Time series parameters
    seq_len = 96  # Input sequence length (24 hours = 96 * 15min intervals)
    pred_len = 96  # Prediction length (24 hours = 96 * 15min intervals)
    
    # Model parameters (simplified for better convergence)
    d_model = 128  # Much smaller model
    n_heads = 4    # Fewer attention heads
    e_layers = 1   # Single encoder layer
    d_layers = 1   # Single decoder layer
    d_ff = 256     # Smaller feed-forward dimension
    dropout = 0.3  # Higher dropout for regularization
    
    # PatchTST parameters
    patch_size = 16
    stride = 8
    
    # Autoformer parameters
    moving_avg = 25
    factor = 3
    
    # Statistical features
    rolling_windows = [12, 24, 48, 96]  # 3h, 6h, 12h, 24h
    lag_features = [1, 4, 12, 24, 48, 96]  # Various lag periods
    
    # Training parameters (much more conservative for MSE < 0.1)
    batch_size = 4  # Very small batches for stable gradients
    learning_rate = 1e-6  # Much smaller learning rate
    num_epochs = 200  # More epochs for slow learning
    patience = 50  # Much more patience
    
    # Gradient clipping and regularization (more aggressive)
    max_grad_norm = 0.05  # Very aggressive gradient clipping
    weight_decay = 5e-3  # Stronger weight decay
    
    # Loss function weights (focus almost entirely on main loss)
    loss_alpha = 0.95  # Almost all weight on main loss
    loss_beta = 0.04   # Minimal individual loss weight  
    loss_gamma = 0.01  # Minimal uncertainty loss weight
    
    # Numerical stability parameters
    eps = 1e-8  # Small epsilon for numerical stability
    min_std = 1e-6  # Minimum standard deviation for uncertainty
    max_std = 5.0  # Maximum standard deviation for uncertainty
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Features (excluding target and time columns)
    feature_cols = ['temp', 'dew', 'humidity', 'winddir', 'windspeed', 'pressure', 'cloudcover']
    time_cols = ['dayofyear', 'timeofday'] 
    
    # Model complexity control (for MSE < 0.1 target)
    use_simple_mode = True    # Use simplified architecture
    use_wavelet = False       # Disable wavelet (complex)
    use_uncertainty = False   # Disable uncertainty (complex)  
    use_residual_connection = True  # Keep residual (helpful)
    
    # Hybrid model components (simplified)
    num_experts = 2          # Fewer experts (was 4)
    fusion_hidden_dim = 64   # Smaller fusion network 