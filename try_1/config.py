import torch

class Config:
    # Data parameters
    data_path = 'data/sl_t/'
    target_col = 'Solar Power Output'
    
    # Time series parameters
    seq_len = 96  # Input sequence length (24 hours = 96 * 15min intervals)
    pred_len = 96  # Prediction length (24 hours = 96 * 15min intervals)
    
    # Model parameters (reduced for stability)
    d_model = 256  # Reduced from 512 for better stability
    n_heads = 8
    e_layers = 2  # Reduced from 3 for stability
    d_layers = 1  # Reduced from 2 for stability
    d_ff = 1024  # Reduced from 2048 for stability
    dropout = 0.2  # Increased dropout for regularization
    
    # PatchTST parameters
    patch_size = 16
    stride = 8
    
    # Autoformer parameters
    moving_avg = 25
    factor = 3
    
    # Statistical features
    rolling_windows = [12, 24, 48, 96]  # 3h, 6h, 12h, 24h
    lag_features = [1, 4, 12, 24, 48, 96]  # Various lag periods
    
    # Training parameters (more conservative)
    batch_size = 8  # Further reduced for more stable gradients
    learning_rate = 5e-6  # Much smaller learning rate
    num_epochs = 100
    patience = 25  # Even more patience for stable training
    
    # Gradient clipping and regularization
    max_grad_norm = 0.1  # Very aggressive gradient clipping
    weight_decay = 1e-3  # Stronger weight decay
    
    # Loss function weights (more conservative)
    loss_alpha = 0.9  # Focus almost entirely on main loss
    loss_beta = 0.08  # Minimal individual loss weight
    loss_gamma = 0.02  # Minimal uncertainty loss weight
    
    # Numerical stability parameters
    eps = 1e-8  # Small epsilon for numerical stability
    min_std = 1e-6  # Minimum standard deviation for uncertainty
    max_std = 5.0  # Maximum standard deviation for uncertainty
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Features (excluding target and time columns)
    feature_cols = ['temp', 'dew', 'humidity', 'winddir', 'windspeed', 'pressure', 'cloudcover']
    time_cols = ['dayofyear', 'timeofday'] 