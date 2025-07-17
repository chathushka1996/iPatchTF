import torch

class UltraConservativeConfig:
    """
    Ultra-conservative configuration specifically designed to achieve MSE < 0.1
    for solar power prediction with 55% zero values
    """
    # Data parameters
    data_path = 'data/sl_t/'
    target_col = 'Solar Power Output'
    
    # Time series parameters
    seq_len = 48   # Reduced sequence length for simpler patterns
    pred_len = 24  # Reduced prediction length for easier learning
    
    # Model parameters (ultra-simplified)
    d_model = 64   # Very small model
    n_heads = 2    # Minimal attention heads
    e_layers = 1   # Single layer everything
    d_layers = 1   
    d_ff = 128     # Small feed-forward
    dropout = 0.1  # Light dropout
    
    # PatchTST parameters (simplified)
    patch_size = 8
    stride = 4
    
    # Statistical features (reduced)
    rolling_windows = [6, 12, 24]  # Shorter windows
    lag_features = [1, 6, 12, 24]  # Fewer lags
    
    # Training parameters (ultra-conservative)
    batch_size = 2        # Extremely small batches
    learning_rate = 5e-7  # Very small learning rate
    num_epochs = 300      # More epochs
    patience = 100        # Very patient
    
    # Gradient control (ultra-aggressive)
    max_grad_norm = 0.01  # Very small gradients
    weight_decay = 1e-4   # Light regularization
    
    # Loss function (MSE only)
    loss_alpha = 1.0      # Only main loss
    loss_beta = 0.0       # No auxiliary losses
    loss_gamma = 0.0      
    
    # Model simplification
    use_simple_mode = True
    use_wavelet = False
    use_uncertainty = False
    use_residual_connection = False  # Even simpler
    num_experts = 1  # Single expert (simplest)
    fusion_hidden_dim = 32
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Features (core weather features only)
    feature_cols = ['temp', 'humidity', 'cloudcover']  # Only 3 most important
    time_cols = ['dayofyear', 'timeofday']
    
    # MSE monitoring
    target_mse = 0.1
    print_mse_every = 1  # Print MSE every epoch 