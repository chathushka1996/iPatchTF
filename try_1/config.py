import torch

class Config:
    # Data parameters
    data_path = 'data/sl_t/'
    target_col = 'Solar Power Output'
    
    # Time series parameters
    seq_len = 96  # Input sequence length (24 hours = 96 * 15min intervals)
    pred_len = 96  # Prediction length (24 hours = 96 * 15min intervals)
    
    # Model parameters
    d_model = 512
    n_heads = 8
    e_layers = 3  # Encoder layers
    d_layers = 2  # Decoder layers
    d_ff = 2048
    dropout = 0.1
    
    # PatchTST parameters
    patch_size = 16
    stride = 8
    
    # Autoformer parameters
    moving_avg = 25
    factor = 3
    
    # Statistical features
    rolling_windows = [12, 24, 48, 96]  # 3h, 6h, 12h, 24h
    lag_features = [1, 4, 12, 24, 48, 96]  # Various lag periods
    
    # Training parameters
    batch_size = 32
    learning_rate = 5e-5  # Reduced for more stable training
    num_epochs = 100
    patience = 15
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Features (excluding target and time columns)
    feature_cols = ['temp', 'dew', 'humidity', 'winddir', 'windspeed', 'pressure', 'cloudcover']
    time_cols = ['dayofyear', 'timeofday'] 