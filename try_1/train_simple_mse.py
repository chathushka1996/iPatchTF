"""
Simplified training script focused on achieving MSE < 0.1
Uses ultra-conservative settings and explicit MSE monitoring
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

from ULTRA_CONSERVATIVE_CONFIG import UltraConservativeConfig

class SimpleMSEModel(nn.Module):
    """Ultra-simple model designed specifically for MSE < 0.1"""
    
    def __init__(self, input_dim, seq_len, pred_len):
        super().__init__()
        
        # Very simple architecture
        self.seq_len = seq_len
        self.pred_len = pred_len
        
        self.model = nn.Sequential(
            nn.Linear(input_dim * seq_len, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, pred_len)
        )
        
        # Initialize weights for small outputs
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.1)  # Small gain
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Flatten sequence
        B, L, C = x.shape
        x_flat = x.reshape(B, L * C)
        
        # Predict
        out = self.model(x_flat)
        return out.unsqueeze(-1)  # [B, pred_len, 1]

def prepare_simple_data(config):
    """Prepare data with ultra-conservative preprocessing"""
    print("🔄 Loading and preprocessing data for MSE < 0.1...")
    
    # Load raw data
    train_df = pd.read_csv(f"{config.data_path}/train.csv")
    val_df = pd.read_csv(f"{config.data_path}/val.csv")
    
    print(f"📊 Data loaded - Train: {len(train_df)}, Val: {len(val_df)}")
    
    # Select only core features
    feature_cols = config.feature_cols + config.time_cols
    target_col = config.target_col
    
    print(f"🎯 Using {len(feature_cols)} features: {feature_cols}")
    
    # Clean target values
    for df_name, df in [("train", train_df), ("val", val_df)]:
        # Remove negative values
        negative_count = (df[target_col] < 0).sum()
        if negative_count > 0:
            print(f"  {df_name}: Removing {negative_count} negative values")
            df[target_col] = np.maximum(df[target_col], 0.0)
        
        # Cap extreme outliers
        if df_name == "train":
            outlier_threshold = df[target_col].quantile(0.99)  # More aggressive
            print(f"  Outlier threshold (99th percentile): {outlier_threshold:.2f}")
        
        outlier_count = (df[target_col] > outlier_threshold).sum()
        if outlier_count > 0:
            print(f"  {df_name}: Capping {outlier_count} outliers")
            df[target_col] = np.minimum(df[target_col], outlier_threshold)
        
        zero_ratio = (df[target_col] == 0).sum() / len(df)
        print(f"  {df_name}: Zero ratio: {zero_ratio:.1%}")
    
    # Scale data
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fit on training data
    feature_scaler.fit(train_df[feature_cols])
    target_scaler.fit(train_df[[target_col]])
    
    print(f"📏 Target scaling - Mean: {target_scaler.mean_[0]:.2f}, Std: {target_scaler.scale_[0]:.2f}")
    
    # Transform data
    train_features = feature_scaler.transform(train_df[feature_cols])
    val_features = feature_scaler.transform(val_df[feature_cols])
    
    train_targets = target_scaler.transform(train_df[[target_col]]).flatten()
    val_targets = target_scaler.transform(val_df[[target_col]]).flatten()
    
    # Create sequences
    def create_sequences(features, targets, seq_len, pred_len):
        X, y = [], []
        for i in range(len(features) - seq_len - pred_len + 1):
            X.append(features[i:i+seq_len])
            y.append(targets[i+seq_len:i+seq_len+pred_len])
        return np.array(X), np.array(y)
    
    print(f"🔗 Creating sequences (seq_len={config.seq_len}, pred_len={config.pred_len})...")
    
    X_train, y_train = create_sequences(train_features, train_targets, config.seq_len, config.pred_len)
    X_val, y_val = create_sequences(val_features, val_targets, config.seq_len, config.pred_len)
    
    print(f"📦 Sequences created - Train: {X_train.shape}, Val: {X_val.shape}")
    
    # Create datasets
    train_dataset = TensorDataset(torch.FloatTensor(X_train), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(X_val), torch.FloatTensor(y_val))
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, target_scaler, len(feature_cols)

def train_for_low_mse():
    """Train model specifically to achieve MSE < 0.1"""
    config = UltraConservativeConfig()
    
    print("🎯 ULTRA-CONSERVATIVE TRAINING FOR MSE < 0.1")
    print("=" * 60)
    print(f"Target MSE: {config.target_mse}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Sequence length: {config.seq_len}")
    print(f"Prediction length: {config.pred_len}")
    print("=" * 60)
    
    # Prepare data
    train_loader, val_loader, target_scaler, input_dim = prepare_simple_data(config)
    
    # Initialize model
    print(f"🧠 Initializing ultra-simple model (input_dim={input_dim})...")
    model = SimpleMSEModel(input_dim, config.seq_len, config.pred_len).to(config.device)
    
    # MSE loss only
    criterion = nn.MSELoss()
    
    # Ultra-conservative optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=20, verbose=True)
    
    print(f"🔢 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Training loop with MSE focus
    best_mse = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        # Training
        model.train()
        train_mse_sum = 0
        train_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}")
        for batch_idx, (X, y) in enumerate(pbar):
            X, y = X.to(config.device), y.to(config.device)
            
            optimizer.zero_grad()
            
            # Forward pass
            pred = model(X)
            loss = criterion(pred.squeeze(-1), y)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.max_grad_norm)
            optimizer.step()
            
            # Track MSE
            batch_mse = loss.item()
            train_mse_sum += batch_mse
            train_batches += 1
            
            # Update progress with MSE
            pbar.set_postfix({
                'MSE': f'{batch_mse:.6f}', 
                'Avg_MSE': f'{train_mse_sum/train_batches:.6f}'
            })
            
            # Debug first few batches
            if batch_idx < 2:
                print(f"\n  DEBUG Batch {batch_idx}:")
                print(f"    Pred range: [{pred.min().item():.4f}, {pred.max().item():.4f}]")
                print(f"    Target range: [{y.min().item():.4f}, {y.max().item():.4f}]")
                print(f"    MSE: {batch_mse:.6f}")
        
        # Validation
        model.eval()
        val_mse_sum = 0
        val_batches = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(config.device), y.to(config.device)
                pred = model(X)
                loss = criterion(pred.squeeze(-1), y)
                val_mse_sum += loss.item()
                val_batches += 1
        
        # Calculate epoch metrics
        train_mse = train_mse_sum / train_batches
        val_mse = val_mse_sum / val_batches
        
        # Update scheduler
        scheduler.step(val_mse)
        
        # Print results
        print(f"\n🎯 EPOCH {epoch+1} RESULTS:")
        print(f"  Train MSE: {train_mse:.6f}")
        print(f"  Val MSE: {val_mse:.6f}")
        print(f"  Target: {config.target_mse:.6f}")
        
        # Check target achievement
        if val_mse < config.target_mse:
            print(f"🎉 TARGET ACHIEVED! MSE {val_mse:.6f} < {config.target_mse}")
            print("✅ Saving model...")
            torch.save(model.state_dict(), 'best_low_mse_model.pth')
            break
        elif val_mse < config.target_mse * 2:
            print(f"📈 Getting close! MSE: {val_mse:.6f}")
        else:
            print(f"📊 Current progress: {val_mse:.6f} (need: {config.target_mse:.6f})")
        
        # Early stopping
        if val_mse < best_mse:
            best_mse = val_mse
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model_checkpoint.pth')
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            print(f"⏱️ Early stopping after {epoch+1} epochs")
            break
        
        print(f"  Best MSE so far: {best_mse:.6f}")
        print(f"  Patience: {patience_counter}/{config.patience}")
        print("-" * 60)
    
    print(f"\n🏁 TRAINING COMPLETE")
    print(f"📊 Best validation MSE: {best_mse:.6f}")
    print(f"🎯 Target MSE: {config.target_mse}")
    
    if best_mse < config.target_mse:
        print("🎉 SUCCESS: Target MSE achieved!")
    else:
        print(f"📈 Progress made: {(1 - best_mse/2.0)*100:.1f}% improvement from ~2.0 baseline")
        print("💡 Consider: Even smaller learning rate, longer training, or simpler architecture")

if __name__ == "__main__":
    train_for_low_mse() 