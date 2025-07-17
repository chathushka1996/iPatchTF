import pandas as pd
import numpy as np
from config import Config

def inspect_data_scale():
    """Inspect the actual scale of data to understand the massive loss issue"""
    config = Config()
    
    print("="*60)
    print("DATA SCALE INSPECTION")
    print("="*60)
    
    try:
        # Load raw data
        print("Loading raw data...")
        train_df = pd.read_csv(f"{config.data_path}/train.csv")
        val_df = pd.read_csv(f"{config.data_path}/val.csv")
        
        print(f"Train samples: {len(train_df)}")
        print(f"Val samples: {len(val_df)}")
        print()
        
        # Check target column statistics
        target_col = config.target_col
        print(f"TARGET COLUMN: {target_col}")
        print("-" * 40)
        
        if target_col in train_df.columns:
            target_stats = train_df[target_col].describe()
            print("Training Target Statistics:")
            print(target_stats)
            print()
            
            print(f"Target range: [{train_df[target_col].min():.6f}, {train_df[target_col].max():.6f}]")
            print(f"Target mean: {train_df[target_col].mean():.6f}")
            print(f"Target std: {train_df[target_col].std():.6f}")
            print()
            
            # Check for extreme values
            q99 = train_df[target_col].quantile(0.99)
            q01 = train_df[target_col].quantile(0.01)
            print(f"1st percentile: {q01:.6f}")
            print(f"99th percentile: {q99:.6f}")
            print()
            
            # Check for negative values
            negative_count = (train_df[target_col] < 0).sum()
            zero_count = (train_df[target_col] == 0).sum()
            print(f"Negative values: {negative_count}")
            print(f"Zero values: {zero_count}")
            print()
            
        else:
            print(f"ERROR: Target column '{target_col}' not found!")
            print("Available columns:", list(train_df.columns))
            return
            
        # Check other feature columns
        print("FEATURE COLUMNS SCALE:")
        print("-" * 40)
        feature_cols = ['temp', 'dew', 'humidity', 'winddir', 'windspeed', 'pressure', 'cloudcover']
        
        for col in feature_cols:
            if col in train_df.columns:
                col_stats = train_df[col].describe()
                print(f"{col}: min={col_stats['min']:.2f}, max={col_stats['max']:.2f}, "
                      f"mean={col_stats['mean']:.2f}, std={col_stats['std']:.2f}")
        print()
        
        # Test scaling
        print("TESTING ROBUST SCALER:")
        print("-" * 40)
        from sklearn.preprocessing import RobustScaler
        
        scaler = RobustScaler()
        target_scaled = scaler.fit_transform(train_df[[target_col]])
        
        print(f"Scaled target range: [{target_scaled.min():.6f}, {target_scaled.max():.6f}]")
        print(f"Scaled target mean: {target_scaled.mean():.6f}")
        print(f"Scaled target std: {target_scaled.std():.6f}")
        print()
        
        # Test inverse transform
        target_inverse = scaler.inverse_transform(target_scaled)
        reconstruction_error = np.abs(train_df[target_col].values - target_inverse.flatten()).max()
        print(f"Inverse transform max error: {reconstruction_error:.10f}")
        print()
        
        # Simulate a prediction to see what happens
        print("SIMULATION: TESTING LOSS CALCULATION")
        print("-" * 40)
        
        # Create dummy predictions
        dummy_pred_scaled = np.random.normal(0, 1, size=(1000, 1))  # Normal scaled predictions
        dummy_target_scaled = target_scaled[:1000].copy()
        
        # Calculate MSE in scaled space
        mse_scaled = np.mean((dummy_pred_scaled - dummy_target_scaled)**2)
        print(f"MSE in scaled space: {mse_scaled:.6f}")
        
        # Transform back to original space
        dummy_pred_orig = scaler.inverse_transform(dummy_pred_scaled)
        dummy_target_orig = scaler.inverse_transform(dummy_target_scaled)
        
        # Calculate MSE in original space
        mse_orig = np.mean((dummy_pred_orig - dummy_target_orig)**2)
        print(f"MSE in original space: {mse_orig:.6f}")
        
        # Calculate RMSE in original space
        rmse_orig = np.sqrt(mse_orig)
        print(f"RMSE in original space: {rmse_orig:.6f}")
        print()
        
        # Check if this matches what we're seeing
        print("DIAGNOSIS:")
        print("-" * 40)
        
        if mse_orig > 1e6:
            print("⚠️  PROBLEM IDENTIFIED: Inverse scaling is causing massive values!")
            print("   This suggests the scaler parameters are incorrect or")
            print("   there's an issue with the scaling/inverse scaling process.")
        else:
            print("✅ Scaling appears normal. Issue might be elsewhere.")
            
        print()
        print("RECOMMENDATIONS:")
        print("-" * 40)
        print("1. Check if targets are being calculated correctly in loss function")
        print("2. Verify that model outputs and targets are on the same scale")
        print("3. Consider using StandardScaler instead of RobustScaler")
        print("4. Add more debugging to the training loop to see actual values")
        
    except Exception as e:
        print(f"Error during inspection: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    inspect_data_scale() 