import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from config import Config
import warnings
warnings.filterwarnings('ignore')

class StatisticalFeatureGenerator:
    """Generate statistical features for time series data"""
    
    def __init__(self, config):
        self.config = config
        
    def create_lag_features(self, df, target_col):
        """Create lag features"""
        for lag in self.config.lag_features:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        return df
    
    def create_rolling_features(self, df, target_col):
        """Create rolling statistical features"""
        for window in self.config.rolling_windows:
            # Rolling statistics
            df[f'{target_col}_rolling_mean_{window}'] = df[target_col].rolling(window).mean()
            df[f'{target_col}_rolling_std_{window}'] = df[target_col].rolling(window).std()
            df[f'{target_col}_rolling_min_{window}'] = df[target_col].rolling(window).min()
            df[f'{target_col}_rolling_max_{window}'] = df[target_col].rolling(window).max()
            df[f'{target_col}_rolling_q25_{window}'] = df[target_col].rolling(window).quantile(0.25)
            df[f'{target_col}_rolling_q75_{window}'] = df[target_col].rolling(window).quantile(0.75)
        return df
    
    def create_seasonal_features(self, df):
        """Create seasonal and cyclical features"""
        df['hour'] = pd.to_datetime(df['date']).dt.hour
        df['month'] = pd.to_datetime(df['date']).dt.month
        df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
        
        # Cyclical encoding
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['dayofyear_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 366)
        df['dayofyear_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 366)
        
        return df
    
    def create_weather_interaction_features(self, df):
        """Create weather interaction features"""
        # Temperature-humidity interaction
        df['temp_humidity_interaction'] = df['temp'] * df['humidity'] / 100
        
        # Wind chill equivalent for solar
        df['wind_temp_interaction'] = df['windspeed'] * (df['temp'] - 32)
        
        # Cloud cover impact on temperature
        df['cloud_temp_interaction'] = df['cloudcover'] * df['temp'] / 100
        
        # Pressure gradient (approximation)
        df['pressure_gradient'] = df['pressure'].diff()
        
        # Dew point depression
        df['dew_depression'] = df['temp'] - df['dew']
        
        return df
    
    def create_solar_specific_features(self, df):
        """Create solar-specific features"""
        # Solar elevation angle approximation (simplified)
        hour = pd.to_datetime(df['date']).dt.hour + pd.to_datetime(df['date']).dt.minute / 60
        day_of_year = df['dayofyear']
        
        # Declination angle
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Solar elevation (simplified, assuming latitude ~25°N)
        latitude = 25  # Approximate latitude
        elevation = np.arcsin(
            np.sin(np.radians(declination)) * np.sin(np.radians(latitude)) +
            np.cos(np.radians(declination)) * np.cos(np.radians(latitude)) * np.cos(np.radians(hour_angle))
        )
        
        df['solar_elevation'] = np.degrees(elevation)
        df['solar_elevation'] = np.maximum(df['solar_elevation'], 0)  # No negative elevation
        
        # Clear sky potential (elevation * (1 - cloud_cover/100))
        df['clear_sky_potential'] = df['solar_elevation'] * (1 - df['cloudcover'] / 100)
        
        return df
    
    def generate_all_features(self, df):
        """Generate all statistical features"""
        print("Generating lag features...")
        df = self.create_lag_features(df, self.config.target_col)
        
        print("Generating rolling features...")
        df = self.create_rolling_features(df, self.config.target_col)
        
        print("Generating seasonal features...")
        df = self.create_seasonal_features(df)
        
        print("Generating weather interaction features...")
        df = self.create_weather_interaction_features(df)
        
        print("Generating solar-specific features...")
        df = self.create_solar_specific_features(df)
        
        return df

class SolarDataset(Dataset):
    """PyTorch Dataset for solar energy prediction"""
    
    def __init__(self, data, config, mode='train'):
        self.data = data
        self.config = config
        self.mode = mode
        
        # Get feature columns (exclude date, target, and any non-numeric columns)
        self.feature_cols = [col for col in data.columns 
                           if col not in ['date', config.target_col] 
                           and data[col].dtype in ['float64', 'int64']]
        
        # Prepare sequences
        self.prepare_sequences()
    
    def prepare_sequences(self):
        """Prepare input-output sequences"""
        self.sequences = []
        self.targets = []
        
        # Convert to numpy for faster processing
        features = self.data[self.feature_cols].values
        target = self.data[self.config.target_col].values
        
        seq_len = self.config.seq_len
        pred_len = self.config.pred_len
        
        for i in range(len(self.data) - seq_len - pred_len + 1):
            # Input sequence
            seq_x = features[i:i + seq_len]
            
            # Target sequence
            seq_y = target[i + seq_len:i + seq_len + pred_len]
            
            self.sequences.append(seq_x)
            self.targets.append(seq_y)
        
        self.sequences = np.array(self.sequences)
        self.targets = np.array(self.targets)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            'input': torch.FloatTensor(self.sequences[idx]),
            'target': torch.FloatTensor(self.targets[idx])
        }

class DataPreprocessor:
    """Main data preprocessing class"""
    
    def __init__(self, config):
        self.config = config
        self.feature_generator = StatisticalFeatureGenerator(config)
        self.scaler_features = RobustScaler()
        self.scaler_target = RobustScaler()
        
    def load_data(self):
        """Load all datasets"""
        print("Loading datasets...")
        train_df = pd.read_csv(f"{self.config.data_path}/train.csv")
        val_df = pd.read_csv(f"{self.config.data_path}/val.csv")
        test_df = pd.read_csv(f"{self.config.data_path}/test.csv")
        
        print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
        return train_df, val_df, test_df
    
    def preprocess_data(self, train_df, val_df, test_df):
        """Preprocess all datasets"""
        print("Generating features for all datasets...")
        
        # Generate features for each dataset
        train_processed = self.feature_generator.generate_all_features(train_df.copy())
        val_processed = self.feature_generator.generate_all_features(val_df.copy())
        test_processed = self.feature_generator.generate_all_features(test_df.copy())
        
        # Handle missing values (forward fill then backward fill)
        train_processed = train_processed.fillna(method='ffill').fillna(method='bfill')
        val_processed = val_processed.fillna(method='ffill').fillna(method='bfill')
        test_processed = test_processed.fillna(method='ffill').fillna(method='bfill')
        
        # Get feature columns
        feature_cols = [col for col in train_processed.columns 
                       if col not in ['date', self.config.target_col] 
                       and train_processed[col].dtype in ['float64', 'int64']]
        
        print(f"Total features: {len(feature_cols)}")
        
        # Fit scalers on training data
        self.scaler_features.fit(train_processed[feature_cols])
        self.scaler_target.fit(train_processed[[self.config.target_col]])
        
        # Scale features
        train_processed[feature_cols] = self.scaler_features.transform(train_processed[feature_cols])
        val_processed[feature_cols] = self.scaler_features.transform(val_processed[feature_cols])
        test_processed[feature_cols] = self.scaler_features.transform(test_processed[feature_cols])
        
        # Scale target
        train_processed[self.config.target_col] = self.scaler_target.transform(train_processed[[self.config.target_col]]).flatten()
        val_processed[self.config.target_col] = self.scaler_target.transform(val_processed[[self.config.target_col]]).flatten()
        test_processed[self.config.target_col] = self.scaler_target.transform(test_processed[[self.config.target_col]]).flatten()
        
        return train_processed, val_processed, test_processed
    
    def create_datasets(self, train_df, val_df, test_df):
        """Create PyTorch datasets"""
        train_dataset = SolarDataset(train_df, self.config, mode='train')
        val_dataset = SolarDataset(val_df, self.config, mode='val')
        test_dataset = SolarDataset(test_df, self.config, mode='test')
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloaders(self, train_dataset, val_dataset, test_dataset):
        """Create PyTorch dataloaders"""
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True, 
            num_workers=0
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False, 
            num_workers=0
        )
        
        return train_loader, val_loader, test_loader
    
    def inverse_transform_target(self, scaled_target):
        """Inverse transform the target variable"""
        return self.scaler_target.inverse_transform(scaled_target.reshape(-1, 1)).flatten()

def prepare_data(config):
    """Main function to prepare all data"""
    preprocessor = DataPreprocessor(config)
    
    # Load data
    train_df, val_df, test_df = preprocessor.load_data()
    
    # Preprocess data
    train_processed, val_processed, test_processed = preprocessor.preprocess_data(
        train_df, val_df, test_df
    )
    
    # Create datasets
    train_dataset, val_dataset, test_dataset = preprocessor.create_datasets(
        train_processed, val_processed, test_processed
    )
    
    # Create dataloaders
    train_loader, val_loader, test_loader = preprocessor.create_dataloaders(
        train_dataset, val_dataset, test_dataset
    )
    
    print(f"Data preparation complete!")
    print(f"Input feature dim: {train_dataset.sequences.shape[-1]}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    return train_loader, val_loader, test_loader, preprocessor

if __name__ == "__main__":
    config = Config()
    train_loader, val_loader, test_loader, preprocessor = prepare_data(config) 