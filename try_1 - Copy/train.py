import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import our components
from config import Config
from data_preprocessing import prepare_data
from models.hybrid_model import HybridSolarModel, HybridLoss

class Trainer:
    """
    Comprehensive trainer for the hybrid solar energy prediction model
    """
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Prepare data
        print("Preparing data...")
        self.train_loader, self.val_loader, self.test_loader, self.preprocessor = prepare_data(config)
        
        # Initialize model
        print("Initializing model...")
        self.model = HybridSolarModel(config).to(self.device)
        
        # Loss function
        self.criterion = HybridLoss(
            alpha=config.loss_alpha,
            beta=config.loss_beta, 
            gamma=config.loss_gamma
        )
        
        # Optimizer and scheduler
        self.optimizer = optim.AdamW(
            self.model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.7, 
            patience=10,
            verbose=True
        )
        
        # Early stopping
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Tracking
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
        
        # Create results directory
        os.makedirs('results', exist_ok=True)
        os.makedirs('models', exist_ok=True)
        os.makedirs('plots', exist_ok=True)

    def calculate_metrics(self, predictions, targets, is_training=True):
        """Calculate comprehensive evaluation metrics"""
        # Check for NaN values in predictions and targets
        pred_cpu = predictions.cpu().numpy().flatten()
        target_cpu = targets.cpu().numpy().flatten()
        
        if np.isnan(pred_cpu).any():
            print(f"Warning: NaN values detected in predictions! Count: {np.isnan(pred_cpu).sum()}")
            pred_cpu = np.nan_to_num(pred_cpu, nan=0.0)  # Replace NaN with 0
            
        if np.isnan(target_cpu).any():
            print(f"Warning: NaN values detected in targets! Count: {np.isnan(target_cpu).sum()}")
            target_cpu = np.nan_to_num(target_cpu, nan=0.0)  # Replace NaN with 0
        
        # For training, calculate metrics in scaled space to avoid explosion
        if is_training:
            # Calculate metrics in scaled space
            mse = mean_squared_error(target_cpu, pred_cpu)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(target_cpu, pred_cpu)
            
            # For R2, we need to be careful with scaled data
            try:
                r2 = r2_score(target_cpu, pred_cpu)
                if np.isnan(r2) or np.isinf(r2):
                    r2 = -1.0  # Default poor score
            except:
                r2 = -1.0
            
            # MAPE in scaled space (less meaningful but prevents explosion)
            mape = np.mean(np.abs((target_cpu - pred_cpu) / (np.abs(target_cpu) + 1e-8))) * 100
            
            return {
                'MSE': mse,
                'RMSE': rmse,
                'MAE': mae,
                'R2': r2,
                'MAPE': mape
            }
        else:
            # For final evaluation, inverse transform but with safety checks
            try:
                pred_orig = self.preprocessor.inverse_transform_target(pred_cpu)
                target_orig = self.preprocessor.inverse_transform_target(target_cpu)
                
                # Safety check for extreme values after inverse transform
                if np.abs(pred_orig).max() > 1e6 or np.abs(target_orig).max() > 1e6:
                    print("Warning: Extreme values detected after inverse transform!")
                    print(f"Max pred: {np.abs(pred_orig).max()}, Max target: {np.abs(target_orig).max()}")
                    # Fall back to scaled space calculation
                    return self.calculate_metrics(predictions, targets, is_training=True)
                
                # Calculate metrics in original space
                mse = mean_squared_error(target_orig, pred_orig)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(target_orig, pred_orig)
                r2 = r2_score(target_orig, pred_orig)
                
                # Mean Absolute Percentage Error
                mape = np.mean(np.abs((target_orig - pred_orig) / (target_orig + 1e-8))) * 100
                
                return {
                    'MSE': mse,
                    'RMSE': rmse,
                    'MAE': mae,
                    'R2': r2,
                    'MAPE': mape
                }
            except Exception as e:
                print(f"Error in inverse transform: {e}")
                print("Falling back to scaled space metrics")
                return self.calculate_metrics(predictions, targets, is_training=True)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch_idx, batch in enumerate(pbar):
            inputs = batch['input'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            # Calculate loss
            loss_dict = self.criterion(outputs, targets)
            loss = loss_dict['total_loss']
            
            # Check for NaN or infinite loss before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Invalid loss detected at batch {batch_idx}: {loss.item()}")
                print("Skipping this batch to prevent gradient explosion")
                continue
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.config.max_grad_norm)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            all_predictions.append(outputs['prediction'].detach())
            all_targets.append(targets)
            
            # Debug: Print actual values for first few batches
            if batch_idx < 3:
                print(f"\nDEBUG Batch {batch_idx}:")
                print(f"  Prediction range: [{outputs['prediction'].min().item():.6f}, {outputs['prediction'].max().item():.6f}]")
                print(f"  Target range: [{targets.min().item():.6f}, {targets.max().item():.6f}]")
                print(f"  Loss: {loss.item():.6f}")
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg_Loss': f'{total_loss/(batch_idx+1):.4f}'
            })
        
        # Calculate epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.calculate_metrics(all_predictions, all_targets, is_training=True)
        
        return avg_loss, metrics

    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                loss_dict = self.criterion(outputs, targets)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                all_predictions.append(outputs['prediction'])
                all_targets.append(targets)
        
        avg_loss = total_loss / len(self.val_loader)
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        metrics = self.calculate_metrics(all_predictions, all_targets, is_training=True)
        
        return avg_loss, metrics

    def test(self):
        """Test the model"""
        print("Testing model...")
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_outputs = []
        
        with torch.no_grad():
            for batch in tqdm(self.test_loader, desc="Testing"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                outputs = self.model(inputs)
                
                all_predictions.append(outputs['prediction'])
                all_targets.append(targets)
                all_outputs.append(outputs)
        
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate test metrics
        test_metrics = self.calculate_metrics(all_predictions, all_targets, is_training=False)
        
        # Save predictions for analysis
        predictions_np = self.preprocessor.inverse_transform_target(
            all_predictions.cpu().numpy().reshape(-1)
        )
        targets_np = self.preprocessor.inverse_transform_target(
            all_targets.cpu().numpy().reshape(-1)
        )
        
        # Create results DataFrame
        results_df = pd.DataFrame({
            'Actual': targets_np,
            'Predicted': predictions_np,
            'Error': targets_np - predictions_np,
            'Abs_Error': np.abs(targets_np - predictions_np),
            'Squared_Error': (targets_np - predictions_np) ** 2
        })
        
        results_df.to_csv('results/test_predictions.csv', index=False)
        
        return test_metrics, results_df

    def save_model(self, epoch, is_best=False):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, 'models/latest_checkpoint.pth')
        
        # Save best model
        if is_best:
            torch.save(checkpoint, 'models/best_model.pth')
            print(f"New best model saved with validation loss: {self.best_val_loss:.4f}")

    def load_model(self, checkpoint_path):
        """Load model from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_val_loss = checkpoint['best_val_loss']
        return checkpoint['epoch']

    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        # Loss curves
        axes[0, 0].plot(self.train_losses, label='Train Loss', color='blue')
        axes[0, 0].plot(self.val_losses, label='Val Loss', color='red')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Metrics
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'MAPE']
        for i, metric in enumerate(metrics_to_plot):
            row = (i + 1) // 3
            col = (i + 1) % 3
            
            train_values = [m[metric] for m in self.train_metrics]
            val_values = [m[metric] for m in self.val_metrics]
            
            axes[row, col].plot(train_values, label=f'Train {metric}', color='blue')
            axes[row, col].plot(val_values, label=f'Val {metric}', color='red')
            axes[row, col].set_title(f'{metric} Over Training')
            axes[row, col].set_xlabel('Epoch')
            axes[row, col].set_ylabel(metric)
            axes[row, col].legend()
            axes[row, col].grid(True)
        
        # Remove empty subplot
        if len(metrics_to_plot) < 5:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_predictions(self, results_df, sample_size=1000):
        """Plot prediction results"""
        # Sample data for visualization
        sample_df = results_df.sample(min(sample_size, len(results_df)))
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Actual vs Predicted scatter plot
        axes[0, 0].scatter(sample_df['Actual'], sample_df['Predicted'], alpha=0.6, s=20)
        axes[0, 0].plot([sample_df['Actual'].min(), sample_df['Actual'].max()], 
                       [sample_df['Actual'].min(), sample_df['Actual'].max()], 
                       'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual Solar Power Output')
        axes[0, 0].set_ylabel('Predicted Solar Power Output')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Time series plot (first 500 points)
        subset = results_df.head(500)
        axes[0, 1].plot(subset['Actual'], label='Actual', color='blue', alpha=0.7)
        axes[0, 1].plot(subset['Predicted'], label='Predicted', color='red', alpha=0.7)
        axes[0, 1].set_title('Time Series: Actual vs Predicted (First 500 points)')
        axes[0, 1].set_xlabel('Time Steps')
        axes[0, 1].set_ylabel('Solar Power Output')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Error distribution
        axes[1, 0].hist(sample_df['Error'], bins=50, alpha=0.7, color='green')
        axes[1, 0].set_title('Prediction Error Distribution')
        axes[1, 0].set_xlabel('Error (Actual - Predicted)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True)
        
        # Absolute error vs actual values
        axes[1, 1].scatter(sample_df['Actual'], sample_df['Abs_Error'], alpha=0.6, s=20)
        axes[1, 1].set_xlabel('Actual Solar Power Output')
        axes[1, 1].set_ylabel('Absolute Error')
        axes[1, 1].set_title('Absolute Error vs Actual Values')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('plots/predictions_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()

    def train(self):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_metrics = self.train_epoch()
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            # Validate
            val_loss, val_metrics = self.validate()
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            # Learning rate scheduling
            self.scheduler.step(val_loss)
            
            # Print metrics
            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            print(f"Train RMSE: {train_metrics['RMSE']:.4f} | Val RMSE: {val_metrics['RMSE']:.4f}")
            print(f"Train R2: {train_metrics['R2']:.4f} | Val R2: {val_metrics['R2']:.4f}")
            
            # Save model
            is_best = val_loss < self.best_val_loss
            if is_best:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            self.save_model(epoch, is_best)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                print(f"\nEarly stopping triggered after {epoch+1} epochs")
                break
        
        training_time = time.time() - start_time
        print(f"\nTraining completed in {training_time/60:.2f} minutes")
        
        # Load best model for testing
        print("Loading best model for testing...")
        self.load_model('models/best_model.pth')
        
        # Test
        test_metrics, results_df = self.test()
        
        # Print final results
        print("\n" + "="*60)
        print("FINAL TEST RESULTS")
        print("="*60)
        for metric, value in test_metrics.items():
            print(f"{metric}: {value:.4f}")
        
        # Generate plots
        print("\nGenerating plots...")
        self.plot_training_history()
        self.plot_predictions(results_df)
        
        # Save final metrics
        final_results = {
            'test_metrics': test_metrics,
            'best_val_loss': self.best_val_loss,
            'training_time_minutes': training_time/60,
            'total_epochs': epoch+1
        }
        
        pd.DataFrame([final_results]).to_csv('results/final_metrics.csv', index=False)
        
        return final_results

def main():
    """Main function"""
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load configuration
    config = Config()
    
    print("Hybrid Solar Energy Prediction Model")
    print("="*50)
    print(f"Sequence Length: {config.seq_len}")
    print(f"Prediction Length: {config.pred_len}")
    print(f"Batch Size: {config.batch_size}")
    print(f"Learning Rate: {config.learning_rate}")
    print(f"Device: {config.device}")
    print("="*50)
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    results = trainer.train()
    
    print("\nTraining completed successfully!")
    print(f"Best validation loss: {results['best_val_loss']:.4f}")
    print(f"Final test RMSE: {results['test_metrics']['RMSE']:.4f}")
    print(f"Final test R2: {results['test_metrics']['R2']:.4f}")
    
if __name__ == "__main__":
    main() 