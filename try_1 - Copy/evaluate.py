import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import time
import os
from tqdm import tqdm

# Import our components
from config import Config
from data_preprocessing import prepare_data
from models.hybrid_model import HybridSolarModel
from models.patchtst import EnhancedPatchTST

class ModelEvaluator:
    """
    Comprehensive evaluator for comparing different models
    """
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        # Prepare data
        print("Preparing data...")
        self.train_loader, self.val_loader, self.test_loader, self.preprocessor = prepare_data(config)
        
        # Initialize models
        self.models = {}
        
    def add_model(self, name, model):
        """Add a model for evaluation"""
        self.models[name] = model.to(self.device)
        
    def load_model(self, name, checkpoint_path):
        """Load model from checkpoint"""
        if name not in self.models:
            raise ValueError(f"Model {name} not found. Add it first using add_model()")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.models[name].load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded {name} from {checkpoint_path}")
        
    def evaluate_model(self, name, dataloader):
        """Evaluate a single model"""
        if name not in self.models:
            raise ValueError(f"Model {name} not found")
            
        model = self.models[name]
        model.eval()
        
        all_predictions = []
        all_targets = []
        inference_times = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Evaluating {name}"):
                inputs = batch['input'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Measure inference time
                start_time = time.time()
                
                # Get predictions based on model type
                if name == 'Hybrid':
                    outputs = model(inputs)
                    predictions = outputs['prediction']
                else:
                    predictions = model(inputs)
                
                inference_time = time.time() - start_time
                inference_times.append(inference_time)
                
                all_predictions.append(predictions)
                all_targets.append(targets)
        
        # Concatenate all predictions and targets
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate metrics
        metrics = self.calculate_metrics(all_predictions, all_targets)
        metrics['avg_inference_time'] = np.mean(inference_times)
        metrics['total_inference_time'] = np.sum(inference_times)
        
        return metrics, all_predictions, all_targets
    
    def calculate_metrics(self, predictions, targets):
        """Calculate comprehensive evaluation metrics"""
        # Inverse transform predictions and targets
        pred_orig = self.preprocessor.inverse_transform_target(predictions.cpu().numpy().flatten())
        target_orig = self.preprocessor.inverse_transform_target(targets.cpu().numpy().flatten())
        
        # Calculate metrics
        mse = mean_squared_error(target_orig, pred_orig)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(target_orig, pred_orig)
        r2 = r2_score(target_orig, pred_orig)
        
        # Mean Absolute Percentage Error
        mape = np.mean(np.abs((target_orig - pred_orig) / (target_orig + 1e-8))) * 100
        
        # Additional metrics
        max_error = np.max(np.abs(target_orig - pred_orig))
        
        return {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R2': r2,
            'MAPE': mape,
            'Max_Error': max_error
        }
    
    def compare_models(self, dataloader_name="test"):
        """Compare all models"""
        dataloader = getattr(self, f"{dataloader_name}_loader")
        
        results = {}
        all_predictions = {}
        all_targets = None
        
        print(f"\nComparing models on {dataloader_name} set...")
        print("="*60)
        
        for name in self.models.keys():
            print(f"\nEvaluating {name}...")
            metrics, predictions, targets = self.evaluate_model(name, dataloader)
            results[name] = metrics
            all_predictions[name] = predictions
            
            if all_targets is None:
                all_targets = targets
            
            # Print metrics
            print(f"{name} Results:")
            for metric, value in metrics.items():
                if metric.endswith('_time'):
                    print(f"  {metric}: {value:.4f}s")
                else:
                    print(f"  {metric}: {value:.4f}")
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(results).T
        
        # Save results
        os.makedirs('results', exist_ok=True)
        comparison_df.to_csv('results/model_comparison.csv')
        
        # Generate comparison plots
        self.plot_model_comparison(comparison_df, all_predictions, all_targets)
        
        return comparison_df, all_predictions, all_targets
    
    def plot_model_comparison(self, comparison_df, all_predictions, all_targets):
        """Generate comprehensive comparison plots"""
        os.makedirs('plots', exist_ok=True)
        
        # Convert targets to original scale for plotting
        targets_orig = self.preprocessor.inverse_transform_target(
            all_targets.cpu().numpy().flatten()
        )
        
        # Convert predictions to original scale
        pred_orig = {}
        for name, predictions in all_predictions.items():
            pred_orig[name] = self.preprocessor.inverse_transform_target(
                predictions.cpu().numpy().flatten()
            )
        
        # 1. Metrics comparison bar plot
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        
        metrics_to_plot = ['RMSE', 'MAE', 'R2', 'MAPE', 'Max_Error']
        for i, metric in enumerate(metrics_to_plot):
            row = i // 3
            col = i % 3
            
            values = comparison_df[metric].values
            models = comparison_df.index.values
            
            colors = ['lightblue', 'lightgreen', 'lightcoral', 'gold'][:len(models)]
            bars = axes[row, col].bar(models, values, color=colors)
            axes[row, col].set_title(f'{metric} Comparison')
            axes[row, col].set_ylabel(metric)
            axes[row, col].grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                height = bar.get_height()
                axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                   f'{value:.3f}', ha='center', va='bottom')
        
        # Remove empty subplot
        if len(metrics_to_plot) < 6:
            fig.delaxes(axes[1, 2])
        
        plt.tight_layout()
        plt.savefig('plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Prediction comparison scatter plots
        n_models = len(all_predictions)
        fig, axes = plt.subplots(1, n_models, figsize=(6*n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        for i, (name, predictions) in enumerate(pred_orig.items()):
            # Sample data for better visualization
            sample_size = min(1000, len(predictions))
            indices = np.random.choice(len(predictions), sample_size, replace=False)
            
            sample_targets = targets_orig[indices]
            sample_predictions = predictions[indices]
            
            axes[i].scatter(sample_targets, sample_predictions, alpha=0.6, s=20)
            axes[i].plot([sample_targets.min(), sample_targets.max()], 
                        [sample_targets.min(), sample_targets.max()], 
                        'r--', lw=2, label='Perfect Prediction')
            axes[i].set_xlabel('Actual Solar Power Output')
            axes[i].set_ylabel('Predicted Solar Power Output')
            axes[i].set_title(f'{name}: Actual vs Predicted')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add R² score to plot
            r2 = comparison_df.loc[name, 'R2']
            axes[i].text(0.05, 0.95, f'R² = {r2:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('plots/scatter_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Time series comparison
        fig, axes = plt.subplots(n_models, 1, figsize=(15, 4*n_models))
        if n_models == 1:
            axes = [axes]
        
        # Use first 500 points for time series visualization
        subset_size = min(500, len(targets_orig))
        subset_targets = targets_orig[:subset_size]
        
        for i, (name, predictions) in enumerate(pred_orig.items()):
            subset_predictions = predictions[:subset_size]
            
            axes[i].plot(subset_targets, label='Actual', color='blue', alpha=0.7)
            axes[i].plot(subset_predictions, label='Predicted', color='red', alpha=0.7)
            axes[i].set_title(f'{name}: Time Series Comparison')
            axes[i].set_xlabel('Time Steps')
            axes[i].set_ylabel('Solar Power Output')
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)
            
            # Add RMSE to plot
            rmse = comparison_df.loc[name, 'RMSE']
            axes[i].text(0.05, 0.95, f'RMSE = {rmse:.3f}', transform=axes[i].transAxes,
                        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig('plots/timeseries_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("\nComparison plots saved to 'plots/' directory")
    
    def analyze_prediction_errors(self, all_predictions, all_targets):
        """Analyze prediction errors across models"""
        targets_orig = self.preprocessor.inverse_transform_target(
            all_targets.cpu().numpy().flatten()
        )
        
        errors = {}
        for name, predictions in all_predictions.items():
            pred_orig = self.preprocessor.inverse_transform_target(
                predictions.cpu().numpy().flatten()
            )
            errors[name] = targets_orig - pred_orig
        
        # Error distribution comparison
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Box plot of errors
        error_data = list(errors.values())
        error_labels = list(errors.keys())
        
        axes[0].boxplot(error_data, labels=error_labels)
        axes[0].set_title('Error Distribution Comparison')
        axes[0].set_ylabel('Prediction Error')
        axes[0].grid(True, alpha=0.3)
        
        # Error histograms
        for name, error in errors.items():
            axes[1].hist(error, bins=50, alpha=0.7, label=name, density=True)
        
        axes[1].set_title('Error Histogram Comparison')
        axes[1].set_xlabel('Prediction Error')
        axes[1].set_ylabel('Density')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('plots/error_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return errors

# Simple baseline models for comparison
class SimpleBaseline(nn.Module):
    """Simple linear baseline model"""
    def __init__(self, config):
        super().__init__()
        input_size = len(config.feature_cols) + len(config.time_cols)
        self.linear = nn.Sequential(
            nn.Linear(input_size * config.seq_len, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, config.pred_len)
        )
        self.seq_len = config.seq_len
        self.pred_len = config.pred_len
    
    def forward(self, x):
        B, L, C = x.shape
        x = x.reshape(B, -1)
        output = self.linear(x)
        return output.unsqueeze(-1)  # [B, pred_len, 1]

def main():
    """Main evaluation function"""
    config = Config()
    evaluator = ModelEvaluator(config)
    
    print("Model Comparison Evaluation")
    print("="*50)
    
    # Add models for comparison
    print("Adding models...")
    
    # 1. Add Hybrid Model
    hybrid_model = HybridSolarModel(config)
    evaluator.add_model('Hybrid', hybrid_model)
    
    # 2. Add PatchTST
    patchtst_model = EnhancedPatchTST(config)
    evaluator.add_model('PatchTST', patchtst_model)
    
    # 3. Add Simple Baseline
    baseline_model = SimpleBaseline(config)
    evaluator.add_model('Baseline', baseline_model)
    
    # Load trained models (if available)
    try:
        evaluator.load_model('Hybrid', 'models/best_model.pth')
        print("Loaded trained Hybrid model")
    except FileNotFoundError:
        print("No trained Hybrid model found. Using random weights.")
    
    # Compare models
    comparison_df, all_predictions, all_targets = evaluator.compare_models("test")
    
    # Analyze errors
    print("\nAnalyzing prediction errors...")
    errors = evaluator.analyze_prediction_errors(all_predictions, all_targets)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION SUMMARY")
    print("="*60)
    
    best_model = comparison_df['RMSE'].idxmin()
    best_rmse = comparison_df.loc[best_model, 'RMSE']
    best_r2 = comparison_df.loc[best_model, 'R2']
    
    print(f"Best Model: {best_model}")
    print(f"Best RMSE: {best_rmse:.4f}")
    print(f"Best R²: {best_r2:.4f}")
    
    # Calculate improvements
    if 'Hybrid' in comparison_df.index:
        hybrid_rmse = comparison_df.loc['Hybrid', 'RMSE']
        if 'PatchTST' in comparison_df.index:
            patchtst_rmse = comparison_df.loc['PatchTST', 'RMSE']
            improvement = ((patchtst_rmse - hybrid_rmse) / patchtst_rmse) * 100
            print(f"Hybrid vs PatchTST RMSE Improvement: {improvement:.2f}%")
        
        if 'Baseline' in comparison_df.index:
            baseline_rmse = comparison_df.loc['Baseline', 'RMSE']
            improvement = ((baseline_rmse - hybrid_rmse) / baseline_rmse) * 100
            print(f"Hybrid vs Baseline RMSE Improvement: {improvement:.2f}%")
    
    print("\nResults saved to:")
    print("- results/model_comparison.csv")
    print("- plots/metrics_comparison.png")
    print("- plots/scatter_comparison.png")
    print("- plots/timeseries_comparison.png")
    print("- plots/error_analysis.png")

if __name__ == "__main__":
    main() 