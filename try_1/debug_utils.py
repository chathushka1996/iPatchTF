import torch
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional

class NaNDebugger:
    """
    Utility class to help debug and monitor NaN issues during training
    """
    
    def __init__(self, log_file: str = "nan_debug.log"):
        self.log_file = log_file
        self.nan_counts = {}
        self.step_count = 0
        
    def log_message(self, message: str):
        """Log message to file and print to console"""
        print(message)
        with open(self.log_file, 'a') as f:
            f.write(f"Step {self.step_count}: {message}\n")
    
    def check_tensor(self, tensor: torch.Tensor, name: str) -> bool:
        """
        Check tensor for NaN/Inf values and log results
        Returns True if tensor is clean, False if issues found
        """
        if tensor is None:
            self.log_message(f"WARNING: {name} is None")
            return False
        
        # Check for NaN
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        
        if nan_count > 0 or inf_count > 0:
            self.log_message(f"ISSUE: {name} - NaN: {nan_count}, Inf: {inf_count}")
            self.log_message(f"  Shape: {tensor.shape}")
            self.log_message(f"  Min: {tensor.min().item()}, Max: {tensor.max().item()}")
            self.log_message(f"  Mean: {tensor.mean().item()}, Std: {tensor.std().item()}")
            
            # Store in counts
            if name not in self.nan_counts:
                self.nan_counts[name] = {'nan': 0, 'inf': 0}
            self.nan_counts[name]['nan'] += nan_count
            self.nan_counts[name]['inf'] += inf_count
            
            return False
        
        return True
    
    def check_model_outputs(self, outputs: Dict[str, Any]) -> bool:
        """Check all model outputs for NaN/Inf values"""
        all_clean = True
        
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor):
                if not self.check_tensor(value, f"output_{key}"):
                    all_clean = False
            elif isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, torch.Tensor):
                        if not self.check_tensor(sub_value, f"output_{key}_{sub_key}"):
                            all_clean = False
        
        return all_clean
    
    def check_gradients(self, model: torch.nn.Module) -> bool:
        """Check model gradients for NaN/Inf values"""
        all_clean = True
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                if not self.check_tensor(param.grad, f"grad_{name}"):
                    all_clean = False
        
        return all_clean
    
    def check_data_batch(self, batch: Dict[str, torch.Tensor]) -> bool:
        """Check input data batch for NaN/Inf values"""
        all_clean = True
        
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                if not self.check_tensor(value, f"input_{key}"):
                    all_clean = False
        
        return all_clean
    
    def save_problematic_data(self, data: torch.Tensor, name: str):
        """Save problematic data to file for analysis"""
        if torch.isnan(data).any() or torch.isinf(data).any():
            filename = f"problematic_data_{name}_step_{self.step_count}.pt"
            torch.save(data.cpu(), filename)
            self.log_message(f"Saved problematic data to {filename}")
    
    def generate_report(self) -> str:
        """Generate summary report of NaN issues"""
        report = f"\n=== NaN Debug Report (Steps: {self.step_count}) ===\n"
        
        if not self.nan_counts:
            report += "No NaN/Inf issues detected!\n"
        else:
            report += "NaN/Inf Issues Summary:\n"
            for name, counts in self.nan_counts.items():
                report += f"  {name}: NaN={counts['nan']}, Inf={counts['inf']}\n"
        
        report += "="*50 + "\n"
        return report
    
    def increment_step(self):
        """Increment step counter"""
        self.step_count += 1

def validate_dataset(dataloader, config, max_batches: int = 10):
    """
    Validate dataset for NaN/Inf values before training
    """
    print("Validating dataset for NaN/Inf values...")
    debugger = NaNDebugger("dataset_validation.log")
    
    issues_found = 0
    total_samples = 0
    
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx >= max_batches:
            break
        
        debugger.increment_step()
        total_samples += batch['input'].shape[0]
        
        # Check inputs
        if not debugger.check_tensor(batch['input'], f"batch_{batch_idx}_input"):
            issues_found += 1
            debugger.save_problematic_data(batch['input'], f"input_batch_{batch_idx}")
        
        # Check targets
        if not debugger.check_tensor(batch['target'], f"batch_{batch_idx}_target"):
            issues_found += 1
            debugger.save_problematic_data(batch['target'], f"target_batch_{batch_idx}")
    
    print(f"Dataset validation complete:")
    print(f"  Batches checked: {min(max_batches, len(dataloader))}")
    print(f"  Total samples: {total_samples}")
    print(f"  Issues found: {issues_found}")
    print(debugger.generate_report())
    
    return issues_found == 0

def safe_tensor_operation(func, *args, fallback_value=0.0, operation_name="unknown"):
    """
    Safely execute tensor operations with fallback for NaN/Inf results
    """
    try:
        result = func(*args)
        
        if torch.isnan(result).any() or torch.isinf(result).any():
            print(f"Warning: {operation_name} produced NaN/Inf, using fallback value {fallback_value}")
            return torch.full_like(result, fallback_value)
        
        return result
    
    except Exception as e:
        print(f"Error in {operation_name}: {e}, using fallback value {fallback_value}")
        if args:
            return torch.full_like(args[0], fallback_value)
        else:
            return torch.tensor(fallback_value)

def emergency_checkpoint_save(model, optimizer, epoch, loss, filepath="emergency_checkpoint.pth"):
    """
    Save emergency checkpoint when NaN is detected
    """
    try:
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss,
            'emergency_save': True
        }, filepath)
        print(f"Emergency checkpoint saved to {filepath}")
    except Exception as e:
        print(f"Failed to save emergency checkpoint: {e}")

if __name__ == "__main__":
    print("NaN Debugger utility loaded successfully!")
    print("Available functions:")
    print("  - NaNDebugger: Main debugging class")
    print("  - validate_dataset: Validate dataset before training")
    print("  - safe_tensor_operation: Safe tensor operations with fallbacks")
    print("  - emergency_checkpoint_save: Save checkpoint when issues occur") 