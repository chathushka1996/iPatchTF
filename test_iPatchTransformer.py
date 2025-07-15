#!/usr/bin/env python3

"""
Test script for iPatchTransformer model
This script validates that the model can be instantiated and performs a forward pass correctly.
"""

import torch
import torch.nn as nn
import argparse
from model.iPatchTransformer import Model as iPatchTransformer, calculate_patch_config


class MockConfigs:
    """Mock configuration class for testing"""
    def __init__(self):
        self.seq_len = 96
        self.pred_len = 96
        self.output_attention = False
        self.use_norm = True
        self.patch_len = 16
        self.stride = 8
        self.padding_patch = 'end'
        self.d_model = 256
        self.n_heads = 8
        self.e_layers = 2
        self.d_ff = 256
        self.dropout = 0.1
        self.activation = 'gelu'


def test_patch_config():
    """Test automatic patch configuration calculation"""
    print("=== Testing Patch Configuration ===")
    
    # Test different sequence lengths
    for seq_len in [48, 96, 192, 336]:
        config = calculate_patch_config(seq_len, desired_patches=6)
        print(f"Seq Len: {seq_len} -> Patch Len: {config['patch_len']}, "
              f"Stride: {config['stride']}, Patches: {config['num_patches']}")
    
    print()


def test_model_instantiation():
    """Test model instantiation with different configurations"""
    print("=== Testing Model Instantiation ===")
    
    configs = MockConfigs()
    
    try:
        model = iPatchTransformer(configs)
        print(f"✓ Model instantiated successfully")
        print(f"  - Parameters: {sum(p.numel() for p in model.parameters()):,}")
        print(f"  - Trainable: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
        return model
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        return None


def test_forward_pass(model):
    """Test forward pass with different input sizes"""
    print("=== Testing Forward Pass ===")
    
    if model is None:
        print("✗ Skipping forward pass test (model not available)")
        return
    
    # Test configurations: (batch_size, seq_len, num_variables)
    test_cases = [
        (4, 96, 7),    # ETT dataset
        (2, 96, 21),   # Weather dataset  
        (1, 96, 100),  # Medium dataset
    ]
    
    model.eval()
    
    for batch_size, seq_len, num_vars in test_cases:
        try:
            # Create mock input
            x_enc = torch.randn(batch_size, seq_len, num_vars)
            x_mark_enc = torch.randn(batch_size, seq_len, 4)  # Time features
            x_dec = torch.randn(batch_size, seq_len, num_vars)  # Not used in encoder-only
            x_mark_dec = torch.randn(batch_size, seq_len, 4)   # Not used in encoder-only
            
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
                
            expected_shape = (batch_size, model.pred_len, num_vars)
            
            if output.shape == expected_shape:
                print(f"✓ Forward pass successful: Input {x_enc.shape} -> Output {output.shape}")
            else:
                print(f"✗ Shape mismatch: Expected {expected_shape}, got {output.shape}")
                
        except Exception as e:
            print(f"✗ Forward pass failed for {(batch_size, seq_len, num_vars)}: {e}")
    
    print()


def test_attention_output():
    """Test attention output functionality"""
    print("=== Testing Attention Output ===")
    
    configs = MockConfigs()
    configs.output_attention = True
    
    try:
        model = iPatchTransformer(configs)
        
        # Test input
        x_enc = torch.randn(2, 96, 7)
        x_mark_enc = torch.randn(2, 96, 4)
        x_dec = torch.randn(2, 96, 7)
        x_mark_dec = torch.randn(2, 96, 4)
        
        model.eval()
        with torch.no_grad():
            output, attentions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"✓ Attention output enabled")
        print(f"  - Output shape: {output.shape}")
        print(f"  - Number of attention layers: {len(attentions)}")
        if len(attentions) > 0:
            print(f"  - Attention shape: {attentions[0].shape}")
            
    except Exception as e:
        print(f"✗ Attention output test failed: {e}")
    
    print()


def test_different_patch_configs():
    """Test different patch configurations"""
    print("=== Testing Different Patch Configurations ===")
    
    patch_configs = [
        {'patch_len': 8, 'stride': 4},
        {'patch_len': 16, 'stride': 8},
        {'patch_len': 24, 'stride': 12},
        {'patch_len': 32, 'stride': 16},
    ]
    
    for patch_config in patch_configs:
        try:
            configs = MockConfigs()
            configs.patch_len = patch_config['patch_len']
            configs.stride = patch_config['stride']
            
            model = iPatchTransformer(configs)
            
            # Test forward pass
            x_enc = torch.randn(2, 96, 7)
            x_mark_enc = torch.randn(2, 96, 4)
            x_dec = torch.randn(2, 96, 7)
            x_mark_dec = torch.randn(2, 96, 4)
            
            model.eval()
            with torch.no_grad():
                output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
            
            print(f"✓ Patch config {patch_config}: Output shape {output.shape}")
            
        except Exception as e:
            print(f"✗ Patch config {patch_config} failed: {e}")
    
    print()


def test_memory_usage():
    """Test memory usage comparison"""
    print("=== Testing Memory Usage ===")
    
    import psutil
    import os
    
    process = psutil.Process(os.getpid())
    
    configs = MockConfigs()
    
    # Memory before model creation
    mem_before = process.memory_info().rss / 1024 / 1024  # MB
    
    model = iPatchTransformer(configs)
    
    # Memory after model creation
    mem_after = process.memory_info().rss / 1024 / 1024  # MB
    
    print(f"Memory usage:")
    print(f"  - Before model: {mem_before:.1f} MB")
    print(f"  - After model: {mem_after:.1f} MB")
    print(f"  - Model size: {mem_after - mem_before:.1f} MB")
    
    # Test forward pass memory
    x_enc = torch.randn(4, 96, 21)  # Weather dataset size
    x_mark_enc = torch.randn(4, 96, 4)
    x_dec = torch.randn(4, 96, 21)
    x_mark_dec = torch.randn(4, 96, 4)
    
    model.eval()
    with torch.no_grad():
        output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
    
    mem_after_forward = process.memory_info().rss / 1024 / 1024
    print(f"  - After forward pass: {mem_after_forward:.1f} MB")
    
    print()


def main():
    """Main test function"""
    print("iPatchTransformer Model Tests")
    print("=" * 50)
    
    # Run all tests
    test_patch_config()
    
    model = test_model_instantiation()
    
    test_forward_pass(model)
    
    test_attention_output()
    
    test_different_patch_configs()
    
    try:
        test_memory_usage()
    except ImportError:
        print("Skipping memory test (psutil not available)")
    
    print("=" * 50)
    print("All tests completed!")


if __name__ == "__main__":
    main() 