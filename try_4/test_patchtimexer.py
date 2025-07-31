import torch
import torch.nn as nn
import sys
import os

# Add the current directory to the path
sys.path.append('.')

# Import the PatchTimeXer model
try:
    from models.PatchTimeXer import Model as PatchTimeXer
    print("✓ Successfully imported PatchTimeXer model")
except ImportError as e:
    print(f"✗ Failed to import PatchTimeXer: {e}")
    sys.exit(1)

# Create a mock configuration class
class MockConfig:
    def __init__(self):
        # Task configuration
        self.task_name = 'long_term_forecast'
        self.seq_len = 96
        self.pred_len = 96
        self.enc_in = 10  # Number of input features
        
        # Model architecture
        self.d_model = 256  # Reduced for testing
        self.d_ff = 512
        self.n_heads = 8
        self.e_layers = 2  # Reduced for testing
        self.dropout = 0.1
        self.factor = 5
        
        # PatchTimeXer specific
        self.patch_len = 16
        self.stride = 8
        self.scales = [1, 2, 4]
        self.moving_avg = 25
        self.use_norm = True
        
        # Embedding configuration
        self.embed = 'timeF'
        self.freq = 'h'
        self.activation = 'gelu'

def test_model_initialization():
    """Test if the model can be initialized correctly"""
    print("\n=== Testing Model Initialization ===")
    
    config = MockConfig()
    
    try:
        model = PatchTimeXer(config)
        print("✓ Model initialized successfully")
        
        # Print model parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        return model
    except Exception as e:
        print(f"✗ Model initialization failed: {e}")
        return None

def test_forward_pass(model):
    """Test if the model can perform a forward pass"""
    print("\n=== Testing Forward Pass ===")
    
    if model is None:
        print("✗ Cannot test forward pass - model is None")
        return False
    
    # Create dummy input data
    batch_size = 2
    seq_len = 96
    pred_len = 96
    n_features = 10
    
    try:
        # Input tensors
        x_enc = torch.randn(batch_size, seq_len, n_features)  # [B, L, D]
        x_mark_enc = torch.randn(batch_size, seq_len, 4)      # Time features
        x_dec = torch.randn(batch_size, pred_len, n_features) # Decoder input (placeholder)
        x_mark_dec = torch.randn(batch_size, pred_len, 4)     # Decoder time features
        
        print(f"Input shape: {x_enc.shape}")
        print(f"Expected output shape: [{batch_size}, {pred_len}, {n_features}]")
        
        # Forward pass
        model.eval()
        with torch.no_grad():
            output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
        
        print(f"Actual output shape: {output.shape}")
        
        # Verify output shape
        expected_shape = (batch_size, pred_len, n_features)
        if output.shape == expected_shape:
            print("✓ Forward pass successful - output shape correct")
            return True
        else:
            print(f"✗ Forward pass failed - expected {expected_shape}, got {output.shape}")
            return False
            
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_component_integration():
    """Test if individual components work correctly"""
    print("\n=== Testing Component Integration ===")
    
    try:
        from models.PatchTimeXer import (
            MultiScalePatchEmbedding,
            SeasonTrendDecomposer,
            InvertedAttentionLayer,
            GlobalContextLayer,
            AdaptiveForecastHead
        )
        print("✓ All components imported successfully")
        
        # Test MultiScalePatchEmbedding
        patch_emb = MultiScalePatchEmbedding(d_model=256, patch_len=16, stride=8, scales=[1, 2, 4])
        x_test = torch.randn(2, 10, 96)  # [B, N, T]
        patches, n_vars = patch_emb(x_test)
        print(f"✓ MultiScalePatchEmbedding: {x_test.shape} -> {patches.shape}, n_vars={n_vars}")
        
        # Test SeasonTrendDecomposer
        decomposer = SeasonTrendDecomposer(moving_avg=25)
        x_test = torch.randn(2, 96, 10)  # [B, L, D]
        seasonal, trend = decomposer(x_test)
        print(f"✓ SeasonTrendDecomposer: {x_test.shape} -> seasonal={seasonal.shape}, trend={trend.shape}")
        
        # Test InvertedAttentionLayer
        inv_attn = InvertedAttentionLayer(d_model=256, n_heads=8)
        x_test = torch.randn(20, 12, 256)  # [B*N, L, D]
        output = inv_attn(x_test)
        print(f"✓ InvertedAttentionLayer: {x_test.shape} -> {output.shape}")
        
        # Test GlobalContextLayer
        global_ctx = GlobalContextLayer(n_vars=10, d_model=256)
        patch_emb = torch.randn(20, 12, 256)  # [B*N, L, D]
        output = global_ctx(patch_emb)
        print(f"✓ GlobalContextLayer: {patch_emb.shape} -> {output.shape}")
        
        print("✓ All component tests passed")
        return True
        
    except Exception as e:
        print(f"✗ Component testing failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("PatchTimeXer Model Testing")
    print("=" * 50)
    
    # Test 1: Model initialization
    model = test_model_initialization()
    
    # Test 2: Forward pass
    forward_success = test_forward_pass(model)
    
    # Test 3: Component integration
    component_success = test_component_integration()
    
    # Summary
    print("\n" + "=" * 50)
    print("Test Summary:")
    print(f"Model initialization: {'✓ PASS' if model is not None else '✗ FAIL'}")
    print(f"Forward pass: {'✓ PASS' if forward_success else '✗ FAIL'}")
    print(f"Component integration: {'✓ PASS' if component_success else '✗ FAIL'}")
    
    if model is not None and forward_success and component_success:
        print("\n🎉 All tests passed! PatchTimeXer is ready for training.")
        return True
    else:
        print("\n❌ Some tests failed. Please check the implementation.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 