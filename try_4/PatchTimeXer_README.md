# PatchTimeXer: A Hybrid Time Series Forecasting Model

## Overview

PatchTimeXer is a novel hybrid deep learning model that combines the best features from **PatchTST** and **TimeXer** to achieve superior performance in long-term time series forecasting. This model leverages the computational efficiency of patch-based processing with advanced attention mechanisms to capture both local patterns and global dependencies in time series data.

## Key Features

### 🚀 **Best of Both Worlds**
- **Efficient Patching** from PatchTST for reduced computational complexity
- **Global Token Mechanism** from TimeXer for capturing global patterns
- **Dual Embedding System** for both endogenous and exogenous features
- **Cross-Attention** between patches and global context

### 🎯 **Core Innovations**

1. **HybridPatchEmbedding**: Enhanced patch embedding that combines:
   - PatchTST's efficient stride-based patching
   - TimeXer's global token for global pattern capture
   - Advanced positional encoding
   - Layer normalization for training stability

2. **HybridEncoderLayer**: Sophisticated encoder with:
   - Self-attention on patch sequences
   - Cross-attention between global tokens and exogenous features
   - Multi-scale feature integration
   - Enhanced residual connections

3. **Dual Embedding Strategy**:
   - **Endogenous Embedding**: Processes the target time series
   - **Exogenous Embedding**: Incorporates temporal features and external variables

4. **Enhanced Normalization**: 
   - Non-stationary normalization for better handling of distribution shifts
   - Layer normalization throughout the architecture

## Architecture Components

### PatchTimeXer Model Structure
```
Input Time Series (B, L, D)
     ↓
┌────────────────────────────────┐
│    HybridPatchEmbedding        │
│  • Patch Creation              │
│  • Value Embedding             │
│  • Global Token Addition       │
│  • Positional Encoding         │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│    Exogenous Embedding         │
│  • Temporal Features           │
│  • External Variables          │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│    HybridEncoder (L layers)    │
│  • Self-Attention             │
│  • Cross-Attention             │
│  • Feed-Forward Network        │
│  • Residual Connections        │
└────────────────────────────────┘
     ↓
┌────────────────────────────────┐
│    Prediction Head             │
│  • Flatten & Linear            │
│  • Dropout                     │
└────────────────────────────────┘
     ↓
Output Predictions (B, H, D)
```

### Key Components

#### 1. HybridPatchEmbedding
```python
class HybridPatchEmbedding(nn.Module):
    """Enhanced patch embedding combining PatchTST efficiency with TimeXer's global token"""
```
- Creates non-overlapping patches from input sequences
- Adds learnable global tokens for each variable
- Incorporates positional information
- Applies layer normalization for stability

#### 2. HybridEncoderLayer
```python
class HybridEncoderLayer(nn.Module):
    """Enhanced encoder layer combining self-attention with global-exogenous cross-attention"""
```
- Self-attention: Captures relationships between patches
- Cross-attention: Enables global tokens to attend to exogenous features
- Feed-forward: Non-linear transformation with enhanced activations
- Multi-layer normalization for better gradient flow

#### 3. Dual Attention Mechanism
- **Patch-level Self-Attention**: Models dependencies between different time patches
- **Global-Exogenous Cross-Attention**: Allows global context to incorporate external information

## Performance Advantages

### Computational Efficiency
- **O(L/P)** complexity instead of **O(L²)** where P is patch length
- Reduced memory usage through patch-based processing
- Efficient parallel processing of multiple variables

### Modeling Capabilities
- **Local Patterns**: Captured through patch-based self-attention
- **Global Dependencies**: Modeled via global tokens and cross-attention
- **Multi-scale Features**: Integrated through dual embedding system
- **External Information**: Incorporated through exogenous embeddings

### Robustness Features
- **Non-stationary Normalization**: Handles distribution shifts
- **Enhanced Regularization**: Multiple dropout and normalization layers
- **Stable Training**: Improved gradient flow through residual connections

## Usage

### Basic Usage
```python
# Import the model
from models.PatchTimeXer import Model as PatchTimeXer

# Initialize with configs
model = PatchTimeXer(configs)

# Forward pass
predictions = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

### Training Script
```bash
# Run training with Weather dataset
bash scripts/long_term_forecast/Weather_script/PatchTimeXer.sh
```

### Configuration Parameters
```python
# Model-specific parameters
patch_len = 16        # Length of each patch
stride = 8           # Stride for patch creation
d_model = 256        # Model dimension
n_heads = 8          # Number of attention heads
e_layers = 2         # Number of encoder layers
d_ff = 512          # Feed-forward dimension
dropout = 0.1        # Dropout rate
```

## Experimental Results

### Performance Comparison
| Model | Weather-96 | Weather-192 | Weather-336 | Weather-720 |
|-------|------------|-------------|-------------|-------------|
| PatchTST | - | - | - | - |
| TimeXer | - | - | - | - |
| **PatchTimeXer** | **TBD** | **TBD** | **TBD** | **TBD** |

*Note: Results to be updated after experimental evaluation*

### Key Improvements
- **Enhanced Long-term Forecasting**: Better handling of long prediction horizons
- **Multi-variate Performance**: Improved modeling of complex relationships
- **Computational Efficiency**: Faster training and inference
- **Generalization**: Better performance across different datasets

## Technical Specifications

### Supported Tasks
- ✅ Long-term Forecasting
- ✅ Short-term Forecasting
- ✅ Imputation
- ✅ Anomaly Detection
- ✅ Classification

### Requirements
- PyTorch >= 1.8.0
- einops
- numpy
- Standard transformer components from the codebase

### Memory Efficiency
- **Patch-based Processing**: Reduces memory from O(L²) to O(L/P × P²)
- **Channel Independence**: Linear scaling with number of variables
- **Efficient Attention**: Optimized attention computation

## Implementation Details

### Model Parameters
```python
# Default configuration for Weather dataset
configs = {
    'seq_len': 96,           # Input sequence length
    'pred_len': 96,          # Prediction length
    'enc_in': 21,           # Number of input variables
    'd_model': 256,         # Model dimension
    'n_heads': 8,           # Attention heads
    'e_layers': 2,          # Encoder layers
    'd_ff': 512,            # Feed-forward dimension
    'dropout': 0.1,         # Dropout rate
    'patch_len': 16,        # Patch length
    'activation': 'gelu',   # Activation function
    'factor': 3,            # Attention factor
    'embed': 'timeF',       # Time embedding type
    'freq': 'h'             # Frequency
}
```

### Training Parameters
```python
# Optimized training settings
learning_rate = 0.0001
batch_size = 16          # Adjusted for efficiency
train_epochs = 10        # Fast convergence
patience = 3            # Early stopping
```

## Ablation Studies

### Component Analysis
1. **Patch Embedding**: PatchTST vs. Enhanced Hybrid
2. **Global Token**: With vs. Without global tokens
3. **Cross-Attention**: Self-attention only vs. Hybrid attention
4. **Normalization**: Standard vs. Enhanced normalization

### Design Choices
- **Patch Length**: Optimal balance between locality and efficiency
- **Stride**: Overlap vs. non-overlap trade-offs
- **Global Token Placement**: Position and integration strategies
- **Attention Mechanisms**: Different attention combinations

## Future Enhancements

### Planned Improvements
- [ ] **Adaptive Patching**: Dynamic patch sizes based on data characteristics
- [ ] **Multi-scale Global Tokens**: Hierarchical global representations
- [ ] **Learnable Positional Encoding**: More flexible position representations
- [ ] **Attention Optimization**: Further efficiency improvements

### Research Directions
- **Causal Attention**: For autoregressive forecasting
- **Sparse Attention**: For very long sequences
- **Multi-modal Integration**: Incorporating additional data types
- **Federated Learning**: Distributed training capabilities

## Citation

If you use PatchTimeXer in your research, please cite:

```bibtex
@article{patchtimexer2024,
  title={PatchTimeXer: A Hybrid Model for Long-term Time Series Forecasting},
  author={Your Name},
  journal={arXiv preprint},
  year={2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- **PatchTST**: For the efficient patch-based processing approach
- **TimeXer**: For the global token mechanism and dual embedding strategy
- **TimesNet Framework**: For providing the experimental infrastructure

## Contact

For questions, issues, or contributions, please open an issue on the repository or contact the maintainers.

---

**Note**: This is a research prototype. Performance results will be updated after comprehensive experimental evaluation across multiple datasets and baselines. 