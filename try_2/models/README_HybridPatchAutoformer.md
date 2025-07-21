# HybridPatchAutoformer

A hybrid time series forecasting model that combines the best features from **PatchTST** and **Autoformer** to achieve superior performance across multiple time series tasks.

## Overview

The HybridPatchAutoformer integrates two state-of-the-art time series forecasting approaches:

1. **PatchTST's** efficient patch-based tokenization and instance normalization
2. **Autoformer's** series decomposition and AutoCorrelation mechanism

This combination results in a model that can capture both local patterns (through patches) and global temporal dependencies (through AutoCorrelation) while handling non-stationarity effectively.

## Key Features

### 🔧 **Dual Attention Mechanism**
- **AutoCorrelation**: Captures period-based dependencies using FFT-based correlation
- **Self-Attention**: Captures general sequence dependencies  
- **Attention Fusion**: Intelligently combines both attention outputs

### 📦 **Patch-Based Processing**
- Converts time series into patches/tokens for efficient processing
- Reduces computational complexity from O(L²) to O(P²) where P << L
- Enables better handling of long sequences

### 📈 **Series Decomposition**
- Explicitly separates trend and seasonal components
- Progressive decomposition at each layer for refined feature extraction
- Better interpretability and modeling of time series components

### 🔄 **Instance Normalization**
- Handles non-stationary time series data
- Each sample is normalized independently
- Reduces distribution shift issues

### 🎯 **Multi-Task Support**
- Long-term forecasting
- Short-term forecasting  
- Imputation
- Anomaly detection
- Classification

## Architecture

```
Input Time Series
       ↓
Instance Normalization (PatchTST)
       ↓
Series Decomposition (Autoformer)
       ↓
Patch Embedding (PatchTST)
       ↓
┌─────────────────────────────────┐
│    HybridEncoderLayer           │
│  ┌─────────────────────────────┐│
│  │ AutoCorrelation + Self-Attn ││
│  │         ↓                   ││
│  │   Attention Fusion          ││
│  │         ↓                   ││
│  │ Progressive Decomposition   ││
│  │         ↓                   ││
│  │    Feed Forward             ││
│  └─────────────────────────────┘│
└─────────────────────────────────┘
       ↓
Prediction Head
       ↓
De-normalization
       ↓
Output Predictions
```

## Model Components

### HybridEncoderLayer
The core building block that combines:
- **Dual Attention**: Both AutoCorrelation and Self-Attention
- **Attention Fusion**: Linear layer to combine attention outputs
- **Progressive Decomposition**: Series decomposition after attention
- **Feed Forward Network**: Standard transformer FFN with residual connections

### AdaptiveHead
An intelligent prediction head that can handle both patch-based and sequence-based representations for different tasks.

## Usage

### Basic Usage

```python
from models.HybridPatchAutoformer import Model as HybridPatchAutoformer

# Create model configuration
class Config:
    def __init__(self):
        # Task configuration
        self.task_name = 'long_term_forecast'
        self.seq_len = 336
        self.pred_len = 96
        
        # Model architecture
        self.enc_in = 7  # Number of input features
        self.d_model = 512
        self.d_ff = 512
        self.n_heads = 8
        self.e_layers = 2
        self.dropout = 0.1
        self.activation = 'gelu'
        
        # Hybrid-specific
        self.moving_avg = 25  # For series decomposition
        self.embed = 'timeF'
        self.freq = 'h'

# Initialize model
config = Config()
model = HybridPatchAutoformer(config)

# Forward pass
output = model(x_enc, x_mark_enc, x_dec, x_mark_dec)
```

### Training with Existing Framework

The model is fully integrated with the existing training framework. Use it like any other model:

```bash
python run.py \
    --task_name long_term_forecast \
    --model HybridPatchAutoformer \
    --data ETTh1 \
    --seq_len 336 \
    --pred_len 96 \
    --d_model 512 \
    --n_heads 8 \
    --e_layers 2 \
    --moving_avg 25 \
    --patch_len 16 \
    --batch_size 32 \
    --learning_rate 0.0001
```

## Parameters

### Core Parameters
- `d_model`: Model dimension (default: 512)
- `d_ff`: Feed-forward dimension (default: 512) 
- `n_heads`: Number of attention heads (default: 8)
- `e_layers`: Number of encoder layers (default: 2)
- `dropout`: Dropout rate (default: 0.1)

### Hybrid-Specific Parameters
- `moving_avg`: Window size for series decomposition (default: 25)
- `patch_len`: Length of each patch (default: 16)
- `stride`: Stride for patch extraction (default: 8)
- `factor`: Attention factor for AutoCorrelation (default: 3)

## Performance Benefits

### Computational Efficiency
- **Patch-based processing**: Reduces sequence length complexity
- **Efficient attention**: AutoCorrelation has O(L log L) complexity
- **Progressive decomposition**: Gradual refinement reduces redundancy

### Modeling Capability
- **Local patterns**: Patches capture fine-grained temporal patterns
- **Global dependencies**: AutoCorrelation captures long-range periodic patterns
- **Non-stationarity**: Instance normalization handles distribution shifts
- **Interpretability**: Series decomposition provides trend/seasonal insights

## Comparison with Base Models

| Feature | PatchTST | Autoformer | HybridPatchAutoformer |
|---------|----------|------------|----------------------|
| Patch Processing | ✅ | ❌ | ✅ |
| Series Decomposition | ❌ | ✅ | ✅ |
| AutoCorrelation | ❌ | ✅ | ✅ |
| Self-Attention | ✅ | ❌ | ✅ |
| Instance Normalization | ✅ | ❌ | ✅ |
| Progressive Decomposition | ❌ | ✅ | ✅ |
| Multi-task Support | ✅ | ✅ | ✅ |

## Testing

Run the test script to validate the model:

```bash
python test_hybrid_model.py
```

This will test:
- Model instantiation
- Forward pass functionality
- Different task configurations
- Output shape validation

## Example Scripts

### ETTh1 Long-term Forecasting
```bash
bash scripts/long_term_forecast/ETT_script/HybridPatchAutoformer_ETTh1.sh
```

### Weather Long-term Forecasting
```bash
bash scripts/long_term_forecast/Weather_script/HybridPatchAutoformer.sh
```

### Custom Configuration
```bash
python run.py \
    --task_name long_term_forecast \
    --model HybridPatchAutoformer \
    --data your_dataset \
    --seq_len 168 \
    --pred_len 24 \
    --d_model 256 \
    --n_heads 4 \
    --e_layers 3 \
    --moving_avg 12 \
    --patch_len 8 \
    --learning_rate 0.001
```

## Implementation Details

### Memory Efficiency
- Uses checkpoint-friendly design for large models
- Efficient tensor operations with proper device placement
- Gradient accumulation support for large batch training

### Numerical Stability
- Careful normalization to prevent exploding/vanishing gradients
- Stable FFT operations in AutoCorrelation
- Robust handling of zero variance in normalization

### Extensibility
- Modular design allows easy modification of components
- Support for custom attention mechanisms
- Configurable decomposition strategies

## Citation

If you use HybridPatchAutoformer in your research, please cite the original papers:

```bibtex
@article{patchist2023,
    title={A Time Series is Worth 64 Words: Long-term Forecasting with Transformers},
    author={Nie, Yuqi and Nguyen, Nam H and Sinthong, Phanwadee and Kalagnanam, Jayant},
    journal={arXiv preprint arXiv:2211.14730},
    year={2023}
}

@article{autoformer2021,
    title={Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting},
    author={Wu, Haixu and Xu, Jiehui and Wang, Jianmin and Long, Mingsheng},
    journal={Advances in Neural Information Processing Systems},
    year={2021}
}
```

## Contributing

Feel free to submit issues and pull requests to improve the model. Key areas for contribution:
- Performance optimizations
- Additional attention mechanisms
- Enhanced decomposition strategies
- Better normalization techniques

## License

This implementation follows the same license as the original Time-Series-Library framework. 