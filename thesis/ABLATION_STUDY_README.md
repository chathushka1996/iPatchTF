# PatchXFormer Ablation Study

This directory contains implementation files for conducting comprehensive ablation studies on the PatchXFormer architecture. The ablation study systematically evaluates the contribution of each architectural component by removing them individually.

## Ablation Model Variants

### 1. **PatchXFormer** (Full Model)
- **File**: `models/PatchXFormer.py`
- **Components**: All enhancements included
- **Description**: Complete PatchXFormer with all architectural innovations

### 2. **PatchXFormer_NoFreqAttention**
- **File**: `models/PatchXFormer_NoFreqAttention.py` 
- **Components**: Enhanced Embedding + Adaptive Norm + Hybrid Encoder
- **Removed**: Frequency-Enhanced Attention mechanism
- **Purpose**: Evaluate the impact of frequency domain modeling

### 3. **PatchXFormer_NoAdaptiveNorm**
- **File**: `models/PatchXFormer_NoAdaptiveNorm.py`
- **Components**: Enhanced Embedding + Frequency Attention + Hybrid Encoder  
- **Removed**: Adaptive Normalization layers
- **Purpose**: Evaluate the impact of dynamic normalization

### 4. **PatchXFormer_NoEnhancedEmbedding**
- **File**: `models/PatchXFormer_NoEnhancedEmbedding.py`
- **Components**: Frequency Attention + Adaptive Norm + Hybrid Encoder
- **Removed**: Enhanced Patch Embedding (Xavier init, global tokens)
- **Purpose**: Evaluate the impact of improved patch embedding

### 5. **PatchXFormer_NoHybridEncoder**
- **File**: `models/PatchXFormer_NoHybridEncoder.py`
- **Components**: Enhanced Embedding + Frequency Attention + Adaptive Norm
- **Removed**: Hybrid Encoder (cross-attention for exogenous features)
- **Purpose**: Evaluate the impact of exogenous variable integration

### 6. **PatchXFormer_BasicPatchModel**
- **File**: `models/PatchXFormer_BasicPatchModel.py`
- **Components**: Basic patching only
- **Removed**: ALL enhancements
- **Purpose**: Baseline comparison showing impact of all innovations

## Running Ablation Studies

### Option 1: Run Complete Ablation Study
```bash
# Run all ablation variants across all forecast horizons
bash run_ablation_components.sh
```

### Option 2: Run Single Ablation Experiment
```bash
# Run specific ablation variant
python run_single_ablation.py \
    --ablation_model PatchXFormer_NoFreqAttention \
    --pred_len 96 \
    --train_epochs 10
```

### Option 3: Run Python Ablation Suite
```bash
# Run comprehensive Python-based ablation study
python run_ablation_study.py
```

## Analyzing Results

### Collect and Analyze Results
```bash
# Analyze ablation study results
python analyze_ablation_results.py
```

This will generate:
- `ablation_study_analysis.png` - Visualization heatmaps
- `ablation_analysis_report.txt` - Detailed text report
- `ablation_detailed_results.csv` - Raw results data
- `ablation_component_impact.csv` - Component impact analysis

## Expected Results Structure

Based on the thesis ablation study, expected performance degradation when components are removed:

| Component Removed | MSE Degradation (%) | Impact Level |
|-------------------|-------------------|--------------|
| Enhanced Embedding | 5.0-6.3% | High |
| Frequency Attention | 2.7-3.9% | Medium-High |
| Hybrid Encoder | 3.5-4.9% | High |
| Adaptive Norm | 1.6-2.0% | Medium |
| All Enhancements | 8.5-10.4% | Very High |

## Model Architecture Components

### Enhanced Patch Embedding
- **Xavier initialization** for better gradient flow
- **Learnable global tokens** for global context
- **Enhanced positional encoding** with learnable parameters

### Frequency-Enhanced Attention  
- **Dual-domain attention** (time + frequency)
- **FFT-based operations** for spectral modeling
- **Adaptive frequency weighting**

### Adaptive Normalization
- **Dynamic parameter adjustment** based on input statistics
- **Statistical adaptation** for non-stationary data
- **Improved training stability**

### Hybrid Encoder Architecture
- **Self-attention** for temporal modeling
- **Cross-attention** for exogenous feature integration
- **Multi-head attention** with optimal configuration

### Enhanced Prediction Head
- **Multi-path architecture** with residual connections
- **Improved gradient flow**
- **Better prediction stability**

## File Structure

```
models/
├── PatchXFormer.py                    # Full model
├── PatchXFormer_NoFreqAttention.py    # w/o frequency attention
├── PatchXFormer_NoAdaptiveNorm.py     # w/o adaptive normalization
├── PatchXFormer_NoEnhancedEmbedding.py # w/o enhanced embedding
├── PatchXFormer_NoHybridEncoder.py    # w/o hybrid encoder
├── PatchXFormer_BasicPatchModel.py    # basic patch model
└── __init__.py                        # Model imports

Scripts/
├── run_ablation_components.sh         # Bash script for full ablation
├── run_single_ablation.py            # Python script for single experiment
├── run_ablation_study.py             # Python script for full ablation
├── analyze_ablation_results.py       # Results analysis script
└── ABLATION_STUDY_README.md          # This file
```

## Usage Examples

### Run specific component ablation:
```bash
python run_single_ablation.py --ablation_model PatchXFormer_NoFreqAttention --pred_len 96
```

### Run full ablation study:
```bash
bash run_ablation_components.sh
```

### Analyze results:
```bash
python analyze_ablation_results.py
```

## Notes

1. **Training Time**: Each ablation variant requires separate training, so complete ablation study may take several hours depending on hardware.

2. **GPU Memory**: Ensure sufficient GPU memory for the experiments. Reduce batch size if needed.

3. **Results Location**: Results are saved in `./results/` directory with naming pattern `*ablation_*`.

4. **Reproducibility**: Set random seeds for reproducible results across ablation experiments.

5. **Component Dependencies**: Some components may have interdependencies, so removing multiple components simultaneously may show non-additive effects.

## Expected Findings

The ablation study should demonstrate:
- **Enhanced Embedding** has the highest individual impact
- **Frequency Attention** is crucial for long-term forecasting
- **Hybrid Encoder** significantly improves multivariate modeling
- **Adaptive Normalization** provides stability improvements
- **Combined effect** exceeds sum of individual contributions
