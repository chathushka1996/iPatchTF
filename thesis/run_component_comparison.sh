#!/bin/bash

# Component Comparison Script
# This script runs specific component comparisons for detailed analysis

echo "==============================================="
echo "PATCHXFORMER COMPONENT COMPARISON STUDY"
echo "==============================================="

# Check command line arguments
if [ $# -eq 0 ]; then
    echo "Usage: $0 <comparison_type>"
    echo ""
    echo "Available comparison types:"
    echo "  attention     - Compare with/without frequency attention"
    echo "  embedding     - Compare with/without enhanced embedding"
    echo "  normalization - Compare with/without adaptive normalization"
    echo "  encoder       - Compare with/without hybrid encoder"
    echo "  all_vs_basic  - Compare full model vs basic patch model"
    echo ""
    echo "Example: $0 attention"
    exit 1
fi

comparison_type=$1

# Base parameters
data_name="sl_t"
seq_len=96
enc_in=9
d_model=256
n_heads=8
e_layers=3
d_ff=1024
batch_size=32
learning_rate=0.0001
train_epochs=10
pred_len=96  # Single horizon for comparison

case $comparison_type in
    "attention")
        echo "COMPARING: Frequency-Enhanced Attention Impact"
        echo "Models: PatchXFormer vs PatchXFormer_NoFreqAttention"
        models=("PatchXFormer" "PatchXFormer_NoFreqAttention")
        ;;
    "embedding")
        echo "COMPARING: Enhanced Patch Embedding Impact"
        echo "Models: PatchXFormer vs PatchXFormer_NoEnhancedEmbedding"
        models=("PatchXFormer" "PatchXFormer_NoEnhancedEmbedding")
        ;;
    "normalization")
        echo "COMPARING: Adaptive Normalization Impact"
        echo "Models: PatchXFormer vs PatchXFormer_NoAdaptiveNorm"
        models=("PatchXFormer" "PatchXFormer_NoAdaptiveNorm")
        ;;
    "encoder")
        echo "COMPARING: Hybrid Encoder Impact"
        echo "Models: PatchXFormer vs PatchXFormer_NoHybridEncoder"
        models=("PatchXFormer" "PatchXFormer_NoHybridEncoder")
        ;;
    "all_vs_basic")
        echo "COMPARING: All Enhancements vs Basic Model"
        echo "Models: PatchXFormer vs PatchXFormer_BasicPatchModel"
        models=("PatchXFormer" "PatchXFormer_BasicPatchModel")
        ;;
    *)
        echo "Invalid comparison type: $comparison_type"
        echo "Use --help for available options"
        exit 1
        ;;
esac

echo "Forecast Horizon: $pred_len time steps"
echo "Training Epochs: $train_epochs"
echo "==============================================="

# Run comparison
for model in "${models[@]}"; do
    echo ""
    echo "----------------------------------------------"
    echo "Running: $model"
    echo "----------------------------------------------"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/$data_name/ \
        --data_path train.csv \
        --model_id ${data_name}_comparison \
        --model $model \
        --data $data_name \
        --features M \
        --seq_len $seq_len \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers $e_layers \
        --d_layers 1 \
        --factor 1 \
        --enc_in $enc_in \
        --dec_in $enc_in \
        --c_out $enc_in \
        --d_model $d_model \
        --n_heads $n_heads \
        --d_ff $d_ff \
        --dropout 0.1 \
        --fc_dropout 0.1 \
        --head_dropout 0.1 \
        --patch_len 16 \
        --stride 8 \
        --des "comparison_${comparison_type}_${model}" \
        --train_epochs $train_epochs \
        --patience 3 \
        --learning_rate $learning_rate \
        --batch_size $batch_size \
        --use_amp \
        --gpu 0
        
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed $model"
    else
        echo "✗ Error running $model"
    fi
done

echo ""
echo "==============================================="
echo "COMPONENT COMPARISON COMPLETED"
echo "Comparison Type: $comparison_type"
echo "==============================================="
echo "Check ./results/ directory for results matching pattern: *comparison_${comparison_type}_*"
