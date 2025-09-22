#!/bin/bash

# Single Horizon Ablation Study
# This script runs all ablation variants for a specific forecast horizon

if [ $# -eq 0 ]; then
    echo "Usage: $0 <pred_len>"
    echo "Example: $0 96"
    echo "Available forecast horizons: 96, 192, 336, 720"
    exit 1
fi

pred_len=$1

echo "==============================================="
echo "SINGLE HORIZON ABLATION STUDY"
echo "Forecast Horizon: $pred_len time steps"
echo "==============================================="

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

# Model variants for ablation study
declare -a models=(
    "PatchXFormer"
    "PatchXFormer_NoFreqAttention" 
    "PatchXFormer_NoAdaptiveNorm"
    "PatchXFormer_NoEnhancedEmbedding"
    "PatchXFormer_NoHybridEncoder"
    "PatchXFormer_BasicPatchModel"
)

# Model descriptions
declare -A descriptions
descriptions["PatchXFormer"]="Full PatchXFormer (All Components)"
descriptions["PatchXFormer_NoFreqAttention"]="Without Frequency-Enhanced Attention"
descriptions["PatchXFormer_NoAdaptiveNorm"]="Without Adaptive Normalization"
descriptions["PatchXFormer_NoEnhancedEmbedding"]="Without Enhanced Patch Embedding"
descriptions["PatchXFormer_NoHybridEncoder"]="Without Hybrid Encoder (No Cross-Attention)"
descriptions["PatchXFormer_BasicPatchModel"]="Basic Patch Model (No Enhancements)"

# Run all model variants for the specified horizon
for model in "${models[@]}"; do
    echo ""
    echo "=============================================="
    echo "Testing: ${descriptions[$model]}"
    echo "Model: $model"
    echo "Forecast Horizon: $pred_len time steps"
    echo "=============================================="
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/$data_name/ \
        --data_path train.csv \
        --model_id ${data_name}_${pred_len} \
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
        --des "horizon_${pred_len}_${model}" \
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
echo "SINGLE HORIZON ABLATION STUDY COMPLETED"
echo "Forecast Horizon: $pred_len time steps"
echo "==============================================="
echo "Check ./results/ directory for results matching pattern: *horizon_${pred_len}_*"
