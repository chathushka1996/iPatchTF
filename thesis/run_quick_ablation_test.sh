#!/bin/bash

# Quick Ablation Test
# This script runs a quick test with reduced epochs for rapid validation

echo "==============================================="
echo "QUICK ABLATION TEST (REDUCED EPOCHS FOR TESTING)"
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
train_epochs=3  # Reduced for quick testing
pred_len=96     # Single horizon for quick test

# Model variants for quick test
declare -a models=(
    "PatchXFormer"
    "PatchXFormer_NoFreqAttention" 
    "PatchXFormer_NoEnhancedEmbedding"
    "PatchXFormer_BasicPatchModel"
)

# Model descriptions
declare -A descriptions
descriptions["PatchXFormer"]="Full PatchXFormer"
descriptions["PatchXFormer_NoFreqAttention"]="w/o Frequency Attention"
descriptions["PatchXFormer_NoEnhancedEmbedding"]="w/o Enhanced Embedding"
descriptions["PatchXFormer_BasicPatchModel"]="Basic Patch Model"

echo "Testing $pred_len time step forecast with $train_epochs epochs"
echo ""

for model in "${models[@]}"; do
    echo "=============================================="
    echo "Testing: ${descriptions[$model]}"
    echo "=============================================="
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/$data_name/ \
        --data_path train.csv \
        --model_id ${data_name}_quick_test \
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
        --des "quick_test_${model}" \
        --train_epochs $train_epochs \
        --patience 2 \
        --learning_rate $learning_rate \
        --batch_size $batch_size \
        --use_amp \
        --gpu 0
        
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed quick test for $model"
    else
        echo "✗ Error running quick test for $model"
    fi
    echo ""
done

echo "==============================================="
echo "QUICK ABLATION TEST COMPLETED"
echo "==============================================="
echo "This was a quick validation test with reduced epochs."
echo "For full ablation study, use the individual model scripts."
echo "Check ./results/ directory for results matching pattern: *quick_test_*"
