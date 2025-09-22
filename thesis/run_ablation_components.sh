#!/bin/bash

# PatchXFormer Ablation Study Script
# Tests the impact of each architectural component by systematically removing them

echo "==============================================="
echo "PATCHXFORMER ABLATION STUDY"
echo "==============================================="

# Base parameters
data_name="sl_t"
data_path="train.csv"
seq_len=96
enc_in=9
d_model=256
n_heads=8
e_layers=3
d_ff=1024
batch_size=32
learning_rate=0.0001
train_epochs=10

# Prediction horizons to test
pred_lens=(96 192 336 720)

# Model variants for ablation study
models=(
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

# Create results directory
mkdir -p ablation_results

# Run ablation study
for model in "${models[@]}"; do
    echo ""
    echo "=============================================="
    echo "Testing: ${descriptions[$model]}"
    echo "Model: $model"
    echo "=============================================="
    
    for pred_len in "${pred_lens[@]}"; do
        echo ""
        echo "Forecast Horizon: $pred_len time steps"
        echo "----------------------------------------------"
        
        # Run the experiment
        python -u run.py \
            --task_name long_term_forecast \
            --is_training 1 \
            --root_path ./dataset/$data_name/ \
            --data_path $data_path \
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
            --des "ablation_${model}_pred${pred_len}" \
            --train_epochs $train_epochs \
            --patience 3 \
            --learning_rate $learning_rate \
            --batch_size $batch_size \
            --use_amp \
            --gpu 0
            
        if [ $? -eq 0 ]; then
            echo "✓ Successfully completed $model with pred_len $pred_len"
        else
            echo "✗ Error running $model with pred_len $pred_len"
        fi
    done
done

echo ""
echo "==============================================="
echo "ABLATION STUDY COMPLETED"
echo "==============================================="
echo "Check the ./results/ directory for detailed results"
echo "Look for files matching pattern: *ablation_*"
