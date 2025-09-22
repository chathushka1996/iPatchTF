#!/bin/bash

# PatchXFormer Basic Patch Model
# This script runs the basic patch model with ALL enhancements removed

echo "==============================================="
echo "RUNNING BASIC PATCH MODEL (NO ENHANCEMENTS)"
echo "Components: Basic patching only"
echo "Removed: ALL PatchXFormer enhancements"
echo "==============================================="

# Base parameters
data_name="sl_t"
model_name="PatchXFormer_BasicPatchModel"
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
pred_lens="96 192 336 720"

for pred_len in $pred_lens; do
    echo ""
    echo "----------------------------------------------"
    echo "Forecast Horizon: $pred_len time steps"
    echo "----------------------------------------------"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path ./dataset/$data_name/ \
        --data_path train.csv \
        --model_id ${data_name}_${pred_len} \
        --model $model_name \
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
        --des "basic_patch_model_pred${pred_len}" \
        --train_epochs $train_epochs \
        --patience 3 \
        --learning_rate $learning_rate \
        --batch_size $batch_size \
        --use_amp \
        --gpu 0
        
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed Basic Patch Model with pred_len $pred_len"
    else
        echo "✗ Error running Basic Patch Model with pred_len $pred_len"
    fi
done

echo ""
echo "==============================================="
echo "BASIC PATCH MODEL EXPERIMENTS COMPLETED"
echo "Expected Impact: 8.5-10.4% MSE increase compared to full model"
echo "This represents the baseline without any PatchXFormer innovations"
echo "==============================================="
