#!/bin/bash

# PatchXFormer WITHOUT Frequency-Enhanced Attention
# This script runs PatchXFormer with frequency-enhanced attention component removed

echo "==============================================="
echo "RUNNING PATCHXFORMER WITHOUT FREQUENCY-ENHANCED ATTENTION"
echo "Components: Enhanced Embedding + Adaptive Norm + Hybrid Encoder"
echo "Removed: Frequency-Enhanced Attention"
echo "==============================================="

# Base parameters
data_name="sl_t"
model_name="PatchXFormer_NoFreqAttention"
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
        --des "no_freq_attention_pred${pred_len}" \
        --train_epochs $train_epochs \
        --patience 3 \
        --learning_rate $learning_rate \
        --batch_size $batch_size \
        --use_amp \
        --gpu 0
        
    if [ $? -eq 0 ]; then
        echo "✓ Successfully completed PatchXFormer w/o Frequency Attention with pred_len $pred_len"
    else
        echo "✗ Error running PatchXFormer w/o Frequency Attention with pred_len $pred_len"
    fi
done

echo ""
echo "==============================================="
echo "PATCHXFORMER (NO FREQUENCY ATTENTION) EXPERIMENTS COMPLETED"
echo "Expected Impact: 2.7-3.9% MSE increase compared to full model"
echo "==============================================="
