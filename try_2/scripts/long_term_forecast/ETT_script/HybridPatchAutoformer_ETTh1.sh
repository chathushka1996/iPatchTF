#!/bin/bash

# HybridPatchAutoformer ETTh1 - Long-term forecasting test script
export CUDA_VISIBLE_DEVICES=0

# Test different prediction lengths with HybridPatchAutoformer
model_name=HybridPatchAutoformer

# ETTh1 dataset configuration
root_path=./dataset/
data_path=ETTh1.csv
data=ETTh1

# Model parameters
d_model=512
d_ff=512
n_heads=8
e_layers=2
dropout=0.1
batch_size=32
learning_rate=0.0001
train_epochs=10
patience=3

# Patch configuration
patch_len=16
stride=8

# Moving average for series decomposition
moving_avg=25

echo "Running HybridPatchAutoformer on ETTh1 dataset..."

for pred_len in 96 192 336 720; do
    echo "Testing prediction length: $pred_len"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ETTh1_${pred_len} \
        --model $model_name \
        --data $data \
        --features M \
        --seq_len 336 \
        --label_len 48 \
        --pred_len $pred_len \
        --e_layers $e_layers \
        --d_layers $e_layers \
        --factor 3 \
        --enc_in 7 \
        --dec_in 7 \
        --c_out 7 \
        --d_model $d_model \
        --d_ff $d_ff \
        --n_heads $n_heads \
        --dropout $dropout \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --moving_avg $moving_avg \
        --patch_len $patch_len \
        --activation 'gelu' \
        --des 'Exp' \
        --itr 1
        
    echo "Completed prediction length: $pred_len"
done

echo "All HybridPatchAutoformer ETTh1 experiments completed!" 