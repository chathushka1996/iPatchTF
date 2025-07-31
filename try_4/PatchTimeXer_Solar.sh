#!/bin/bash

# PatchTimeXer Training Script for Solar Power Forecasting
# Optimized configuration combining best practices from all component models

export CUDA_VISIBLE_DEVICES=0

# Dataset configuration for solar power
dataset=sl_t
data_path=solar.csv
model_name=PatchTimeXer
seq_len=96
label_len=48

# Enhanced model configuration
d_model=512          # Increased model dimension for better representation
d_ff=2048           # Larger feed-forward dimension
n_heads=8           # Optimal number of attention heads
e_layers=3          # Moderate depth to prevent overfitting
dropout=0.1         # Regularization
factor=5            # Attention factor

# Multi-scale patching configuration (inspired by PatchTST + TimeMixer)
patch_len=16        # Patch length for temporal segmentation
stride=8            # Overlapping patches for better coverage
scales="1,2,4"      # Multi-scale processing

# Decomposition configuration (from TimeMixer)
moving_avg=25       # Window for seasonal-trend decomposition

# Training configuration
batch_size=32       # Balanced batch size for stable training
learning_rate=0.0001 # Conservative learning rate
train_epochs=20     # Extended training for convergence
patience=4         # Early stopping patience
itr=1              # Number of iterations

# Enhanced features configuration
features=M          # Multivariate forecasting
use_norm=1         # Use normalization for stability

# Solar-specific configurations
enc_in=10          # Number of input features (adjust based on your data)
dec_in=10          # Decoder input features
c_out=10           # Output features

# Logging and checkpoints
root_path=./dataset/$dataset
model_id_name=solar_enhanced_$dataset
checkpoints=./checkpoints/$model_name
logs_path=./logs

# Create directories if they don't exist
mkdir -p $checkpoints
mkdir -p $logs_path

echo "=== PatchTimeXer Solar Power Forecasting ==="
echo "Model: $model_name"
echo "Dataset: $dataset"
echo "Multi-scale configuration: patches=$patch_len, stride=$stride, scales=$scales"
echo "Model dimensions: d_model=$d_model, layers=$e_layers, heads=$n_heads"
echo "============================================="

# Multi-horizon forecasting for solar power prediction
for pred_len in 96 192 336 720
do
    echo "Training for prediction horizon: $pred_len"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ${model_id_name}_${seq_len}_${pred_len} \
        --model $model_name \
        --data custom \
        --features $features \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers $e_layers \
        --d_layers 1 \
        --factor $factor \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --d_model $d_model \
        --d_ff $d_ff \
        --n_heads $n_heads \
        --dropout $dropout \
        --batch_size $batch_size \
        --learning_rate $learning_rate \
        --train_epochs $train_epochs \
        --patience $patience \
        --des 'PatchTimeXer_Solar' \
        --itr $itr \
        --patch_len $patch_len \
        --moving_avg $moving_avg \
        --use_norm $use_norm \
        # --checkpoints $checkpoints \
        # > $logs_path/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log 2>&1
        
    echo "Completed training for horizon $pred_len. Log saved to $logs_path/${model_name}_${model_id_name}_${seq_len}_${pred_len}.log"
done

echo "=== Training Complete ==="
echo "All prediction horizons trained successfully!"
echo "Logs available in: $logs_path"
echo "Checkpoints saved in: $checkpoints"

# Optional: Run evaluation on all trained models
echo "=== Running Evaluation ==="
for pred_len in 96 192 336 720
do
    echo "Evaluating model for horizon: $pred_len"
    
    python -u run.py \
        --task_name long_term_forecast \
        --is_training 0 \
        --root_path $root_path \
        --data_path $data_path \
        --model_id ${model_id_name}_${seq_len}_${pred_len} \
        --model $model_name \
        --data custom \
        --features $features \
        --seq_len $seq_len \
        --label_len $label_len \
        --pred_len $pred_len \
        --e_layers $e_layers \
        --d_layers 1 \
        --factor $factor \
        --enc_in $enc_in \
        --dec_in $dec_in \
        --c_out $c_out \
        --d_model $d_model \
        --d_ff $d_ff \
        --n_heads $n_heads \
        --dropout $dropout \
        --batch_size $batch_size \
        --des 'PatchTimeXer_Solar_Eval' \
        --itr $itr \
        --patch_len $patch_len \
        --moving_avg $moving_avg \
        --use_norm $use_norm \
        # --checkpoints $checkpoints \
        # > $logs_path/${model_name}_${model_id_name}_${seq_len}_${pred_len}_eval.log 2>&1
        
    echo "Evaluation complete for horizon $pred_len"
done

echo "=== All Tasks Complete ===" 