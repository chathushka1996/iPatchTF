#!/bin/bash

# Quick re-training script to generate checkpoint for PatchTimeXer
# This will create the missing checkpoint for evaluation

export CUDA_VISIBLE_DEVICES=0

# Dataset configuration
dataset=sl_t
data_path=solar.csv
model_name=PatchTimeXer
seq_len=96
label_len=48
pred_len=96  # Just train for 96-hour horizon to create checkpoint

# Model configuration
d_model=512
d_ff=2048
n_heads=8
e_layers=3
dropout=0.1
factor=5
patch_len=16
moving_avg=25
batch_size=32
learning_rate=0.0001
train_epochs=10  # Reduced epochs for quick checkpoint generation
patience=5
itr=1
features=M
use_norm=1
enc_in=10
dec_in=10
c_out=10

# Paths
root_path=./dataset/$dataset
model_id_name=solar_enhanced_$dataset
checkpoints=./checkpoints/$model_name
logs_path=./logs

# Create directories
mkdir -p $checkpoints
mkdir -p $logs_path

echo "=== Generating PatchTimeXer Checkpoint ==="
echo "Model: $model_name"
echo "Prediction horizon: $pred_len"
echo "Checkpoint path: $checkpoints"
echo "============================================="

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
    --des 'PatchTimeXer_Solar_Checkpoint' \
    --itr $itr \
    --patch_len $patch_len \
    --moving_avg $moving_avg \
    --use_norm $use_norm \
    --checkpoints $checkpoints \
    > $logs_path/${model_name}_checkpoint_generation.log 2>&1

echo "=== Checkpoint Generation Complete ==="
echo "Checkpoint saved to: $checkpoints"
echo "Log saved to: $logs_path/${model_name}_checkpoint_generation.log"

# Now run evaluation
echo "=== Running Evaluation ==="
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
    --checkpoints $checkpoints \
    > $logs_path/${model_name}_evaluation.log 2>&1

echo "=== Evaluation Complete ==="
echo "Results saved to: $logs_path/${model_name}_evaluation.log" 