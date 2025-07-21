#!/bin/bash

# HybridPatchAutoformer Weather - Long-term forecasting script
export CUDA_VISIBLE_DEVICES=0

# Test different prediction lengths with HybridPatchAutoformer
model_name=HybridPatchAutoformer

# Weather dataset configuration
root_path=./dataset/weather/
data_path=weather.csv
data=custom

# Model parameters optimized for Weather dataset
d_model=256
d_ff=512
n_heads=8
e_layers=2
d_layers=1
dropout=0.1
batch_size=64
learning_rate=0.0001
train_epochs=5
patience=3

# Patch configuration
patch_len=16
stride=8

# Moving average for series decomposition (optimized for weather patterns)
moving_avg=25

echo "Running HybridPatchAutoformer on Weather dataset..."

# Weather 96_96
echo "Testing Weather prediction length: 96"
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id weather_96_96 \
    --model $model_name \
    --data $data \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 96 \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
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

echo "Completed Weather prediction length: 96"

# Weather 96_192
echo "Testing Weather prediction length: 192"
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id weather_96_192 \
    --model $model_name \
    --data $data \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 192 \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_model $d_model \
    --d_ff $d_ff \
    --n_heads 16 \
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

echo "Completed Weather prediction length: 192"

# Weather 96_336
echo "Testing Weather prediction length: 336"
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id weather_96_336 \
    --model $model_name \
    --data $data \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 336 \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_model $d_model \
    --d_ff $d_ff \
    --n_heads $n_heads \
    --dropout $dropout \
    --batch_size 128 \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --moving_avg $moving_avg \
    --patch_len $patch_len \
    --activation 'gelu' \
    --des 'Exp' \
    --itr 1

echo "Completed Weather prediction length: 336"

# Weather 96_720
echo "Testing Weather prediction length: 720"
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path $root_path \
    --data_path $data_path \
    --model_id weather_96_720 \
    --model $model_name \
    --data $data \
    --features M \
    --seq_len 96 \
    --label_len 48 \
    --pred_len 720 \
    --e_layers $e_layers \
    --d_layers $d_layers \
    --factor 3 \
    --enc_in 21 \
    --dec_in 21 \
    --c_out 21 \
    --d_model $d_model \
    --d_ff $d_ff \
    --n_heads $n_heads \
    --dropout $dropout \
    --batch_size 128 \
    --learning_rate $learning_rate \
    --train_epochs $train_epochs \
    --patience $patience \
    --moving_avg $moving_avg \
    --patch_len $patch_len \
    --activation 'gelu' \
    --des 'Exp' \
    --itr 1

echo "Completed Weather prediction length: 720"

echo "All HybridPatchAutoformer Weather experiments completed!"
echo ""
echo "Summary of experiments:"
echo "- Weather 96->96: Short-term weather forecasting"
echo "- Weather 96->192: Medium-term weather forecasting" 
echo "- Weather 96->336: Long-term weather forecasting (2 weeks)"
echo "- Weather 96->720: Extended weather forecasting (1 month)"
echo ""
echo "Model features utilized:"
echo "- ✓ Patch-based processing for weather time series efficiency"
echo "- ✓ AutoCorrelation for capturing periodic weather patterns"
echo "- ✓ Self-Attention for general weather dependencies"
echo "- ✓ Series decomposition for trend/seasonal weather analysis"
echo "- ✓ Instance normalization for handling weather non-stationarity" 