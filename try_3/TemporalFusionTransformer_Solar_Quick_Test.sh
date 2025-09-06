#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TemporalFusionTransformer

echo "🌞 Quick Test: TemporalFusionTransformer on Solar dataset"
echo "This script runs a fast evaluation to verify TFT works with solar data."

# Quick test with reduced epochs for fast validation
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/sl/ \
  --data_path train.csv \
  --model_id solar_quick_test \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --des 'TFT_Solar_Quick_Test' \
  --learning_rate 0.001 \
  --batch_size 64 \
  --train_epochs 2 \
  --patience 2 \
  --itr 1

echo "✅ TemporalFusionTransformer solar quick test completed!"
echo "🌞 If this runs successfully, you can use the full solar TFT evaluation scripts."
