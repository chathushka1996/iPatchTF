#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TemporalFusionTransformer

echo "🚀 Quick Test: TemporalFusionTransformer on ETTh1 dataset"
echo "This script runs a fast evaluation to verify TFT is working correctly."

# Quick test with reduced epochs for fast validation
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_quick_test \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 128 \
  --d_ff 256 \
  --n_heads 4 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --des 'TFT_Quick_Test' \
  --learning_rate 0.001 \
  --batch_size 64 \
  --train_epochs 2 \
  --patience 2 \
  --itr 1

echo "✅ TemporalFusionTransformer quick test completed!"
echo "If this runs successfully, you can use the full evaluation scripts."
