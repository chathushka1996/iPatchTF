#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TemporalFusionTransformer

echo "🌞 Starting TemporalFusionTransformer evaluation on Enhanced Solar dataset..."
echo "Dataset: Solar Power Enhanced (11 variables with temporal features)"

# Solar Enhanced 96 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/sl_t/ \
  --data_path train.csv \
  --model_id solar_enhanced_96_96 \
  --model $model_name \
  --data solar_enhanced \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --des 'TFT_Solar_Enhanced_96' \
  --learning_rate 0.0001 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 5 \
  --itr 1

# Solar Enhanced 192 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/sl_t/ \
  --data_path train.csv \
  --model_id solar_enhanced_96_192 \
  --model $model_name \
  --data solar_enhanced \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --des 'TFT_Solar_Enhanced_192' \
  --learning_rate 0.0001 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 5 \
  --itr 1

# Solar Enhanced 336 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/sl_t/ \
  --data_path train.csv \
  --model_id solar_enhanced_96_336 \
  --model $model_name \
  --data solar_enhanced \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 320 \
  --d_ff 640 \
  --n_heads 8 \
  --dropout 0.15 \
  --embed timeF \
  --freq h \
  --des 'TFT_Solar_Enhanced_336' \
  --learning_rate 0.00008 \
  --batch_size 12 \
  --train_epochs 12 \
  --patience 6 \
  --itr 1

# Solar Enhanced 720 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/sl_t/ \
  --data_path train.csv \
  --model_id solar_enhanced_96_720 \
  --model $model_name \
  --data solar_enhanced \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 4 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 11 \
  --dec_in 11 \
  --c_out 11 \
  --d_model 384 \
  --d_ff 768 \
  --n_heads 8 \
  --dropout 0.2 \
  --embed timeF \
  --freq h \
  --des 'TFT_Solar_Enhanced_720' \
  --learning_rate 0.00005 \
  --batch_size 8 \
  --train_epochs 15 \
  --patience 8 \
  --itr 1

echo "✅ TemporalFusionTransformer Enhanced Solar evaluation completed!"
