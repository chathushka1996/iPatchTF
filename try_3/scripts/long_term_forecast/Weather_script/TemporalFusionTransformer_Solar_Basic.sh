#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TemporalFusionTransformer

echo "🌞 Starting TemporalFusionTransformer evaluation on Basic Solar dataset..."
echo "Dataset: Solar Power Basic (8 variables)"
echo "Variables: temp, dew, humidity, winddir, windspeed, pressure, cloudcover, Solar Power Output"

# Solar Basic 96 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/sl/ \
  --data_path train.csv \
  --model_id solar_basic_96_96 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --des 'TFT_Solar_Basic_96' \
  --learning_rate 0.0001 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 5 \
  --itr 1

# Solar Basic 192 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/sl/ \
  --data_path train.csv \
  --model_id solar_basic_96_192 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --des 'TFT_Solar_Basic_192' \
  --learning_rate 0.0001 \
  --batch_size 16 \
  --train_epochs 10 \
  --patience 5 \
  --itr 1

# Solar Basic 336 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/sl/ \
  --data_path train.csv \
  --model_id solar_basic_96_336 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 320 \
  --d_ff 640 \
  --n_heads 8 \
  --dropout 0.15 \
  --embed timeF \
  --freq h \
  --des 'TFT_Solar_Basic_336' \
  --learning_rate 0.00008 \
  --batch_size 12 \
  --train_epochs 12 \
  --patience 6 \
  --itr 1

# Solar Basic 720 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/sl/ \
  --data_path train.csv \
  --model_id solar_basic_96_720 \
  --model $model_name \
  --data solar \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 4 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 8 \
  --dec_in 8 \
  --c_out 8 \
  --d_model 384 \
  --d_ff 768 \
  --n_heads 8 \
  --dropout 0.2 \
  --embed timeF \
  --freq h \
  --des 'TFT_Solar_Basic_720' \
  --learning_rate 0.00005 \
  --batch_size 8 \
  --train_epochs 15 \
  --patience 8 \
  --itr 1

echo "✅ TemporalFusionTransformer Basic Solar evaluation completed!"
