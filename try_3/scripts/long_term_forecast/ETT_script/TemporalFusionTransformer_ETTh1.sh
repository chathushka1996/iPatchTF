#!/bin/bash

export CUDA_VISIBLE_DEVICES=0

model_name=TemporalFusionTransformer

echo "Starting TemporalFusionTransformer evaluation on ETTh1 dataset..."

# ETTh1 96 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_96 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 96 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --des 'TFT_ETTh1_96' \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 5 \
  --itr 1

# ETTh1 192 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_192 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 192 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 256 \
  --d_ff 512 \
  --n_heads 8 \
  --dropout 0.1 \
  --embed timeF \
  --freq h \
  --des 'TFT_ETTh1_192' \
  --learning_rate 0.0001 \
  --batch_size 32 \
  --train_epochs 10 \
  --patience 5 \
  --itr 1

# ETTh1 336 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_336 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 336 \
  --e_layers 3 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 320 \
  --d_ff 640 \
  --n_heads 8 \
  --dropout 0.15 \
  --embed timeF \
  --freq h \
  --des 'TFT_ETTh1_336' \
  --learning_rate 0.00008 \
  --batch_size 24 \
  --train_epochs 12 \
  --patience 6 \
  --itr 1

# ETTh1 720 prediction
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_96_720 \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len 96 \
  --label_len 48 \
  --pred_len 720 \
  --e_layers 4 \
  --d_layers 2 \
  --factor 3 \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --d_model 384 \
  --d_ff 768 \
  --n_heads 8 \
  --dropout 0.2 \
  --embed timeF \
  --freq h \
  --des 'TFT_ETTh1_720' \
  --learning_rate 0.00005 \
  --batch_size 16 \
  --train_epochs 15 \
  --patience 8 \
  --itr 1

echo "TemporalFusionTransformer ETTh1 evaluation completed!"
