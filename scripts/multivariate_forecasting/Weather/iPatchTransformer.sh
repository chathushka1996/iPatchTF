export CUDA_VISIBLE_DEVICES=0

model_name=iPatchTransformer

# Weather dataset - 96 prediction length
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_96 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 96 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --patch_len 16 \
  --stride 8 \
  --use_norm \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --itr 1

# Weather dataset - 192 prediction length  
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_192 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 192 \
  --e_layers 2 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 256 \
  --d_ff 256 \
  --patch_len 16 \
  --stride 8 \
  --use_norm \
  --batch_size 128 \
  --learning_rate 0.0001 \
  --itr 1

# Weather dataset - 336 prediction length
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_336 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 336 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --patch_len 16 \
  --stride 8 \
  --use_norm \
  --batch_size 64 \
  --learning_rate 0.0001 \
  --itr 1

# Weather dataset - 720 prediction length
python -u run.py \
  --is_training 1 \
  --root_path ./dataset/weather/ \
  --data_path weather.csv \
  --model_id weather_96_720 \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len 96 \
  --pred_len 720 \
  --e_layers 3 \
  --enc_in 21 \
  --dec_in 21 \
  --c_out 21 \
  --des 'Exp' \
  --d_model 512 \
  --d_ff 512 \
  --patch_len 24 \
  --stride 12 \
  --use_norm \
  --batch_size 32 \
  --learning_rate 0.0001 \
  --itr 1 