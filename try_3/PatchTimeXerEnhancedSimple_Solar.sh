#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
path=./drive/MyDrive/msc-val
if [ ! -d "$path/logs" ]; then
    mkdir $path/logs -p
fi
model_name=PatchTimeXerEnhancedSimple
seq_len=96
dataset=sl
root_path_name=./dataset/$dataset
data_path_name=train.csv
model_id_name=solar_$dataset
data_name=custom
random_seed=2021
checkpoints=$path/model_log

echo "🌞 Starting PatchTimeXerEnhancedSimple evaluation on Solar dataset..."
echo "Model: $model_name"
echo "Dataset: $dataset (Solar Power Output)"
echo "Variables: 9 (temp, dew, humidity, winddir, windspeed, pressure, cloudcover, Solar Power Output)"
echo "Sequence Length: $seq_len"
echo "Data Path: $root_path_name/$data_path_name"

for pred_len in 96 192 336 720
do
  echo "🚀 Training for prediction length: $pred_len"
  
  # Solar-optimized configuration for stability and performance
  if [ $pred_len -eq 96 ]; then
    d_model=256
    d_ff=512
    e_layers=2
    n_heads=8
    batch_size=16
    train_epochs=15
    patience=5
    learning_rate=0.0001
    dropout=0.1
    echo "📊 Short-term solar forecasting (96 steps = 24 hours)"
  elif [ $pred_len -eq 192 ]; then
    d_model=384
    d_ff=768
    e_layers=3
    n_heads=8
    batch_size=12
    train_epochs=18
    patience=6
    learning_rate=0.00008
    dropout=0.12
    echo "📊 Medium-term solar forecasting (192 steps = 48 hours)"
  elif [ $pred_len -eq 336 ]; then
    d_model=512
    d_ff=1024
    e_layers=3
    n_heads=8
    batch_size=8
    train_epochs=20
    patience=7
    learning_rate=0.00006
    dropout=0.15
    echo "📊 Long-term solar forecasting (336 steps = 84 hours)"
  elif [ $pred_len -eq 720 ]; then
    d_model=512
    d_ff=1024
    e_layers=4
    n_heads=8
    batch_size=6
    train_epochs=25
    patience=10
    learning_rate=0.00004
    dropout=0.18
    echo "📊 Extended solar forecasting (720 steps = 180 hours)"
  fi
  
  echo "⚙️  Configuration: d_model=$d_model, e_layers=$e_layers, batch_size=$batch_size, lr=$learning_rate"
  
  python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --model_id $model_id_name$seq_len'_'$pred_len \
  --model $model_name \
  --data custom \
  --features M \
  --seq_len $seq_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers $e_layers \
  --factor 3 \
  --enc_in 9 \
  --dec_in 9 \
  --c_out 9 \
  --des 'Solar_Enhanced' \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout $dropout \
  --activation gelu \
  --embed timeF \
  --freq h \
  --itr 1
  #--checkpoints $checkpoints > $path/logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 2>&1
  
  echo "✅ Completed training for prediction length: $pred_len"
  echo "📝 Log saved to: $path/logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log"
  echo "----------------------------------------"
done

echo "🎉 All PatchTimeXerEnhancedSimple Solar evaluations completed!"
echo "📁 Check logs in: $path/logs/"
echo "🌞 Solar power forecasting results ready for analysis!"
