export CUDA_VISIBLE_DEVICES=0
path=./drive/MyDrive/msc-val
if [ ! -d "$path/logs" ]; then
    mkdir $path/logs -p
fi
model_name=PatchTimeXerEnhanced
seq_len=96
dataset=sl_t
root_path_name=./dataset/$dataset
data_path_name=solar.csv
model_id_name=solar_$dataset
data_name=custom
random_seed=2021
checkpoints=$path/model_log

echo "Starting PatchTimeXerEnhanced evaluation on solar dataset..."
echo "Model: $model_name"
echo "Dataset: $dataset"
echo "Sequence Length: $seq_len"

for pred_len in 96 192 336 720
do
  echo "Training for prediction length: $pred_len"
  
  # Adaptive model configuration based on prediction length
  if [ $pred_len -eq 96 ]; then
    d_model=512
    d_ff=1024
    e_layers=3
    n_heads=8
    batch_size=16
    train_epochs=15
    patience=5
    learning_rate=0.0001
    dropout=0.1
  elif [ $pred_len -eq 192 ]; then
    d_model=512
    d_ff=1024
    e_layers=3
    n_heads=8
    batch_size=12
    train_epochs=18
    patience=6
    learning_rate=0.0001
    dropout=0.1
  elif [ $pred_len -eq 336 ]; then
    d_model=768
    d_ff=1536
    e_layers=4
    n_heads=12
    batch_size=8
    train_epochs=22
    patience=8
    learning_rate=0.00008
    dropout=0.12
  elif [ $pred_len -eq 720 ]; then
    d_model=768
    d_ff=1536
    e_layers=4
    n_heads=12
    batch_size=4
    train_epochs=25
    patience=10
    learning_rate=0.00005
    dropout=0.15
  fi
  
  echo "Configuration: d_model=$d_model, e_layers=$e_layers, batch_size=$batch_size"
  
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
  --enc_in 10 \
  --dec_in 10 \
  --c_out 10 \
  --des 'Exp' \
  --d_model $d_model \
  --d_ff $d_ff \
  --n_heads $n_heads \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --patience $patience \
  --dropout $dropout \
  --random_seed $random_seed \
  --use_gpu True \
  --gpu 0 \
  --use_multi_gpu False \
  --itr 1 \
  --checkpoints $checkpoints > $path/logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log 2>&1
  
  echo "Completed training for prediction length: $pred_len"
  echo "Log saved to: $path/logs/$model_name'_'$model_id_name'_'$seq_len'_'$pred_len.log"
  echo "----------------------------------------"
done

echo "All PatchTimeXerEnhanced evaluations completed!"
echo "Check logs in: $path/logs/" 