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
model_id_name=solar_quick_test
data_name=custom
random_seed=2021

echo "🌞 Quick Test: PatchTimeXerEnhancedSimple on Solar dataset"
echo "This script runs a fast evaluation to verify the model works with solar data."
echo "Dataset: $dataset (9 variables)"
echo "Prediction Length: 96 (24 hours)"

# Quick test configuration - small model, few epochs
d_model=128
d_ff=256
e_layers=2
n_heads=4
batch_size=32
train_epochs=3
patience=2
learning_rate=0.001
dropout=0.1

echo "⚙️  Quick Test Configuration: d_model=$d_model, e_layers=$e_layers, batch_size=$batch_size"
echo "🚀 Starting quick solar forecasting test..."

python -u run.py \
--task_name long_term_forecast \
--is_training 1 \
--root_path $root_path_name \
--data_path $data_path_name \
--model_id $model_id_name \
--model $model_name \
--data custom \
--features M \
--seq_len $seq_len \
--label_len 48 \
--pred_len 96 \
--e_layers $e_layers \
--factor 3 \
--enc_in 9 \
--dec_in 9 \
--c_out 9 \
--des 'Solar_Quick_Test' \
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

echo "✅ PatchTimeXerEnhancedSimple solar quick test completed!"
echo "🌞 If this runs successfully, you can use the full solar evaluation scripts."
echo "📊 Next steps:"
echo "   - Run PatchTimeXerEnhancedSimple_Solar.sh for full evaluation"
echo "   - Run PatchTimeXerEnhancedSimple_Solar_Enhanced.sh for temporal features version"
