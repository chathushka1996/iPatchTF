#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
path=./drive/MyDrive/msc-val
if [ ! -d "$path/logs" ]; then
    mkdir $path/logs -p
fi

echo "🌞 Solar Dataset Comparison: Basic vs Enhanced Temporal Features"
echo "This script compares PatchTimeXerEnhancedSimple performance on:"
echo "1. Basic Solar Dataset (sl) - 9 variables"
echo "2. Enhanced Solar Dataset (sl_t) - 11 variables with temporal features"
echo "=========================================================================="

# Configuration for comparison
seq_len=96
pred_len=96  # Focus on 24-hour prediction for comparison
model_name=PatchTimeXerEnhancedSimple
random_seed=2021

# Optimized configuration for fair comparison
d_model=256
d_ff=512
e_layers=3
n_heads=8
batch_size=16
train_epochs=10
patience=4
learning_rate=0.0001
dropout=0.1

echo "⚙️  Comparison Configuration:"
echo "   - Model: $model_name"
echo "   - Prediction Length: $pred_len (24 hours)"
echo "   - d_model: $d_model, e_layers: $e_layers"
echo "   - Batch Size: $batch_size, Epochs: $train_epochs"
echo "=========================================================================="

# Test 1: Basic Solar Dataset (sl)
echo "🔥 Test 1: Basic Solar Dataset (9 variables)"
dataset=sl
root_path_name=./dataset/$dataset
data_path_name=train.csv
model_id_name=solar_basic_comparison

echo "📊 Training on basic solar data..."
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
--pred_len $pred_len \
--e_layers $e_layers \
--factor 3 \
--enc_in 9 \
--dec_in 9 \
--c_out 9 \
--des 'Solar_Basic_Comparison' \
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

echo "✅ Basic solar dataset evaluation completed!"
echo "=========================================================================="

# Test 2: Enhanced Solar Dataset (sl_t)
echo "🔥 Test 2: Enhanced Solar Dataset (11 variables with temporal features)"
dataset=sl_t
root_path_name=./dataset/$dataset
data_path_name=train.csv
model_id_name=solar_enhanced_comparison

echo "📊 Training on enhanced solar data with temporal features..."
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
--pred_len $pred_len \
--e_layers $e_layers \
--factor 3 \
--enc_in 11 \
--dec_in 11 \
--c_out 11 \
--des 'Solar_Enhanced_Comparison' \
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

echo "✅ Enhanced solar dataset evaluation completed!"
echo "=========================================================================="

echo "🎉 Solar Dataset Comparison Completed!"
echo "📊 Results Summary:"
echo "   - Basic Solar (9 vars): Check results for solar_basic_comparison"
echo "   - Enhanced Solar (11 vars): Check results for solar_enhanced_comparison"
echo ""
echo "💡 Analysis Tips:"
echo "   - Compare MSE/MAE between basic and enhanced versions"
echo "   - Enhanced version should perform better due to temporal features"
echo "   - dayofyear captures seasonal patterns"
echo "   - timeofday captures daily solar generation cycles"
echo ""
echo "📁 Check detailed logs in: $path/logs/"
