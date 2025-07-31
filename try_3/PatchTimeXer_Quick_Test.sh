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
pred_len=96  # Quick test with shortest horizon
random_seed=2021
checkpoints=$path/model_log

echo "🚀 Quick Test: PatchTimeXerEnhanced"
echo "Dataset: Solar ($dataset)"
echo "Prediction Length: $pred_len"
echo "Purpose: Fast validation and debugging"
echo "=================================="

# Quick test configuration (reduced for speed)
d_model=256
d_ff=512
e_layers=2
n_heads=8
batch_size=32
train_epochs=3  # Reduced for quick testing
patience=2
learning_rate=0.001  # Higher LR for faster convergence
dropout=0.1

echo "Quick Config: d_model=$d_model, e_layers=$e_layers, epochs=$train_epochs"
echo "Starting quick test..."

python -u run.py \
--task_name long_term_forecast \
--is_training 1 \
--root_path $root_path_name \
--data_path $data_path_name \
--model_id QUICKTEST_$model_id_name$seq_len'_'$pred_len \
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
--checkpoints $checkpoints

echo ""
echo "✅ Quick test completed!"
echo "If this runs successfully, you can proceed with full evaluation using:"
echo "bash PatchTimeXerEnhanced.sh"
echo ""
echo "💡 Tips:"
echo "- Check for any import errors or compatibility issues"
echo "- Verify model trains without CUDA memory errors"
echo "- Confirm data loading works correctly" 