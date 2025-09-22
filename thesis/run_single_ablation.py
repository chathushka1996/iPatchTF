import argparse
import os
import sys

def run_single_ablation():
    """
    Run a single ablation experiment
    """
    parser = argparse.ArgumentParser(description='Run Single PatchXFormer Ablation Experiment')
    
    # Ablation specific arguments
    parser.add_argument('--ablation_model', type=str, required=True,
                        choices=['PatchXFormer', 'PatchXFormer_NoFreqAttention', 
                                'PatchXFormer_NoAdaptiveNorm', 'PatchXFormer_NoEnhancedEmbedding',
                                'PatchXFormer_NoHybridEncoder', 'PatchXFormer_BasicPatchModel'],
                        help='Which ablation model variant to run')
    
    parser.add_argument('--pred_len', type=int, required=True,
                        choices=[96, 192, 336, 720],
                        help='Prediction length for the experiment')
    
    # Basic experiment parameters
    parser.add_argument('--data_name', type=str, default='sl_t', help='Dataset name')
    parser.add_argument('--seq_len', type=int, default=96, help='Input sequence length')
    parser.add_argument('--train_epochs', type=int, default=10, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device')
    
    args = parser.parse_args()
    
    # Model descriptions
    descriptions = {
        'PatchXFormer': 'Full PatchXFormer (All Components)',
        'PatchXFormer_NoFreqAttention': 'Without Frequency-Enhanced Attention',
        'PatchXFormer_NoAdaptiveNorm': 'Without Adaptive Normalization',
        'PatchXFormer_NoEnhancedEmbedding': 'Without Enhanced Patch Embedding',
        'PatchXFormer_NoHybridEncoder': 'Without Hybrid Encoder (No Cross-Attention)',
        'PatchXFormer_BasicPatchModel': 'Basic Patch Model (No Enhancements)'
    }
    
    print("="*80)
    print("SINGLE ABLATION EXPERIMENT")
    print("="*80)
    print(f"Model: {descriptions[args.ablation_model]}")
    print(f"Forecast Horizon: {args.pred_len} time steps")
    print(f"Dataset: {args.data_name}")
    print("="*80)
    
    # Build the command
    cmd = [
        'python', '-u', 'run.py',
        '--task_name', 'long_term_forecast',
        '--is_training', '1',
        '--root_path', f'./dataset/{args.data_name}/',
        '--data_path', 'train.csv',
        '--model_id', f'{args.data_name}_{args.pred_len}',
        '--model', args.ablation_model,
        '--data', args.data_name,
        '--features', 'M',
        '--seq_len', str(args.seq_len),
        '--label_len', '48',
        '--pred_len', str(args.pred_len),
        '--e_layers', '3',
        '--d_layers', '1',
        '--factor', '1',
        '--enc_in', '9',
        '--dec_in', '9',
        '--c_out', '9',
        '--d_model', '256',
        '--n_heads', '8',
        '--d_ff', '1024',
        '--dropout', '0.1',
        '--fc_dropout', '0.1',
        '--head_dropout', '0.1',
        '--patch_len', '16',
        '--stride', '8',
        '--des', f'ablation_{args.ablation_model}_pred{args.pred_len}',
        '--train_epochs', str(args.train_epochs),
        '--patience', '3',
        '--learning_rate', str(args.learning_rate),
        '--batch_size', str(args.batch_size),
        '--use_amp',
        '--gpu', str(args.gpu)
    ]
    
    # Run the command
    print("Running experiment...")
    print("Command:", ' '.join(cmd))
    print("-" * 80)
    
    result = os.system(' '.join(cmd))
    
    if result == 0:
        print("✓ Experiment completed successfully!")
    else:
        print("✗ Experiment failed!")
        sys.exit(1)

if __name__ == '__main__':
    run_single_ablation()
