import argparse
import os
import torch
import numpy as np
import pandas as pd
from exp.exp_long_term_forecasting import Exp_Long_Term_Forecast
import warnings
warnings.filterwarnings('ignore')

def run_ablation_study():
    """
    Comprehensive ablation study for PatchXFormer components
    Tests the impact of each architectural component by systematically removing them
    """
    
    # Base configuration
    parser = argparse.ArgumentParser(description='PatchXFormer Ablation Study')
    
    # Basic settings
    parser.add_argument('--task_name', type=str, default='long_term_forecast',
                        help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='sl_t', help='model id')
    parser.add_argument('--model', type=str, default='PatchXFormer', help='model name')
    
    # Data loader
    parser.add_argument('--data', type=str, default='sl_t', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset/sl_t/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='train.csv', help='data file')
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='solar_generation', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='15min',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    
    # Forecasting task
    parser.add_argument('--seq_len', type=int, default=96, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=48, help='start token length')
    parser.add_argument('--pred_len', type=int, default=96, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    
    # Model parameters
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=9, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=9, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=9, help='output size')
    parser.add_argument('--d_model', type=int, default=256, help='dimension of model')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=3, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=1024, help='dimension of fcn')
    parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')
    parser.add_argument('--distil', default=True, help='whether to use distilling in encoder, using this argument means not using distilling')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
    
    # Optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    
    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    
    # Ablation study configurations
    ablation_configs = [
        {
            'model': 'PatchXFormer',
            'description': 'Full PatchXFormer (All Components)',
            'components': 'Enhanced Embedding + Frequency Attention + Adaptive Norm + Hybrid Encoder'
        },
        {
            'model': 'PatchXFormer_NoFreqAttention',
            'description': 'PatchXFormer without Frequency-Enhanced Attention',
            'components': 'Enhanced Embedding + Adaptive Norm + Hybrid Encoder'
        },
        {
            'model': 'PatchXFormer_NoAdaptiveNorm',
            'description': 'PatchXFormer without Adaptive Normalization',
            'components': 'Enhanced Embedding + Frequency Attention + Hybrid Encoder'
        },
        {
            'model': 'PatchXFormer_NoEnhancedEmbedding',
            'description': 'PatchXFormer without Enhanced Patch Embedding',
            'components': 'Frequency Attention + Adaptive Norm + Hybrid Encoder'
        },
        {
            'model': 'PatchXFormer_NoHybridEncoder',
            'description': 'PatchXFormer without Hybrid Encoder (no cross-attention)',
            'components': 'Enhanced Embedding + Frequency Attention + Adaptive Norm'
        },
        {
            'model': 'PatchXFormer_BasicPatchModel',
            'description': 'Basic Patch Model (No Enhancements)',
            'components': 'Basic patching only'
        }
    ]
    
    # Forecast horizons to test
    pred_lens = [96, 192, 336, 720]
    
    # Results storage
    results = []
    
    print("="*80)
    print("PATCHXFORMER ABLATION STUDY")
    print("="*80)
    print(f"Testing {len(ablation_configs)} model variants across {len(pred_lens)} forecast horizons")
    print("="*80)
    
    for config in ablation_configs:
        print(f"\n{'='*60}")
        print(f"Testing: {config['description']}")
        print(f"Components: {config['components']}")
        print(f"{'='*60}")
        
        for pred_len in pred_lens:
            print(f"\nForecast Horizon: {pred_len} time steps")
            print("-" * 40)
            
            # Update arguments for current configuration
            args.model = config['model']
            args.pred_len = pred_len
            args.des = f"ablation_{config['model']}_pred{pred_len}"
            
            # Run experiment
            try:
                # Set up experiment
                Exp = Exp_Long_Term_Forecast
                
                for ii in range(args.itr):
                    # Setting record of experiments
                    setting = f"{args.model_id}_{args.model}_{args.data}_{args.features}_sl{args.seq_len}_ll{args.label_len}_pl{args.pred_len}_dm{args.d_model}_nh{args.n_heads}_el{args.e_layers}_dl{args.d_layers}_df{args.d_ff}_fc{args.factor}_eb{args.embed}_dt{args.des}_{ii}"
                    
                    exp = Exp(args)  # Set experiments
                    
                    print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>>>>>>>>>>>>>')
                    exp.train(setting)
                    
                    print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
                    mse, mae = exp.test(setting)
                    
                    # Store results
                    results.append({
                        'Model': config['model'],
                        'Description': config['description'],
                        'Components': config['components'],
                        'Pred_Len': pred_len,
                        'MSE': mse,
                        'MAE': mae,
                        'Iteration': ii
                    })
                    
                    print(f"Results - MSE: {mse:.6f}, MAE: {mae:.6f}")
                    
                    torch.cuda.empty_cache()
                    
            except Exception as e:
                print(f"Error running {config['model']} with pred_len {pred_len}: {str(e)}")
                continue
    
    # Save results to CSV
    results_df = pd.DataFrame(results)
    results_df.to_csv('ablation_study_results.csv', index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("ABLATION STUDY SUMMARY")
    print("="*80)
    
    # Group by model and pred_len, calculate mean performance
    summary = results_df.groupby(['Model', 'Pred_Len']).agg({
        'MSE': 'mean',
        'MAE': 'mean'
    }).reset_index()
    
    for pred_len in pred_lens:
        print(f"\nForecast Horizon: {pred_len} time steps")
        print("-" * 50)
        subset = summary[summary['Pred_Len'] == pred_len].sort_values('MSE')
        for _, row in subset.iterrows():
            print(f"{row['Model']:30} MSE: {row['MSE']:.6f} MAE: {row['MAE']:.6f}")
    
    print(f"\nDetailed results saved to: ablation_study_results.csv")
    print("="*80)

if __name__ == '__main__':
    run_ablation_study()
