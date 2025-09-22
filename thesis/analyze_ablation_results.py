import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import re

def collect_ablation_results():
    """
    Collect and analyze ablation study results from the results directory
    """
    results_dir = Path('./results')
    
    if not results_dir.exists():
        print("Results directory not found. Please run the ablation study first.")
        return None
    
    # Pattern to match ablation result files
    pattern = r'.*ablation_(.+)_pred(\d+).*\.txt'
    
    results = []
    
    # Scan for result files
    for file_path in results_dir.rglob('*.txt'):
        match = re.search(pattern, str(file_path))
        if match:
            model_variant = match.group(1)
            pred_len = int(match.group(2))
            
            try:
                # Read the result file
                with open(file_path, 'r') as f:
                    content = f.read()
                
                # Extract MSE and MAE values (assuming they're in the file)
                mse_match = re.search(r'mse:([0-9.]+)', content, re.IGNORECASE)
                mae_match = re.search(r'mae:([0-9.]+)', content, re.IGNORECASE)
                
                if mse_match and mae_match:
                    mse = float(mse_match.group(1))
                    mae = float(mae_match.group(1))
                    
                    results.append({
                        'Model': model_variant,
                        'Pred_Len': pred_len,
                        'MSE': mse,
                        'MAE': mae,
                        'File': str(file_path)
                    })
                    
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue
    
    if not results:
        print("No ablation results found. Please check the results directory.")
        return None
    
    return pd.DataFrame(results)

def analyze_component_impact(df):
    """
    Analyze the impact of each component removal
    """
    # Component mapping
    component_map = {
        'PatchXFormer': 'Full Model',
        'PatchXFormer_NoFreqAttention': 'w/o Frequency Attention',
        'PatchXFormer_NoAdaptiveNorm': 'w/o Adaptive Norm',
        'PatchXFormer_NoEnhancedEmbedding': 'w/o Enhanced Embedding',
        'PatchXFormer_NoHybridEncoder': 'w/o Hybrid Encoder',
        'PatchXFormer_BasicPatchModel': 'Basic Patch Model'
    }
    
    df['Component'] = df['Model'].map(component_map)
    
    # Calculate performance degradation compared to full model
    full_model_results = df[df['Model'] == 'PatchXFormer'].set_index('Pred_Len')
    
    impact_analysis = []
    
    for pred_len in df['Pred_Len'].unique():
        full_mse = full_model_results.loc[pred_len, 'MSE']
        full_mae = full_model_results.loc[pred_len, 'MAE']
        
        pred_len_data = df[df['Pred_Len'] == pred_len]
        
        for _, row in pred_len_data.iterrows():
            if row['Model'] != 'PatchXFormer':
                mse_degradation = ((row['MSE'] - full_mse) / full_mse) * 100
                mae_degradation = ((row['MAE'] - full_mae) / full_mae) * 100
                
                impact_analysis.append({
                    'Component_Removed': row['Component'],
                    'Pred_Len': pred_len,
                    'MSE_Degradation_%': mse_degradation,
                    'MAE_Degradation_%': mae_degradation,
                    'MSE_Absolute': row['MSE'] - full_mse,
                    'MAE_Absolute': row['MAE'] - full_mae
                })
    
    return pd.DataFrame(impact_analysis)

def create_visualizations(df, impact_df):
    """
    Create visualizations for ablation study results
    """
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. MSE Performance Comparison
    pivot_mse = df.pivot(index='Component', columns='Pred_Len', values='MSE')
    sns.heatmap(pivot_mse, annot=True, fmt='.4f', cmap='Reds', ax=axes[0,0])
    axes[0,0].set_title('MSE Performance Across Components and Forecast Horizons')
    axes[0,0].set_xlabel('Forecast Horizon (time steps)')
    axes[0,0].set_ylabel('Model Variant')
    
    # 2. MAE Performance Comparison
    pivot_mae = df.pivot(index='Component', columns='Pred_Len', values='MAE')
    sns.heatmap(pivot_mae, annot=True, fmt='.4f', cmap='Blues', ax=axes[0,1])
    axes[0,1].set_title('MAE Performance Across Components and Forecast Horizons')
    axes[0,1].set_xlabel('Forecast Horizon (time steps)')
    axes[0,1].set_ylabel('Model Variant')
    
    # 3. Component Impact on MSE
    pivot_impact_mse = impact_df.pivot(index='Component_Removed', columns='Pred_Len', values='MSE_Degradation_%')
    sns.heatmap(pivot_impact_mse, annot=True, fmt='.2f', cmap='Oranges', ax=axes[1,0])
    axes[1,0].set_title('MSE Degradation % When Components Removed')
    axes[1,0].set_xlabel('Forecast Horizon (time steps)')
    axes[1,0].set_ylabel('Component Removed')
    
    # 4. Component Impact on MAE
    pivot_impact_mae = impact_df.pivot(index='Component_Removed', columns='Pred_Len', values='MAE_Degradation_%')
    sns.heatmap(pivot_impact_mae, annot=True, fmt='.2f', cmap='Purples', ax=axes[1,1])
    axes[1,1].set_title('MAE Degradation % When Components Removed')
    axes[1,1].set_xlabel('Forecast Horizon (time steps)')
    axes[1,1].set_ylabel('Component Removed')
    
    plt.tight_layout()
    plt.savefig('ablation_study_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def generate_ablation_report(df, impact_df):
    """
    Generate a comprehensive ablation study report
    """
    report = []
    report.append("="*80)
    report.append("PATCHXFORMER ABLATION STUDY ANALYSIS REPORT")
    report.append("="*80)
    report.append("")
    
    # Overall performance summary
    report.append("1. OVERALL PERFORMANCE SUMMARY")
    report.append("-" * 40)
    report.append("")
    
    for pred_len in sorted(df['Pred_Len'].unique()):
        report.append(f"Forecast Horizon: {pred_len} time steps")
        subset = df[df['Pred_Len'] == pred_len].sort_values('MSE')
        for _, row in subset.iterrows():
            report.append(f"  {row['Component']:25} MSE: {row['MSE']:.6f} MAE: {row['MAE']:.6f}")
        report.append("")
    
    # Component impact analysis
    report.append("2. COMPONENT IMPACT ANALYSIS")
    report.append("-" * 40)
    report.append("")
    
    # Average impact across all horizons
    avg_impact = impact_df.groupby('Component_Removed').agg({
        'MSE_Degradation_%': 'mean',
        'MAE_Degradation_%': 'mean'
    }).round(2)
    
    report.append("Average Performance Degradation When Components Removed:")
    report.append("")
    for component, row in avg_impact.iterrows():
        report.append(f"  {component:30} MSE: +{row['MSE_Degradation_%']:5.2f}% MAE: +{row['MAE_Degradation_%']:5.2f}%")
    
    report.append("")
    report.append("3. KEY FINDINGS")
    report.append("-" * 40)
    report.append("")
    
    # Find most impactful components
    most_impactful_mse = avg_impact['MSE_Degradation_%'].idxmax()
    most_impactful_mae = avg_impact['MAE_Degradation_%'].idxmax()
    
    report.append(f"Most impactful component (MSE): {most_impactful_mse}")
    report.append(f"  - Removal causes {avg_impact.loc[most_impactful_mse, 'MSE_Degradation_%']:.2f}% MSE increase")
    report.append("")
    report.append(f"Most impactful component (MAE): {most_impactful_mae}")
    report.append(f"  - Removal causes {avg_impact.loc[most_impactful_mae, 'MAE_Degradation_%']:.2f}% MAE increase")
    report.append("")
    
    # Performance ranking
    report.append("4. COMPONENT IMPORTANCE RANKING (by MSE impact)")
    report.append("-" * 40)
    report.append("")
    
    ranked_components = avg_impact.sort_values('MSE_Degradation_%', ascending=False)
    for i, (component, row) in enumerate(ranked_components.iterrows(), 1):
        report.append(f"{i}. {component} (+{row['MSE_Degradation_%']:.2f}% MSE degradation)")
    
    report.append("")
    report.append("="*80)
    
    return "\n".join(report)

def main():
    """
    Main function to run ablation analysis
    """
    print("Collecting ablation study results...")
    
    # Collect results
    df = collect_ablation_results()
    
    if df is None:
        return
    
    print(f"Found {len(df)} result entries")
    print(f"Models tested: {df['Model'].unique()}")
    print(f"Forecast horizons: {sorted(df['Pred_Len'].unique())}")
    
    # Analyze component impact
    impact_df = analyze_component_impact(df)
    
    # Create visualizations
    print("Creating visualizations...")
    create_visualizations(df, impact_df)
    
    # Generate report
    print("Generating analysis report...")
    report = generate_ablation_report(df, impact_df)
    
    # Save report
    with open('ablation_analysis_report.txt', 'w') as f:
        f.write(report)
    
    # Save detailed results
    df.to_csv('ablation_detailed_results.csv', index=False)
    impact_df.to_csv('ablation_component_impact.csv', index=False)
    
    print(report)
    print("\nAnalysis complete!")
    print("Files saved:")
    print("  - ablation_study_analysis.png (visualizations)")
    print("  - ablation_analysis_report.txt (detailed report)")
    print("  - ablation_detailed_results.csv (raw results)")
    print("  - ablation_component_impact.csv (impact analysis)")

if __name__ == '__main__':
    main()
