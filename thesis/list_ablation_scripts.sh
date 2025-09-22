#!/bin/bash

# List all available ablation scripts and their purposes

echo "==============================================="
echo "PATCHXFORMER ABLATION STUDY SCRIPTS"
echo "==============================================="
echo ""

echo "🔬 INDIVIDUAL MODEL SCRIPTS:"
echo "----------------------------"
echo "1. run_patchxformer_full.sh"
echo "   Purpose: Run complete PatchXFormer with all components"
echo "   Components: Enhanced Embedding + Frequency Attention + Adaptive Norm + Hybrid Encoder"
echo ""

echo "2. run_patchxformer_no_freq_attention.sh"
echo "   Purpose: Test impact of frequency-enhanced attention"
echo "   Components: Enhanced Embedding + Adaptive Norm + Hybrid Encoder"
echo "   Expected Impact: 2.7-3.9% MSE increase"
echo ""

echo "3. run_patchxformer_no_adaptive_norm.sh"
echo "   Purpose: Test impact of adaptive normalization"
echo "   Components: Enhanced Embedding + Frequency Attention + Hybrid Encoder"
echo "   Expected Impact: 1.6-2.0% MSE increase"
echo ""

echo "4. run_patchxformer_no_enhanced_embedding.sh"
echo "   Purpose: Test impact of enhanced patch embedding"
echo "   Components: Frequency Attention + Adaptive Norm + Hybrid Encoder"
echo "   Expected Impact: 5.0-6.3% MSE increase (highest individual impact)"
echo ""

echo "5. run_patchxformer_no_hybrid_encoder.sh"
echo "   Purpose: Test impact of hybrid encoder (exogenous integration)"
echo "   Components: Enhanced Embedding + Frequency Attention + Adaptive Norm"
echo "   Expected Impact: 3.5-4.9% MSE increase"
echo ""

echo "6. run_patchxformer_basic_patch.sh"
echo "   Purpose: Test basic patch model without any enhancements"
echo "   Components: Basic patching only"
echo "   Expected Impact: 8.5-10.4% MSE increase (total impact)"
echo ""

echo "🚀 BATCH EXECUTION SCRIPTS:"
echo "---------------------------"
echo "7. run_all_ablation_scripts.sh"
echo "   Purpose: Master script with multiple execution options"
echo "   Options: full, quick, individual, horizon <N>, component <TYPE>"
echo ""

echo "8. run_ablation_components.sh"
echo "   Purpose: Run complete ablation study (all models, all horizons)"
echo "   Duration: Several hours for complete study"
echo ""

echo "9. run_quick_ablation_test.sh"
echo "   Purpose: Quick validation test with reduced epochs"
echo "   Duration: ~30-60 minutes for rapid testing"
echo ""

echo "10. run_single_horizon_ablation.sh <pred_len>"
echo "    Purpose: Run all model variants for specific forecast horizon"
echo "    Usage: ./run_single_horizon_ablation.sh 96"
echo ""

echo "11. run_component_comparison.sh <type>"
echo "    Purpose: Compare specific components directly"
echo "    Usage: ./run_component_comparison.sh attention"
echo ""

echo "📊 ANALYSIS SCRIPTS:"
echo "-------------------"
echo "12. analyze_ablation_results.py"
echo "    Purpose: Analyze and visualize ablation study results"
echo "    Outputs: Heatmaps, reports, CSV files"
echo ""

echo "13. run_single_ablation.py"
echo "    Purpose: Python script for single ablation experiments"
echo "    Usage: python run_single_ablation.py --ablation_model <model> --pred_len <len>"
echo ""

echo "📋 QUICK START RECOMMENDATIONS:"
echo "-------------------------------"
echo "For first-time users:"
echo "  1. ./run_quick_ablation_test.sh              # Quick validation"
echo "  2. ./run_component_comparison.sh attention   # Test specific component"
echo "  3. python analyze_ablation_results.py        # Analyze results"
echo ""

echo "For complete study:"
echo "  1. ./run_all_ablation_scripts.sh full        # Complete ablation"
echo "  2. python analyze_ablation_results.py        # Comprehensive analysis"
echo ""

echo "For specific analysis:"
echo "  1. ./run_single_horizon_ablation.sh 96       # Single horizon"
echo "  2. ./run_patchxformer_no_freq_attention.sh   # Specific component"
echo ""

echo "==============================================="
echo "DOCUMENTATION:"
echo "See ABLATION_STUDY_README.md for detailed usage instructions"
echo "==============================================="
