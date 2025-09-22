#!/bin/bash

# Master Ablation Study Script
# This script provides options to run different types of ablation studies

echo "==============================================="
echo "PATCHXFORMER ABLATION STUDY MASTER SCRIPT"
echo "==============================================="

show_help() {
    echo "Usage: $0 <option>"
    echo ""
    echo "Available options:"
    echo "  full              - Run complete ablation study (all models, all horizons)"
    echo "  quick             - Run quick test (reduced epochs, single horizon)"
    echo "  individual        - Run individual model scripts sequentially"
    echo "  horizon <N>       - Run all models for specific horizon (96/192/336/720)"
    echo "  component <TYPE>  - Run component comparison (attention/embedding/normalization/encoder/all_vs_basic)"
    echo ""
    echo "Examples:"
    echo "  $0 full                    # Complete ablation study"
    echo "  $0 quick                   # Quick validation test"
    echo "  $0 individual              # Run all individual model scripts"
    echo "  $0 horizon 96              # All models for 96 time step horizon"
    echo "  $0 component attention     # Compare with/without frequency attention"
    echo ""
    echo "Individual model scripts available:"
    echo "  ./run_patchxformer_full.sh"
    echo "  ./run_patchxformer_no_freq_attention.sh"
    echo "  ./run_patchxformer_no_adaptive_norm.sh"
    echo "  ./run_patchxformer_no_enhanced_embedding.sh"
    echo "  ./run_patchxformer_no_hybrid_encoder.sh"
    echo "  ./run_patchxformer_basic_patch.sh"
}

if [ $# -eq 0 ]; then
    show_help
    exit 1
fi

option=$1

case $option in
    "full")
        echo "Running COMPLETE ablation study..."
        echo "This will take several hours depending on your hardware."
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            bash run_ablation_components.sh
        else
            echo "Aborted."
        fi
        ;;
        
    "quick")
        echo "Running QUICK ablation test..."
        bash run_quick_ablation_test.sh
        ;;
        
    "individual")
        echo "Running INDIVIDUAL model scripts sequentially..."
        echo "This runs each model variant separately across all horizons."
        read -p "Continue? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            echo "Running Full PatchXFormer..."
            bash run_patchxformer_full.sh
            
            echo "Running PatchXFormer without Frequency Attention..."
            bash run_patchxformer_no_freq_attention.sh
            
            echo "Running PatchXFormer without Adaptive Normalization..."
            bash run_patchxformer_no_adaptive_norm.sh
            
            echo "Running PatchXFormer without Enhanced Embedding..."
            bash run_patchxformer_no_enhanced_embedding.sh
            
            echo "Running PatchXFormer without Hybrid Encoder..."
            bash run_patchxformer_no_hybrid_encoder.sh
            
            echo "Running Basic Patch Model..."
            bash run_patchxformer_basic_patch.sh
            
            echo "All individual scripts completed!"
        else
            echo "Aborted."
        fi
        ;;
        
    "horizon")
        if [ $# -ne 2 ]; then
            echo "Error: horizon option requires a forecast horizon value"
            echo "Usage: $0 horizon <96|192|336|720>"
            exit 1
        fi
        horizon=$2
        echo "Running ablation study for horizon: $horizon time steps..."
        bash run_single_horizon_ablation.sh $horizon
        ;;
        
    "component")
        if [ $# -ne 2 ]; then
            echo "Error: component option requires a comparison type"
            echo "Usage: $0 component <attention|embedding|normalization|encoder|all_vs_basic>"
            exit 1
        fi
        comp_type=$2
        echo "Running component comparison: $comp_type..."
        bash run_component_comparison.sh $comp_type
        ;;
        
    "help"|"--help"|"-h")
        show_help
        ;;
        
    *)
        echo "Invalid option: $option"
        echo ""
        show_help
        exit 1
        ;;
esac

echo ""
echo "==============================================="
echo "ABLATION STUDY SCRIPT COMPLETED"
echo "==============================================="
echo "To analyze results, run: python analyze_ablation_results.py"
