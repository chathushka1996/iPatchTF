#!/bin/bash

# Test script to verify bash syntax is correct

echo "Testing bash syntax for ablation scripts..."

scripts=(
    "run_patchxformer_full.sh"
    "run_patchxformer_no_freq_attention.sh"
    "run_patchxformer_no_adaptive_norm.sh"
    "run_patchxformer_no_enhanced_embedding.sh"
    "run_patchxformer_no_hybrid_encoder.sh"
    "run_patchxformer_basic_patch.sh"
    "run_ablation_components.sh"
    "run_quick_ablation_test.sh"
    "run_single_horizon_ablation.sh"
    "run_component_comparison.sh"
    "run_all_ablation_scripts.sh"
)

for script in "${scripts[@]}"; do
    if [ -f "$script" ]; then
        echo "Checking syntax: $script"
        bash -n "$script"
        if [ $? -eq 0 ]; then
            echo "✓ $script - Syntax OK"
        else
            echo "✗ $script - Syntax Error"
        fi
    else
        echo "⚠ $script - File not found"
    fi
done

echo ""
echo "Syntax check completed!"
