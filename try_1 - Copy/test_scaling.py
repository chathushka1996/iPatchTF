"""
Simple script to test and diagnose scaling issues
Run this to understand what's happening with your data
"""

def test_scaling_hypothesis():
    """Test the scaling hypothesis with minimal dependencies"""
    import os
    
    # Check if data files exist
    data_path = "data/sl_t/"
    train_file = os.path.join(data_path, "train.csv")
    
    if not os.path.exists(train_file):
        print(f"❌ Training file not found: {train_file}")
        return
    
    print("🔍 SCALING DIAGNOSIS")
    print("=" * 50)
    
    try:
        # Try to read with basic Python (no pandas)
        with open(train_file, 'r') as f:
            header = f.readline().strip()
            print(f"Header: {header}")
            
            # Read more lines to get better sample
            sample_lines = []
            all_lines = []
            for i, line in enumerate(f):
                all_lines.append(line.strip())
                if i < 100:  # Read first 100 lines
                    sample_lines.append(line.strip())
                if i > 10000:  # Don't read the entire file
                    break
        
        print(f"\nTotal lines read: {len(all_lines)}")
        
        # Try to extract target column values
        header_cols = header.split(',')
        target_col = 'Solar Power Output'
        
        if target_col in header_cols:
            target_idx = header_cols.index(target_col)
            print(f"\n📊 Target column '{target_col}' found at index {target_idx}")
            
            # Analyze all available data
            target_values = []
            zero_count = 0
            positive_count = 0
            
            for line in all_lines:
                try:
                    values = line.split(',')
                    if len(values) > target_idx:
                        target_val = float(values[target_idx])
                        target_values.append(target_val)
                        if target_val == 0.0:
                            zero_count += 1
                        elif target_val > 0:
                            positive_count += 1
                except:
                    pass
            
            if target_values:
                print(f"\n📈 TARGET VALUE ANALYSIS:")
                print(f"  Total samples: {len(target_values)}")
                print(f"  Zero values: {zero_count} ({zero_count/len(target_values)*100:.1f}%)")
                print(f"  Positive values: {positive_count} ({positive_count/len(target_values)*100:.1f}%)")
                print(f"  Range: [{min(target_values):.6f}, {max(target_values):.6f}]")
                print(f"  Average: {sum(target_values)/len(target_values):.6f}")
                
                # Find non-zero values
                non_zero_values = [v for v in target_values if v > 0]
                if non_zero_values:
                    print(f"  Non-zero range: [{min(non_zero_values):.6f}, {max(non_zero_values):.6f}]")
                    print(f"  Non-zero average: {sum(non_zero_values)/len(non_zero_values):.6f}")
                
                # Show some sample non-zero values
                sample_nonzero = [v for v in target_values[:1000] if v > 0]
                if sample_nonzero:
                    print(f"  Sample non-zero values: {sample_nonzero[:10]}")
                
                # Analyze the scale
                max_val = max(target_values)
                print(f"\n🎯 SCALE ANALYSIS:")
                if max_val < 1:
                    print("✅ Max value < 1 - good for neural networks")
                elif max_val < 100:
                    print("⚠️  Max value < 100 - scaling may help")
                elif max_val < 10000:
                    print("⚠️  Max value < 10,000 - scaling recommended")
                else:
                    print("❌ Max value >= 10,000 - scaling critical!")
                
                # Check distribution
                if zero_count / len(target_values) > 0.5:
                    print("⚠️  WARNING: >50% of values are zero!")
                    print("   This creates challenges for:")
                    print("   - Scaling algorithms (RobustScaler)")
                    print("   - Neural network training")
                    print("   - Loss calculation")
                    
            else:
                print("❌ Could not extract target values")
        else:
            print(f"❌ Target column '{target_col}' not found in header")
            print(f"Available columns: {header_cols}")
    
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    print("\n🎯 LIKELY ISSUES:")
    print("-" * 30)
    if zero_count / len(target_values) > 0.5:
        print("1. HIGH ZERO RATIO PROBLEM:")
        print("   - Many zero values (night time) create scaling issues")
        print("   - RobustScaler may not handle this distribution well")
        print("   - Model might predict non-zero for zero targets")
        print("   - This creates large MSE even for small prediction errors")
        print()
    print("2. SCALING AMPLIFICATION:")
    print("   - Small errors in scaled space become large in original space")
    print("   - Inverse transform amplifies prediction errors")
    print("3. LOSS CALCULATION MISMATCH:")
    print("   - Training uses scaled space, metrics use original space")
    
    print("\n💡 SOLUTIONS:")
    print("-" * 30)
    print("1. Use StandardScaler instead of RobustScaler")
    print("2. Keep training metrics in scaled space")
    print("3. Add epsilon to zero values before scaling")
    print("4. Use log1p transform for skewed data")
    print("5. Separate day/night training or balanced sampling")

if __name__ == "__main__":
    test_scaling_hypothesis() 