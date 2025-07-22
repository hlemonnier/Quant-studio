"""
Debug script to identify why spread is still 0.0bps
"""

import sys
import pathlib
import math

# Enable imports
repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from strategies.MM.config import mm_config
from strategies.MM.avellaneda_stoikov import AvellanedaStoikovQuoter

def debug_spread_calculation():
    """Debug spread calculation step by step"""
    
    print("🔍 Debugging Spread Calculation")
    print("=" * 50)
    
    # Check config values
    print(f"📋 Configuration:")
    print(f"  gamma (γ): {mm_config.gamma}")
    print(f"  k: {mm_config.k}")
    print(f"  T: {mm_config.T}")
    print(f"  min_spread_bps: {mm_config.min_spread_bps}")
    print(f"  max_spread_bps: {mm_config.max_spread_bps}")
    print()
    
    # Step by step calculation
    gamma = mm_config.gamma
    k = mm_config.k
    
    print(f"🧮 Step-by-step calculation:")
    print(f"  1. gamma + k = {gamma} + {k} = {gamma + k}")
    print(f"  2. gamma / k = {gamma} / {k} = {gamma / k}")
    print(f"  3. 1 + gamma/k = 1 + {gamma / k} = {1 + gamma / k}")
    
    # Calculate delta_star
    try:
        arg = 1 + gamma / k
        log_arg = math.log(arg)
        delta_star = (1 / gamma) * log_arg
        optimal_spread = 2 * delta_star
        
        print(f"  4. ln(1 + γ/k) = ln({arg:.4f}) = {log_arg:.6f}")
        print(f"  5. δ* = (1/γ) × ln(1+γ/k) = (1/{gamma}) × {log_arg:.6f} = {delta_star:.6f}")
        print(f"  6. Full spread = 2 × δ* = 2 × {delta_star:.6f} = {optimal_spread:.6f}")
        
        # Convert to bps
        mid_price = 50000
        spread_bps = (optimal_spread / mid_price) * 10000
        print(f"  7. Spread in bps = ({optimal_spread:.6f} / {mid_price}) × 10000 = {spread_bps:.2f} bps")
        
        # Check bounds
        min_spread = mm_config.min_spread_bps / 10000
        max_spread = mm_config.max_spread_bps / 10000
        
        print(f"\n🚦 Bounds checking:")
        print(f"  min_spread (fraction): {min_spread:.6f}")
        print(f"  max_spread (fraction): {max_spread:.6f}")
        print(f"  optimal_spread: {optimal_spread:.6f}")
        
        clamped_spread = max(min_spread, min(max_spread, optimal_spread))
        clamped_bps = (clamped_spread / mid_price) * 10000
        
        print(f"  clamped_spread: {clamped_spread:.6f}")
        print(f"  clamped_bps: {clamped_bps:.2f} bps")  # Added "bps" for clarity
        
        # Final result
        print(f"\n📊 Final Result:")
        if clamped_bps > 0:
            print(f"  ✅ Spread: {clamped_bps:.1f} bps")
        else:
            print(f"  ❌ Spread: {clamped_bps:.1f} bps (ZERO!)")
            
        # Check if the spread would be reported as 0.0bps in the trading system
        # This happens when the formatted display rounds to 0.0
        if round(clamped_bps, 1) == 0.0:
            print(f"\n⚠️ ISSUE DETECTED: When formatted with 1 decimal place, {clamped_bps:.2f} bps rounds to 0.0 bps")
            print("  This is why the trading system shows '0.0bps' even though the actual value is non-zero")
            
        return clamped_bps > 0
        
    except Exception as e:
        print(f"  ❌ Calculation error: {e}")
        return False

def test_different_parameters():
    """Test with different parameter values"""
    
    print("\n🧪 Testing different parameter combinations:")
    print("=" * 50)
    
    test_cases = [
        {"gamma": 0.1, "k": 1.5},
        {"gamma": 0.01, "k": 1.0},
        {"gamma": 0.001, "k": 0.5},
        {"gamma": 1.0, "k": 2.0},
        {"gamma": 0.1, "k": 0.1},  # Added more extreme case
    ]
    
    for i, params in enumerate(test_cases):
        gamma = params["gamma"]
        k = params["k"]
        
        try:
            delta_star = (1 / gamma) * math.log(1 + gamma / k)
            optimal_spread = 2 * delta_star
            spread_bps = (optimal_spread / 50000) * 10000
            
            # Check if this would be clamped
            min_spread_bps = mm_config.min_spread_bps
            max_spread_bps = mm_config.max_spread_bps
            
            final_bps = max(min_spread_bps, min(max_spread_bps, spread_bps))
            
            print(f"  Test {i+1}: γ={gamma}, k={k} → {spread_bps:.2f} bps → after clamping: {final_bps:.2f} bps")
            
        except Exception as e:
            print(f"  Test {i+1}: γ={gamma}, k={k} → ERROR: {e}")

def test_actual_quoter():
    """Test the actual AvellanedaStoikovQuoter implementation"""
    print("\n🔬 Testing actual quoter implementation:")
    print("=" * 50)
    
    quoter = AvellanedaStoikovQuoter('BTCUSDT')
    mid_price = 50000
    inventory = 0.0
    
    # Test with different parameters
    original_gamma = mm_config.gamma
    original_k = mm_config.k
    
    test_params = [
        {"gamma": 0.1, "k": 1.5, "label": "Default"},
        {"gamma": 0.01, "k": 0.5, "label": "Lower gamma & k"},
        {"gamma": 0.5, "k": 0.5, "label": "Higher gamma"},
    ]
    
    for params in test_params:
        mm_config.gamma = params["gamma"]
        mm_config.k = params["k"]
        
        try:
            # Compute quotes with the quoter
            quotes = quoter.compute_quotes(mid_price, inventory)
            
            # Extract values
            spread = quotes['optimal_spread']
            spread_bps = quotes['spread_bps']
            bid = quotes['bid_price']
            ask = quotes['ask_price']
            
            print(f"  {params['label']}: γ={params['gamma']}, k={params['k']}")
            print(f"    Spread: {spread:.6f} ({spread_bps:.2f} bps)")
            print(f"    Bid: {bid:.2f}, Ask: {ask:.2f}")
            
            # Check if spread is too small when formatted
            if round(spread_bps, 1) == 0.0:
                print(f"    ⚠️ WARNING: Spread rounds to 0.0 bps when formatted with 1 decimal place")
            
        except Exception as e:
            print(f"  {params['label']}: ERROR: {e}")
    
    # Restore original values
    mm_config.gamma = original_gamma
    mm_config.k = original_k

if __name__ == "__main__":
    success = debug_spread_calculation()
    test_different_parameters()
    test_actual_quoter()
    
    if not success:
        print("\n🔧 RECOMMENDED FIXES:")
        print("1. Check that gamma > 0 and k > 0")
        print("2. Ensure 1 + gamma/k > 0 (for log calculation)")
        print("3. Verify min_spread_bps is reasonable (should be > 0)")
        print("4. Consider increasing gamma or decreasing k for larger spreads")
        print("5. If spread is very small but non-zero, it might display as 0.0 due to rounding")
        print("\n💡 SOLUTION OPTIONS:")
        print("A. Increase min_spread_bps to at least 5.0")
        print("B. Decrease k to 0.5 (increases spread)")
        print("C. Increase gamma to 0.5 (increases spread)")
