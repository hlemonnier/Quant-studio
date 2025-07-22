"""
Quick diagnostic to identify the exact spread calculation issue
and provide a direct fix
"""

import sys
import pathlib
import math

# Enable imports
repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from strategies.MM.config import mm_config

def diagnose_spread_issue():
    """Diagnose the exact spread calculation issue"""
    
    print("🔍 SPREAD ISSUE DIAGNOSIS")
    print("=" * 60)
    
    gamma = mm_config.gamma  # 0.1
    k = mm_config.k         # 1.5
    mid_price = 50000       # Test price
    
    print(f"Parameters: γ={gamma}, k={k}")
    print(f"Mid price: ${mid_price}")
    
    # Step 1: Calculate delta_star (half-spread in units)
    delta_star = (1 / gamma) * math.log(1 + gamma / k)
    print(f"\n1. Half-spread δ* = (1/γ) × ln(1+γ/k) = {delta_star:.6f}")
    
    # Step 2: Full spread in units
    full_spread_units = 2 * delta_star
    print(f"2. Full spread (units) = 2 × δ* = {full_spread_units:.6f}")
    
    # Step 3: This is where the bug is! 
    # The A&S formula gives spread in UNITS, not as a fraction of price
    # We need to convert to fraction for comparison with bps
    
    # WRONG: Treating spread_units as fraction
    wrong_spread_bps = full_spread_units * 10000
    print(f"\n❌ WRONG calculation:")
    print(f"   Treating {full_spread_units:.6f} as fraction → {wrong_spread_bps:.1f} bps (absurd!)")
    
    # CORRECT: The A&S spread should be normalized by mid_price first
    # OR we should use min_spread in the same units
    
    print(f"\n✅ CORRECT approaches:")
    
    # Approach 1: Normalize spread by price, then compare bps
    spread_as_fraction = full_spread_units / mid_price
    spread_bps = spread_as_fraction * 10000
    print(f"   A1: Normalize by price: {full_spread_units:.6f} / {mid_price} = {spread_as_fraction:.6f} = {spread_bps:.2f} bps")
    
    # Apply min/max in bps
    min_bps = mm_config.min_spread_bps  # 5
    max_bps = mm_config.max_spread_bps  # 200
    clamped_bps = max(min_bps, min(max_bps, spread_bps))
    final_spread_fraction = clamped_bps / 10000
    
    print(f"   A1: After clamping: {clamped_bps:.1f} bps = {final_spread_fraction:.6f} fraction")
    
    # Approach 2: Convert min/max to same units as A&S output
    min_spread_units = (min_bps / 10000) * mid_price
    max_spread_units = (max_bps / 10000) * mid_price
    clamped_units = max(min_spread_units, min(max_spread_units, full_spread_units))
    clamped_units_bps = (clamped_units / mid_price) * 10000
    
    print(f"   A2: Convert limits to units: min={min_spread_units:.2f}, max={max_spread_units:.2f}")
    print(f"   A2: Clamped: {clamped_units:.6f} units = {clamped_units_bps:.1f} bps")
    
    return final_spread_fraction

def show_current_bug():
    """Show what the current implementation does wrong"""
    print(f"\n🐛 CURRENT BUG IN CODE:")
    print(f"   optimal_spread = 2 × δ* = {1.29:.6f} (this is in UNITS, not fraction)")
    print(f"   spread_bps = optimal_spread × 10000 = {1.29 * 10000:.0f} bps (WRONG!)")
    print(f"   This gets clamped to max 200 bps")
    print(f"   Then spread_bps / mid_price × 10000 gives tiny value")

def test_fix():
    """Test the correct implementation"""
    print(f"\n🔧 TESTING CORRECT FIX:")
    
    gamma = mm_config.gamma
    k = mm_config.k
    mid_price = 119089.99  # Current BTC price from logs
    
    # Correct A&S calculation
    delta_star = (1 / gamma) * math.log(1 + gamma / k)
    spread_units = 2 * delta_star
    
    # Convert to fraction for proper handling
    spread_fraction = spread_units / mid_price
    spread_bps = spread_fraction * 10000
    
    print(f"   Mid price: ${mid_price:.2f}")
    print(f"   Spread (units): {spread_units:.6f}")
    print(f"   Spread (fraction): {spread_fraction:.8f}")  
    print(f"   Spread (bps): {spread_bps:.4f}")
    
    # Apply limits
    min_bps = mm_config.min_spread_bps
    max_bps = mm_config.max_spread_bps
    final_bps = max(min_bps, min(max_bps, spread_bps))
    final_fraction = final_bps / 10000
    
    print(f"   After limits: {final_bps:.1f} bps = {final_fraction:.6f} fraction")
    
    # Calculate quotes
    half_spread = final_fraction / 2
    bid = mid_price * (1 - half_spread) 
    ask = mid_price * (1 + half_spread)
    
    print(f"   Bid: ${bid:.2f}")
    print(f"   Ask: ${ask:.2f}")
    print(f"   Actual spread: ${ask - bid:.2f} = {((ask - bid) / mid_price) * 10000:.1f} bps")
    
if __name__ == "__main__":
    diagnose_spread_issue()
    show_current_bug()
    test_fix()
    
    print(f"\n💡 SOLUTION:")
    print(f"   The A&S formula gives spread in absolute units ($)")
    print(f"   Must divide by mid_price to get fraction before applying bps limits")
    print(f"   Current code wrongly treats units as fraction → massive bps values")
