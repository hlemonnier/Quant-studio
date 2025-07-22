"""
Test for the spread calculation fix

This test verifies that the spread calculation works correctly
after fixing the units normalization bug.
"""

import sys
import pathlib
import logging

# Enable imports
repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from strategies.MM.config import mm_config
from strategies.MM.avellaneda_stoikov import AvellanedaStoikovQuoter

# Configure logging
logging.basicConfig(level=logging.INFO)


def test_spread_calculation():
    """Test that spread calculation works correctly"""
    
    print("🧪 Testing spread calculation fix...")
    print(f"Parameters: γ={mm_config.gamma}, k={mm_config.k}, T={mm_config.T}")
    
    quoter = AvellanedaStoikovQuoter('BTCUSDT')
    
    # Test spread calculation
    inventory = 0.1
    mid_price = 50000
    
    # Compute quotes
    quotes = quoter.compute_quotes(mid_price, inventory, ofi=0.5)
    
    # Extract values
    spread = quotes['optimal_spread']
    spread_bps = quotes['spread_bps']
    bid = quotes['bid_price']
    ask = quotes['ask_price']
    center_shift = quotes['center_shift']
    
    print(f"✅ Optimal spread: {spread:.6f} ({spread_bps:.1f} bps)")
    
    print(f"\n📊 Quote Test Results:")
    print(f"  Mid Price: ${quotes['mid_price']:.2f}")
    print(f"  Reservation Price: ${quotes['reservation_price']:.2f}")
    print(f"  Bid: ${quotes['bid_price']:.2f}")
    print(f"  Ask: ${quotes['ask_price']:.2f}")
    print(f"  Spread: {quotes['spread_bps']:.1f} bps")
    print(f"  OFI shift: {center_shift:.4f}")
    
    # Validate
    assert spread_bps >= mm_config.min_spread_bps, f"Spread {spread_bps} bps should be >= {mm_config.min_spread_bps} bps"
    assert bid < ask, "Bid should be less than ask"
    
    print("✅ Spread calculation fixed!")
    return True


if __name__ == "__main__":
    test_spread_calculation()
