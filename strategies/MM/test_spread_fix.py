"""
Quick test to verify the spread calculation is working correctly
"""

import sys
import pathlib

# Enable imports
repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from strategies.MM.config import mm_config
from strategies.MM.avellaneda_stoikov import AvellanedaStoikovQuoter

def test_spread_calculation():
    """Test that spread calculation works correctly"""
    
    print("🧪 Testing spread calculation fix...")
    print(f"Parameters: γ={mm_config.gamma}, k={mm_config.k}, T={mm_config.T}")
    
    quoter = AvellanedaStoikovQuoter('BTCUSDT')
    
    # Test spread calculation
    inventory = 0.1
    spread = quoter.compute_optimal_spread(inventory)
    
    print(f"✅ Optimal spread: {spread:.6f} ({spread*10000:.1f} bps)")
    
    # Test quote generation
    mid_price = 50000
    quotes = quoter.compute_quotes(mid_price, inventory, ofi=0.5)
    
    print(f"\n📊 Quote Test Results:")
    print(f"  Mid Price: ${quotes['mid_price']:.2f}")
    print(f"  Reservation Price: ${quotes['reservation_price']:.2f}")
    print(f"  Bid: ${quotes['bid_price']:.2f}")
    print(f"  Ask: ${quotes['ask_price']:.2f}")
    print(f"  Spread: {quotes['spread_bps']:.1f} bps")
    print(f"  OFI shift: {quotes['center_shift']:.4f}")
    
    # Validate
    if quotes['spread_bps'] > 0:
        print("✅ Spread calculation fixed!")
        return True
    else:
        print("❌ Spread still 0 - check parameters")
        return False

if __name__ == "__main__":
    test_spread_calculation()
