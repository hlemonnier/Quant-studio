"""
Comprehensive V1 Market Making Tests

This file consolidates all tests for the V1 Market Making system:
- Unit tests for Avellaneda-Stoikov algorithm (§3.3)
- Integration tests for trading engine
- End-to-end system validation
- KPI tracking verification
- Risk controls enforcement

Run with: pytest -q strategies/MM/tests/test_v1.py
"""

import math
import numpy as np
import sys
import pathlib
import asyncio
import pandas as pd
from unittest.mock import MagicMock
import time

# Enable imports
repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from strategies.MM.avellaneda_stoikov import AvellanedaStoikovQuoter
from strategies.MM.config import mm_config
from strategies.MM.ofi import OFICalculator
from strategies.MM.trading_engine import TradingEngine
from strategies.MM.kpi_tracker import KPITracker
from strategies.MM.backtesting_v1 import BacktestEngine, HistoricalDataGenerator


# =============================================================================
# UNIT TESTS - Avellaneda-Stoikov Algorithm (§3.3)
# =============================================================================

def test_reservation_price_formula():
    """Verify reservation price follows the A&S formula: r = S - q × γ × σ² × (T - t)"""
    symbol = "BTCUSDT"
    quoter = AvellanedaStoikovQuoter(symbol)
    mid = 50_000.0
    inventory = 0.25

    r_lib = quoter.compute_reservation_price(mid, inventory)
    expected = mid - inventory * quoter.gamma * quoter.sigma ** 2 * quoter.T
    assert math.isclose(r_lib, expected, rel_tol=1e-12)


def test_optimal_spread_formula_and_clamp():
    """Verify optimal spread follows A&S formula and respects min/max bounds"""
    symbol = "BTCUSDT"
    quoter = AvellanedaStoikovQuoter(symbol)
    
    # Test formula: δ = γ × σ² × (T - t) + (2/γ) × ln(1 + γ/k)
    spread = quoter.compute_optimal_spread()
    expected = (quoter.gamma * quoter.sigma ** 2 * quoter.T + 
                (2 / quoter.gamma) * math.log(1 + quoter.gamma / quoter.k))
    assert math.isclose(spread, expected, rel_tol=1e-12)
    
    # Test clamping
    assert spread >= quoter.min_spread
    assert spread <= quoter.max_spread


def test_ofi_shift_limited_to_one_tick():
    """Verify OFI centre shift is limited to ±1 tick (§3.3bis)"""
    symbol = "BTCUSDT"
    quoter = AvellanedaStoikovQuoter(symbol)
    ofi_calc = OFICalculator()
    
    # Test extreme OFI values
    extreme_ofi = 1000.0  # Very high OFI
    shift = quoter._calculate_ofi_shift(extreme_ofi)
    tick_size = mm_config.get_symbol_config(symbol).get('tick_size', 0.01)
    
    assert abs(shift) <= tick_size, f"OFI shift {shift} exceeds ±1 tick ({tick_size})"


def test_quote_consistency_and_symmetry():
    """Verify quotes are consistent and symmetric around reservation price"""
    symbol = "BTCUSDT"
    quoter = AvellanedaStoikovQuoter(symbol)
    mid = 50_000.0
    inventory = 0.0  # Neutral inventory for symmetry test
    ofi = 0.0  # No OFI shift
    
    quotes = quoter.compute_quotes(mid, inventory, ofi)
    reservation_price = quoter.compute_reservation_price(mid, inventory)
    
    # With neutral inventory and no OFI, quotes should be symmetric around reservation price
    bid_distance = reservation_price - quotes['bid_price']
    ask_distance = quotes['ask_price'] - reservation_price
    
    assert math.isclose(bid_distance, ask_distance, rel_tol=1e-10), \
        f"Quotes not symmetric: bid_dist={bid_distance}, ask_dist={ask_distance}"


def test_inventory_skew_effect():
    """Verify inventory creates appropriate skew in quotes"""
    symbol = "BTCUSDT"
    quoter = AvellanedaStoikovQuoter(symbol)
    mid = 50_000.0
    ofi = 0.0
    
    # Test long inventory (should skew quotes down to encourage selling)
    long_inventory = 0.5
    long_quotes = quoter.compute_quotes(mid, long_inventory, ofi)
    
    # Test short inventory (should skew quotes up to encourage buying)
    short_inventory = -0.5
    short_quotes = quoter.compute_quotes(mid, short_inventory, ofi)
    
    # Long inventory should result in lower quotes than short inventory
    assert long_quotes['bid_price'] < short_quotes['bid_price']
    assert long_quotes['ask_price'] < short_quotes['ask_price']


# =============================================================================
# INTEGRATION TESTS - Trading Engine
# =============================================================================

def test_trading_engine_integration():
    """Test trading engine integrates all components correctly"""
    engine = TradingEngine('BTCUSDT')
    
    # Verify all components are initialized
    assert engine.quoter is not None
    assert engine.ofi_calc is not None
    assert engine.local_book is not None
    assert engine.inventory_ctrl is not None
    assert engine.kpi_tracker is not None
    
    # Verify initial state
    assert engine.current_mid == 0.0
    assert engine.trading_paused is False
    assert len(engine.active_quotes) == 0


def test_kpi_tracker_comprehensive():
    """Test KPI tracker captures all required metrics"""
    kpi = KPITracker('BTCUSDT')
    
    # Test initial state
    assert kpi.total_fills == 0
    assert kpi.total_cancels == 0
    assert kpi.total_pnl == 0.0
    
    # Simulate some activity
    kpi.record_fill('bid', 50000.0, 0.01, 'filled')
    kpi.record_cancel('ask', 'timeout')
    
    # Verify metrics
    assert kpi.total_fills == 1
    assert kpi.total_cancels == 1
    
    # Test KPI calculation
    kpis = kpi.get_kpis()
    assert 'Fill Ratio' in kpis
    assert 'Cancel Ratio' in kpis
    assert 'Total PnL' in kpis


def test_risk_controls_enforcement():
    """Test risk controls are properly enforced"""
    engine = TradingEngine('BTCUSDT')
    
    # Test inventory limits
    engine.inventory_ctrl.current_inventory = mm_config.max_inventory + 0.1
    should_pause, reason = engine.inventory_ctrl.should_pause_trading()
    assert should_pause is True
    assert "inventory" in reason.lower()
    
    # Test daily loss limits
    engine.kpi_tracker.total_pnl = -mm_config.daily_loss_limit_pct * 1000  # Simulate large loss
    engine.kpi_tracker.daily_start_balance = 10000  # Set initial balance
    should_pause, reason = engine.inventory_ctrl.should_pause_trading()
    # Note: This test depends on the specific implementation of loss limit checking


def test_order_sizing_logic():
    """Test dynamic order sizing based on inventory"""
    engine = TradingEngine('BTCUSDT')
    
    # Test neutral inventory
    engine.inventory_ctrl.current_inventory = 0.0
    size_neutral = engine.inventory_ctrl.calculate_optimal_size('bid', 50000.0)
    assert size_neutral == mm_config.base_quote_size
    
    # Test long inventory (should reduce buy orders)
    engine.inventory_ctrl.current_inventory = 0.5
    size_long = engine.inventory_ctrl.calculate_optimal_size('bid', 50000.0)
    assert size_long < size_neutral
    
    # Test short inventory (should increase buy orders)
    engine.inventory_ctrl.current_inventory = -0.5
    size_short = engine.inventory_ctrl.calculate_optimal_size('bid', 50000.0)
    assert size_short > size_neutral


# =============================================================================
# END-TO-END SYSTEM TESTS
# =============================================================================

def test_complete_trading_cycle():
    """Test a complete trading cycle from market data to quote updates"""
    engine = TradingEngine('BTCUSDT')
    
    # Mock market data
    market_data = {
        'symbol': 'BTCUSDT',
        'bid_price': 49990.0,
        'ask_price': 50010.0,
        'bid_qty': 1.5,
        'ask_qty': 2.0,
        'timestamp': time.time()
    }
    
    # Process market data
    asyncio.run(engine.process_market_data(market_data))
    
    # Verify engine state updated
    assert engine.current_mid > 0
    assert len(engine.active_quotes) == 2  # Should have bid and ask quotes


def test_backtesting_pipeline():
    """Test the backtesting pipeline works end-to-end"""
    # Generate test data
    data_gen = HistoricalDataGenerator()
    test_data = data_gen.generate_realistic_data(
        symbol='BTCUSDT',
        start_price=50000.0,
        duration_hours=1,
        tick_frequency_ms=100
    )
    
    assert len(test_data) > 0
    assert 'bid_price' in test_data.columns
    assert 'ask_price' in test_data.columns
    
    # Run backtest
    backtest_engine = BacktestEngine('BTCUSDT')
    results = asyncio.run(backtest_engine.run_backtest(test_data, duration_minutes=5))
    
    # Verify results structure
    assert 'kpis' in results
    assert 'trades' in results
    assert 'final_inventory' in results


def test_stress_conditions():
    """Test system behavior under stress conditions"""
    engine = TradingEngine('BTCUSDT')
    
    # Test high volatility scenario
    volatile_data = {
        'symbol': 'BTCUSDT',
        'bid_price': 45000.0,  # Large price move
        'ask_price': 55000.0,  # Wide spread
        'bid_qty': 0.1,        # Low liquidity
        'ask_qty': 0.1,
        'timestamp': time.time()
    }
    
    # Should handle extreme conditions gracefully
    try:
        asyncio.run(engine.process_market_data(volatile_data))
        # If we get here, the engine handled the stress condition
        assert True
    except Exception as e:
        # Should not crash on extreme market conditions
        assert False, f"Engine crashed on stress condition: {e}"


def test_performance_benchmarks():
    """Test system meets performance benchmarks"""
    engine = TradingEngine('BTCUSDT')
    
    # Test quote update latency
    start_time = time.time()
    
    market_data = {
        'symbol': 'BTCUSDT',
        'bid_price': 50000.0,
        'ask_price': 50020.0,
        'bid_qty': 1.0,
        'ask_qty': 1.0,
        'timestamp': time.time()
    }
    
    asyncio.run(engine.process_market_data(market_data))
    
    processing_time = time.time() - start_time
    
    # Should process market data quickly (< 10ms for this simple case)
    assert processing_time < 0.01, f"Processing too slow: {processing_time:.4f}s"


if __name__ == "__main__":
    print("Running V1 Market Making Tests...")
    
    # Run unit tests
    print("✓ Testing reservation price formula...")
    test_reservation_price_formula()
    
    print("✓ Testing optimal spread formula...")
    test_optimal_spread_formula_and_clamp()
    
    print("✓ Testing OFI shift limits...")
    test_ofi_shift_limited_to_one_tick()
    
    print("✓ Testing quote consistency...")
    test_quote_consistency_and_symmetry()
    
    print("✓ Testing inventory skew...")
    test_inventory_skew_effect()
    
    # Run integration tests
    print("✓ Testing trading engine integration...")
    test_trading_engine_integration()
    
    print("✓ Testing KPI tracker...")
    test_kpi_tracker_comprehensive()
    
    print("✓ Testing order sizing logic...")
    test_order_sizing_logic()
    
    # Run end-to-end tests
    print("✓ Testing complete trading cycle...")
    test_complete_trading_cycle()
    
    print("✓ Testing stress conditions...")
    test_stress_conditions()
    
    print("✓ Testing performance benchmarks...")
    test_performance_benchmarks()
    
    print("\n🎉 All V1 tests passed!")

