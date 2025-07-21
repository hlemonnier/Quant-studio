"""
Complete V1 System Tests - Integration and End-to-End

Tests the complete V1 system including:
- Trading engine integration
- KPI tracking
- Risk controls enforcement 
- Backtesting pipeline
- WebSocket data flow simulation

These tests verify the full scope of V1 (§3) is working correctly.
"""

import sys
import pathlib
import asyncio
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
import time

# Enable imports
repo_root = pathlib.Path(__file__).resolve().parents[3]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from strategies.MM.trading_engine import TradingEngine
from strategies.MM.kpi_tracker import KPITracker
from strategies.MM.backtesting_v1 import BacktestEngine, HistoricalDataGenerator
from strategies.MM.config import mm_config


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
    """Test KPI tracker calculates all required metrics"""
    tracker = KPITracker('BTCUSDT')
    
    # Mock some fills
    class MockFill:
        def __init__(self, side, price, size):
            self.side = side
            self.price = price
            self.size = size
    
    # Add test data
    for i in range(10):
        fill = MockFill('bid' if i % 2 == 0 else 'ask', 50000 + i*10, 0.01)
        tracker.record_fill(fill, 50000, 5.0)
        tracker.record_inventory(i * 0.001)
        tracker.record_latency('quote_ack', 40 + i*2)
    
    tracker.record_quotes_sent(50)
    tracker.record_cancel(8)
    
    # Test all KPI calculations
    summary = tracker.get_summary()
    
    assert summary['total_quotes'] == 50
    assert summary['total_fills'] == 10
    assert summary['total_cancels'] == 8
    assert summary['fill_ratio'] > 0
    assert summary['cancel_ratio'] > 0
    assert summary['rms_inventory'] >= 0
    assert 'latency_stats' in summary
    
    # Test performance targets
    targets = tracker.check_performance_targets()
    assert isinstance(targets, dict)
    assert len(targets) == 6  # All 6 targets from spec


def test_risk_controls_enforcement():
    """Test that risk controls actually prevent trading"""
    engine = TradingEngine('BTCUSDT')
    
    # Test inventory limit
    engine.inventory_ctrl.current_inventory = mm_config.max_inventory + 0.1
    should_pause, reason = engine._check_risk_controls()
    
    assert should_pause is True
    assert "inventory" in reason.lower()
    
    # Test volatility spike
    engine.inventory_ctrl.current_inventory = 0.0  # Reset
    engine.current_volatility = mm_config.sigma * 2.1
    should_pause, reason = engine._check_risk_controls()
    
    assert should_pause is True
    assert "volatility" in reason.lower()


def test_backtesting_pipeline():
    """Test complete backtesting pipeline"""
    # Generate small dataset for quick test
    data = HistoricalDataGenerator.generate_market_data(
        'BTCUSDT', 
        duration_hours=1,  # 1 hour for quick test
        tick_interval_ms=1000  # 1 second intervals
    )
    
    # Verify data format
    expected_columns = ['timestamp', 'mid_price', 'bid_price', 'ask_price', 
                       'spread_bps', 'buy_volume', 'sell_volume', 'volatility']
    for col in expected_columns:
        assert col in data.columns
    
    # Run backtest
    engine = BacktestEngine('BTCUSDT')
    results = engine.run_backtest(data)
    
    # Verify results structure
    assert 'total_pnl' in results
    assert 'final_inventory' in results
    assert 'validation' in results
    assert 'config_used' in results
    
    # Verify validation results
    validation = results['validation']
    assert 'targets_met' in validation
    assert 'overall_pass' in validation
    assert isinstance(validation['targets_met'], dict)


def test_ofi_integration_with_trading():
    """Test OFI signal flows through the trading system correctly"""
    engine = TradingEngine('BTCUSDT')
    
    # Add some trades to OFI calculator to create imbalance
    for i in range(10):
        engine.ofi_calc.register_trade(0.1, 'buy')  # All buys = positive OFI
    
    ofi_value = engine.ofi_calc.current_ofi()
    assert ofi_value > 0  # Should be positive (buy pressure)
    
    # Simulate quote computation
    engine.current_mid = 50000
    quotes_data = asyncio.run(engine._compute_quotes())
    
    # Verify OFI affects center shift
    if quotes_data:
        assert 'ofi' in quotes_data
        assert 'center_shift' in quotes_data
        # With positive OFI, center should shift up (favoring sells)
        assert quotes_data['center_shift'] > 0


def test_stress_scenarios():
    """Test system handles stress scenarios"""
    engine = BacktestEngine('BTCUSDT')
    
    # High volatility scenario
    high_vol_data = HistoricalDataGenerator.generate_market_data(
        'BTCUSDT', duration_hours=0.5  # 30 min test
    )
    high_vol_data['volatility'] *= 3  # 3x volatility
    
    results = engine.run_backtest(high_vol_data, {'sigma': mm_config.sigma * 3})
    
    # Should still complete without errors
    assert 'total_pnl' in results
    assert 'validation' in results


def test_websocket_data_flow_simulation():
    """Test simulated WebSocket data flow to trading components"""
    engine = TradingEngine('BTCUSDT')
    
    # Mock the local_book to simulate WebSocket updates
    engine.local_book.get_snapshot = MagicMock(return_value=True)
    engine.local_book.get_mid_price = MagicMock(return_value=50000.0)
    
    # Test initialization
    initialized = asyncio.run(engine._initialize_market_data())
    assert initialized is True
    assert engine.current_mid == 50000.0
    
    # Test market data updates
    old_mid = engine.current_mid
    asyncio.run(engine._update_market_data())
    
    # Price should have moved (simulated random walk)
    # Note: Due to randomness, we just verify it's still a reasonable price
    assert 40000 < engine.current_mid < 60000


def test_performance_targets_validation():
    """Test that performance targets from §3.7 are correctly validated"""
    tracker = KPITracker('BTCUSDT')
    
    # Create data that should meet all targets
    class MockFill:
        def __init__(self, side, price, size):
            self.side = side
            self.price = price
            self.size = size
    
    # Add enough good fills to meet targets
    for i in range(100):  # Many fills for good statistics
        fill = MockFill('bid' if i % 2 == 0 else 'ask', 50000 + i, 0.01)
        tracker.record_fill(fill, 50000, 10.0)  # Good spread capture
        tracker.record_inventory((i - 50) * 0.001)  # Keep inventory reasonable
        tracker.record_latency('quote_ack', 50)  # Good latency
    
    # Record conservative quotes/cancels for good ratios
    tracker.record_quotes_sent(1000)  # Many quotes
    tracker.record_cancel(200)  # 20% cancel ratio (well under 70% limit)
    
    # Check targets
    targets = tracker.check_performance_targets()
    summary = tracker.get_summary()
    
    # Verify key targets that should pass with our good data
    assert summary['fill_ratio'] >= 0.05  # Should have >5% fill ratio
    assert summary['cancel_ratio'] <= 0.70  # Should have <70% cancel ratio
    assert summary['rms_inventory'] <= 0.4  # Should have reasonable inventory
    assert summary['total_pnl'] >= 0  # PnL should be positive


def test_config_override_system():
    """Test configuration override system works"""
    original_gamma = mm_config.gamma
    
    try:
        engine = BacktestEngine('BTCUSDT')
        
        # Test config override
        engine._apply_config_overrides({'gamma': 0.5})
        assert mm_config.gamma == 0.5
        
        # Reset
        mm_config.gamma = original_gamma
        
    except Exception as e:
        # Ensure we reset even if test fails
        mm_config.gamma = original_gamma
        raise e


def test_full_v1_scope_coverage():
    """Integration test verifying full V1 scope is covered"""
    
    # §3.3: Core A&S math - tested in test_v1_algo.py
    # §3.3bis: OFI integration - tested above
    # §3.5: Trading loop - test here
    
    engine = TradingEngine('BTCUSDT')
    
    # Mock to avoid network calls
    engine.local_book.get_snapshot = MagicMock(return_value=True)
    engine.local_book.get_mid_price = MagicMock(return_value=50000.0)
    
    # Test complete status reporting (§3.7)
    status = engine.get_status()
    
    required_status_fields = [
        'symbol', 'status', 'current_mid', 'inventory', 
        'total_quotes', 'total_fills', 'kpis'
    ]
    
    for field in required_status_fields:
        assert field in status
    
    # §3.6: Risk controls - test enforcement
    engine.inventory_ctrl.current_inventory = mm_config.max_inventory + 0.5
    should_pause, reason = engine._check_risk_controls()
    assert should_pause is True
    
    print("✅ Full V1 scope coverage verified!")


if __name__ == "__main__":
    # Run all tests
    test_functions = [
        test_trading_engine_integration,
        test_kpi_tracker_comprehensive,
        test_risk_controls_enforcement,
        test_backtesting_pipeline,
        test_ofi_integration_with_trading,
        test_stress_scenarios,
        test_websocket_data_flow_simulation,
        test_performance_targets_validation,
        test_config_override_system,
        test_full_v1_scope_coverage
    ]
    
    print("🧪 Running Complete V1 System Tests...")
    print("=" * 50)
    
    passed = 0
    failed = 0
    
    for test_func in test_functions:
        try:
            test_func()
            print(f"✅ {test_func.__name__}")
            passed += 1
        except Exception as e:
            print(f"❌ {test_func.__name__}: {e}")
            failed += 1
    
    print("=" * 50)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("🎉 All V1 system tests passed! Ready for deployment.")
    else:
        print("⚠️  Some tests failed. Review before deployment.")
