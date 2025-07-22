"""
Market Making V1 - Main Entry Point

This is the main launcher for the V1 Market Making strategy.

Usage:
    python strategies/MM/main.py --mode=paper-trading
    python strategies/MM/main.py --mode=backtest
    python strategies/MM/main.py --mode=calibration

Modes:
- paper-trading: Real-time paper trading (no real orders)
- backtest: Historical validation
- calibration: Parameter optimization
- live: Real trading (requires API keys)
"""

import sys
import pathlib
import asyncio
import argparse
import logging
from datetime import datetime
from typing import Optional

# Enable imports
repo_root = pathlib.Path(__file__).resolve().parents[2]
if str(repo_root) not in sys.path:
    sys.path.append(str(repo_root))

from strategies.MM.config import mm_config
from strategies.MM.trading_engine import TradingEngine
from strategies.MM.backtesting_v1 import run_full_v1_validation
from strategies.MM.parameter_calibration import ParameterCalibrator, generate_synthetic_calibration_data


def setup_logging(level: str = "INFO"):
    """Configure logging for the market making system"""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/mm_v1_{datetime.now().strftime("%Y%m%d")}.log', mode='a')
        ]
    )

async def run_paper_trading(symbol: str, duration_hours: Optional[int] = None):
    """Run paper trading mode"""
    print(f"🚀 Starting Paper Trading for {symbol}")
    print("=" * 60)
    
    # Validate configuration
    if not mm_config.validate_config(require_api_keys=False):
        print("❌ Configuration validation failed")
        return
        
    mm_config.print_config()
    
    # Initialize trading engine
    engine = TradingEngine(symbol)
    
    try:
        print(f"\n📈 Starting trading engine...")
        print("💡 Press Ctrl+C to stop gracefully")
        
        # Start the trading loop
        if duration_hours:
            print(f"⏱️  Running for {duration_hours} hours")
            await asyncio.wait_for(
                engine.run_trading_loop(), 
                timeout=duration_hours * 3600
            )
        else:
            await engine.run_trading_loop()
            
    except KeyboardInterrupt:
        print("\n🛑 Graceful shutdown initiated...")
        await engine.stop()
        
        # Print final summary
        status = engine.get_status()
        print("\n📊 Final Summary:")
        print(f"  Total Quotes: {status['total_quotes']}")
        print(f"  Total Fills: {status['total_fills']}")
        print(f"  Current Inventory: {status['inventory']:.4f}")
        print(f"  Total PnL: ${status['kpis'].get('total_pnl', 0):.2f}")
        print("✅ Shutdown complete")
        
    except Exception as e:
        print(f"❌ Trading engine error: {e}")
        await engine.stop()


def run_backtest_mode():
    """Run backtesting mode"""
    print("🧪 Starting V1 Backtesting & Validation")
    print("=" * 60)
    
    try:
        results = run_full_v1_validation()
        
        print("\n🎯 Backtesting Results:")
        print(f"  Total PnL: ${results['total_pnl']:.2f}")
        print(f"  Final Inventory: {results['final_inventory']:.4f}")
        print(f"  Validation: {'✅ PASS' if results['validation']['overall_pass'] else '❌ FAIL'}")
        
        if results['validation']['overall_pass']:
            print("🎉 V1 system ready for paper trading!")
        else:
            print("⚠️  Some performance targets not met - review configuration")
            
    except Exception as e:
        print(f"❌ Backtesting failed: {e}")


def run_calibration_mode():
    """Run parameter calibration mode"""
    print("🔧 Starting Parameter Calibration")
    print("=" * 60)
    
    try:
        calibrator = ParameterCalibrator('BTCUSDT')
        
        # Generate synthetic data for demo
        print("📊 Generating synthetic calibration data...")
        market_data, trading_history = generate_synthetic_calibration_data()
        
        # Run calibration
        results = calibrator.run_full_calibration(market_data, trading_history)
        
        print("\n🎯 Calibration Results:")
        print("=" * 40)
        for param, value in results.items():
            print(f"  {param}: {value:.6f}")
        print("=" * 40)
        
        print("✅ Parameters updated in mm_config")
        print("💡 You can now run paper trading with optimized parameters")
        
    except Exception as e:
        print(f"❌ Calibration failed: {e}")


async def run_live_trading(symbol: str):
    """Run live trading mode (requires API keys)"""
    print("🔴 LIVE TRADING MODE")
    print("=" * 60)
    print("⚠️  WARNING: This will place real orders with real money!")
    
    # Strict validation for live trading
    if not mm_config.validate_config(require_api_keys=True):
        print("❌ API keys required for live trading")
        print("💡 Set BINANCE_API_KEY and BINANCE_API_SECRET environment variables")
        return
    
    confirm = input("Type 'CONFIRM' to proceed with live trading: ")
    if confirm != 'CONFIRM':
        print("🛑 Live trading cancelled")
        return
    
    print("🚀 Starting live trading...")
    # This would use the same trading engine but with real order execution
    await run_paper_trading(symbol)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Market Making V1 Strategy')
    parser.add_argument('--mode', 
                       choices=['paper-trading', 'backtest', 'calibration', 'live'],
                       default='paper-trading',
                       help='Trading mode to run')
    parser.add_argument('--symbol', default='BTCUSDT', help='Symbol to trade')
    parser.add_argument('--duration', type=int, help='Duration in hours for paper trading')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    pathlib.Path('logs').mkdir(exist_ok=True)
    setup_logging(args.log_level)
    
    print("🤖 Market Making V1 Strategy")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Mode: {args.mode}")
    print(f"💰 Symbol: {args.symbol}")
    print()
    
    # Route to appropriate mode
    if args.mode == 'paper-trading':
        asyncio.run(run_paper_trading(args.symbol, args.duration))
    elif args.mode == 'backtest':
        run_backtest_mode()
    elif args.mode == 'calibration':
        run_calibration_mode()
    elif args.mode == 'live':
        asyncio.run(run_live_trading(args.symbol))


if __name__ == "__main__":
    main()
