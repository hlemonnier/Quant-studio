#!/usr/bin/env python3
"""
Market Making V1 - Main Entry Point (Version 2)

Version améliorée avec support multi-symboles et meilleure gestion du PnL.

Usage:
    python strategies/MM/main.py --mode=paper-trading
    python strategies/MM/main.py --mode=paper-trading --symbol=all
    python strategies/MM/main.py --mode=paper-trading --symbol=ETHUSDT
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
from typing import List, Optional

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
    pathlib.Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/mm_v1_{datetime.now().strftime("%Y%m%d")}.log', mode='a')
        ]
    )


async def run_paper_trading(symbols: List[str], duration_hours: Optional[int] = None):
    """
    Run paper trading mode with support for multiple symbols
    
    Args:
        symbols: List of symbols to trade
        duration_hours: Optional duration in hours
    """
    print(f"🚀 Starting Paper Trading for {', '.join(symbols)}")
    print("=" * 60)
    
    # Validate configuration
    if not mm_config.validate_config(require_api_keys=False):
        print("❌ Configuration validation failed")
        return
    
    mm_config.print_config()
    
    # Initialize trading engines for each symbol
    engines = []
    for symbol in symbols:
        print(f"\n📈 Initializing trading engine for {symbol}...")
        engine = TradingEngine(symbol)
        engines.append(engine)
    
    try:
        print(f"\n🚀 Starting trading engines...")
        print("💡 Press Ctrl+C to stop gracefully")
        print("\n⚠️  IMPORTANT: PnL négatif est NORMAL en market making!")
        print("   Le système s'arrêtera seulement si:")
        print(f"   - Perte quotidienne > {mm_config.daily_loss_limit_pct}% (configurable)")
        print("   - Inventaire > limites configurées")
        print("   - Volatilité excessive détectée")
        
        # Start all trading loops
        tasks = []
        for engine in engines:
            tasks.append(asyncio.create_task(engine.run_trading_loop()))
        
        # Wait for all tasks or timeout
        if duration_hours:
            print(f"⏱️  Running for {duration_hours} hours")
            await asyncio.wait_for(
                asyncio.gather(*tasks),
                timeout=duration_hours * 3600
            )
        else:
            await asyncio.gather(*tasks)
    
    except KeyboardInterrupt:
        print("\n🛑 Graceful shutdown initiated...")
    except asyncio.TimeoutError:
        print("\n⏱️  Trading duration completed")
    except Exception as e:
        print(f"\n❌ Error in trading loop: {e}")
    finally:
        # Stop all engines
        for engine in engines:
            await engine.stop()
        
        # Print final summary for each engine
        print("\n📊 Final Summary:")
        for engine in engines:
            status = engine.get_status()
            print(f"\n  {engine.symbol}:")
            print(f"    Total Quotes: {status['total_quotes']}")
            print(f"    Total Fills: {status['total_fills']}")
            print(f"    Current Inventory: {status['inventory']:.4f}")
            print(f"    Total PnL: ${status['kpis'].get('total_pnl', 0):.2f}")
        
        print("✅ Shutdown complete")


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


def run_calibration_mode(symbols: List[str]):
    """Run parameter calibration mode for one or several symbols"""
    print("🔧 Starting Parameter Calibration")
    print("=" * 60)

    for sym in symbols:
        print(f"\n📈 Calibrating parameters for {sym} ...")
        try:
            calibrator = ParameterCalibrator(sym)

            # TODO: Replace synthetic generator by real historical data loading
            market_data, trading_history = generate_synthetic_calibration_data()

            results = calibrator.run_full_calibration(market_data, trading_history)

            print("\n🎯 Calibration Results for", sym)
            print("=" * 40)
            for param, value in results.items():
                print(f"  {param}: {value:.6f}")
            print("=" * 40)
            print("✅ Parameters updated and saved for", sym)

        except Exception as e:
            print(f"❌ Calibration failed for {sym}: {e}")


async def run_live_trading(symbols: List[str]):
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
    await run_paper_trading(symbols)  # Uses same engine but with real orders


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Market Making V1 Strategy')
    parser.add_argument('--mode', 
                       choices=['paper-trading', 'backtest', 'calibration', 'live'],
                       default='paper-trading',
                       help='Trading mode to run')
    parser.add_argument('--symbol', default='BTCUSDT', 
                       help='Symbol to trade (use "all" for all configured symbols)')
    parser.add_argument('--duration', type=int, help='Duration in hours for paper trading')
    parser.add_argument('--log-level', default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    pathlib.Path('logs').mkdir(exist_ok=True)
    setup_logging(args.log_level)
    
    print("🤖 Market Making V1 Strategy")
    print(f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🎯 Mode: {args.mode}")
    
    # Determine which symbols to trade
    symbols = []
    if args.symbol.lower() == 'all':
        symbols = mm_config.symbols
        print(f"💰 Symbols: {', '.join(symbols)} (all configured symbols)")
    else:
        if args.symbol in mm_config.symbols:
            symbols = [args.symbol]
            print(f"💰 Symbol: {args.symbol}")
        else:
            print(f"⚠️  Symbol {args.symbol} not in configured symbols, using {mm_config.default_symbol}")
            symbols = [mm_config.default_symbol]
            print(f"💰 Symbol: {mm_config.default_symbol}")
    
    print()
    
    # Route to appropriate mode
    if args.mode == 'paper-trading':
        asyncio.run(run_paper_trading(symbols, args.duration))
    elif args.mode == 'backtest':
        run_backtest_mode()
    elif args.mode == 'calibration':
        run_calibration_mode(symbols)
    elif args.mode == 'live':
        asyncio.run(run_live_trading(symbols))


if __name__ == "__main__":
    main()
