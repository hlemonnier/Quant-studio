#!/usr/bin/env python3
"""
Market Making V1.5 - Main Entry Point

Enhanced version with V1.5 support including:
- Depth Imbalance (DI) signal integration
- Quote ageing system
- Enhanced multi-signal pricing
- Version selection support

Usage:
    python strategies/MM/main.py --mode=paper-trading --version=V1.5
    python strategies/MM/main.py --mode=paper-trading --version=V1-α
    python strategies/MM/main.py --mode=paper-trading --symbol=all --version=V1.5
    python strategies/MM/main.py --mode=backtest
    python strategies/MM/main.py --mode=calibration

Modes:
- paper-trading: Real-time paper trading (no real orders)
- backtest: Historical validation
- calibration: Parameter optimization
- live: Real trading (requires API keys)

Versions:
- V1-α: Original Avellaneda-Stoikov with OFI
- V1.5: Enhanced with DI, quote ageing, and dynamic spread
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

from strategies.MM.utils.config import mm_config
from strategies.MM.trading_engine import TradingEngine
from strategies.MM.simple_backtest import run_simple_backtest
from strategies.MM.utils.parameter_calibration import ParameterCalibrator, generate_synthetic_calibration_data


def setup_logging(level: str = "INFO"):
    """Configure logging for the market making system"""
    pathlib.Path('logs').mkdir(exist_ok=True)
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(f'logs/mm_{datetime.now().strftime("%Y%m%d")}.log', mode='a')
        ]
    )


async def run_paper_trading(symbols: List[str], duration_hours: Optional[int] = None, version: str = "V1-α"):
    """
    Run paper trading mode with support for multiple symbols and versions
    
    Args:
        symbols: List of symbols to trade
        duration_hours: Optional duration in hours
        version: Strategy version ("V1-α" or "V1.5")
    """
    print(f"🚀 Starting Paper Trading {version} for {', '.join(symbols)}")
    print("=" * 60)
    
    # Validate configuration
    if not mm_config.validate_config(require_api_keys=False):
        print("❌ Configuration validation failed")
        return
    
    # Set version in config
    mm_config.strategy_version = version
    mm_config.print_config()
    
    # Initialize trading engines for each symbol
    engines = []
    for symbol in symbols:
        print(f"\n📈 Initializing {version} trading engine for {symbol}...")
        engine = TradingEngine(symbol, version=version)
        engines.append(engine)
    
    try:
        print(f"\n🚀 Starting trading engines...")
        print("💡 Press Ctrl+C to stop gracefully")
        print(f"\n⚠️  IMPORTANT: PnL négatif est NORMAL en market making!")
        print("   Le système s'arrêtera seulement si:")
        print(f"   - Perte quotidienne > {mm_config.daily_loss_limit_pct}% (configurable)")
        print("   - Inventaire > limites configurées")
        print("   - Volatilité excessive détectée")
        
        if version == "V1.5":
            print(f"\n🔬 V1.5 Enhanced Features:")
            print(f"   - Depth Imbalance (DI) signal integration")
            print(f"   - Quote ageing with {mm_config.quote_ageing_ms}ms timeout")
            print(f"   - Dynamic spread adjustment")
            print(f"   - Enhanced risk controls")
        
        # Start all trading loops
        tasks = []
        for engine in engines:
            tasks.append(asyncio.create_task(engine.run_trading_loop()))
        
        # Wait for all tasks or timeout
        try:
            if duration_hours:
                print(f"⏱️  Running for {duration_hours} hours")
                await asyncio.wait_for(
                    asyncio.gather(*tasks),
                    timeout=duration_hours * 3600
                )
            else:
                await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n🛑 Graceful shutdown initiated...")
            # Cancel all running tasks
            for task in tasks:
                if not task.done():
                    task.cancel()
            
            # Wait for all tasks to complete cancellation
            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)
    
    except asyncio.TimeoutError:
        print("\n⏱️  Trading duration completed")
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    except Exception as e:
        print(f"\n❌ Error in trading loop: {e}")
        # Cancel remaining tasks
        for task in tasks:
            if not task.done():
                task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
    finally:
        # Stop all engines
        for engine in engines:
            try:
                await engine.stop()
            except Exception as e:
                print(f"⚠️ Error stopping engine {engine.symbol}: {e}")
        
        # Print final summary for each engine
        print("\n📊 Final Summary:")
        for engine in engines:
            engine.print_status()
        
        print("✅ Shutdown complete")


def run_backtest_mode():
    """Run backtesting mode"""
    print("🧪 Starting V1 Backtesting & Validation")
    print("=" * 60)
    
    try:
        results = run_simple_backtest()
        
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

            print(f"\n🎯 Calibration Results for {sym}")
            print("=" * 40)
            for param, value in results.items():
                print(f"  {param}: {value:.6f}")
            print("=" * 40)
            print(f"✅ Parameters updated and saved for {sym}")

        except Exception as e:
            print(f"❌ Calibration failed for {sym}: {e}")


async def run_live_trading(symbols: List[str], version: str = "V1-α"):
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
    await run_paper_trading(symbols, version=version)  # Uses same engine but with real orders


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Market Making Strategy with V1.5 Support')
    parser.add_argument('--mode', 
                       choices=['paper-trading', 'backtest', 'calibration', 'live'],
                       default='paper-trading',
                       help='Trading mode')
    parser.add_argument('--version',
                       choices=['V1-α', 'V1.5'],
                       default='V1-α',
                       help='Strategy version (V1-α or V1.5)')
    parser.add_argument('--symbol', 
                       default='BTCUSDT',
                       help='Symbol to trade (or "all" for all configured symbols)')
    parser.add_argument('--duration', 
                       type=int,
                       help='Duration in hours (for paper trading)')
    parser.add_argument('--log-level',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO',
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Determine symbols to trade
    if args.symbol.lower() == 'all':
        symbols = mm_config.symbols
    else:
        symbols = [args.symbol]
    
    print(f"🎯 Mode: {args.mode}")
    print(f"🔬 Version: {args.version}")
    print(f"📈 Symbols: {', '.join(symbols)}")
    print()
    
    # Auto-calibration before other modes (except calibration itself)
    if args.mode != 'calibration':
        print(f"\n🛠  Auto-calibration avant lancement du mode {args.mode}")
        run_calibration_mode(mm_config.symbols)

    try:
        if args.mode == 'paper-trading':
            asyncio.run(run_paper_trading(symbols, duration_hours=args.duration, version=args.version))
        elif args.mode == 'backtest':
            run_backtest_mode()
        elif args.mode == 'calibration':
            run_calibration_mode(symbols)
        elif args.mode == 'live':
            asyncio.run(run_live_trading(symbols, version=args.version))
    except KeyboardInterrupt:
        print("\n🛑 Program interrupted by user")
        print("✅ Shutdown complete")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
