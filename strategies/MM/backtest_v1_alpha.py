"""
V1-α Systematic Backtesting Framework (§3.8)

Implements comprehensive 7-day backtesting as specified in V1-α requirements:
- Historical data simulation with realistic market microstructure
- Complete trading loop validation
- Performance target verification
- Stress testing scenarios
- Detailed compliance reporting

This is the final validation step before live deployment.
"""

import asyncio
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
import logging
import time
import json
from pathlib import Path

from strategies.MM.config import MMConfig
from strategies.MM.avellaneda_stoikov import AvellanedaStoikovQuoter
from strategies.MM.kpi_tracker import KPITracker
from strategies.MM.performance_validator import PerformanceValidator
from strategies.MM.inventory_control import InventoryController

mm_config = MMConfig()


class MarketDataSimulator:
    """
    Simulates realistic market data for backtesting
    
    Features:
    - Realistic price dynamics with volatility clustering
    - Order book depth simulation
    - Trade flow simulation with size distribution
    - Market regime changes (normal, volatile, trending)
    """
    
    def __init__(self, symbol: str, initial_price: float = 50000.0):
        self.symbol = symbol
        self.initial_price = initial_price
        self.logger = logging.getLogger(f"MarketSim-{symbol}")
        
        # Market microstructure parameters
        self.tick_size = 0.01 if symbol == 'BTCUSDT' else 0.001
        self.min_spread_ticks = 1
        self.max_spread_ticks = 10
        
        # Volatility regimes
        self.regimes = {
            'normal': {'vol': 0.6, 'trend': 0.0, 'jump_prob': 0.001},
            'volatile': {'vol': 1.2, 'trend': 0.0, 'jump_prob': 0.005},
            'trending_up': {'vol': 0.8, 'trend': 0.2, 'jump_prob': 0.002},
            'trending_down': {'vol': 0.8, 'trend': -0.2, 'jump_prob': 0.002}
        }
        
    def generate_7day_data(self, regime_schedule: Optional[Dict] = None) -> pd.DataFrame:
        """
        Generate 7 days of market data (§3.8 requirement)
        
        Args:
            regime_schedule: Dict mapping day -> regime name
        
        Returns:
            DataFrame with columns: timestamp, mid_price, bid, ask, volume, regime
        """
        if regime_schedule is None:
            # Default: mix of regimes over 7 days
            regime_schedule = {
                0: 'normal',      # Day 1: Normal conditions
                1: 'normal',      # Day 2: Normal conditions  
                2: 'volatile',    # Day 3: High volatility
                3: 'trending_up', # Day 4: Uptrend
                4: 'trending_down', # Day 5: Downtrend
                5: 'volatile',    # Day 6: High volatility
                6: 'normal'       # Day 7: Normal conditions
            }
        
        all_data = []
        current_price = self.initial_price
        
        for day in range(7):
            regime_name = regime_schedule.get(day, 'normal')
            regime = self.regimes[regime_name]
            
            self.logger.info(f"📅 Day {day+1}: {regime_name} regime")
            
            # Generate one day of data (100ms ticks)
            day_data = self._generate_day_data(
                current_price, regime, day, regime_name
            )
            
            all_data.extend(day_data)
            current_price = day_data[-1]['mid_price']  # Continue from last price
        
        df = pd.DataFrame(all_data)
        self.logger.info(f"✅ Generated {len(df)} ticks over 7 days")
        return df
    
    def _generate_day_data(self, start_price: float, regime: Dict, 
                          day: int, regime_name: str) -> List[Dict]:
        """Generate one day of market data"""
        ticks_per_day = 24 * 60 * 60 * 10  # 100ms ticks = 864,000 per day
        
        # Time parameters
        dt = 1.0 / (365 * 24 * 60 * 60 * 10)  # 100ms in year fraction
        
        # Price process parameters
        vol = regime['vol']
        trend = regime['trend']
        jump_prob = regime['jump_prob']
        
        data = []
        current_price = start_price
        base_time = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=day)
        
        for tick in range(ticks_per_day):
            # Generate price movement
            # 1. Normal diffusion
            normal_return = trend * dt + vol * np.sqrt(dt) * np.random.normal()
            
            # 2. Jump component
            if np.random.random() < jump_prob:
                jump_size = np.random.normal(0, 0.02)  # 2% jump std
                normal_return += jump_size
            
            # Update price
            current_price *= (1 + normal_return)
            
            # Generate bid-ask spread (realistic microstructure)
            spread_ticks = np.random.randint(
                self.min_spread_ticks, self.max_spread_ticks + 1
            )
            spread = spread_ticks * self.tick_size
            
            bid_price = current_price - spread / 2
            ask_price = current_price + spread / 2
            
            # Generate volume (log-normal distribution)
            volume = np.random.lognormal(mean=2.0, sigma=1.0)
            
            # Create tick data
            tick_time = base_time + timedelta(milliseconds=tick * 100)
            
            data.append({
                'timestamp': tick_time,
                'mid_price': current_price,
                'bid_price': bid_price,
                'ask_price': ask_price,
                'spread': spread,
                'spread_bps': (spread / current_price) * 10000,
                'volume': volume,
                'regime': regime_name,
                'day': day + 1
            })

        return data


class BacktestEngine:
    """
    Main backtesting engine for V1-α validation
    
    Simulates complete trading loop:
    1. Market data ingestion
    2. Quote computation (A&S + OFI)
    3. Order placement simulation
    4. Fill simulation with realistic probabilities
    5. Risk control validation
    6. Performance measurement
    """
    
    def __init__(self, symbol: str, initial_capital: float = 100000.0):
        self.symbol = symbol
        self.initial_capital = initial_capital
        self.logger = logging.getLogger(f"BacktestEngine-{symbol}")
        
        # Initialize components
        self.quoter = AvellanedaStoikovQuoter(symbol)
        self.kpi_tracker = KPITracker(symbol)
        self.validator = PerformanceValidator(symbol)
        self.inventory_ctrl = InventoryController(symbol, initial_capital)
        
        # Backtest state
        self.current_time = None
        self.active_quotes = {}
        self.trade_history = []
        self.quote_history = []
        
        # Performance tracking
        self.daily_results = []
        self.regime_results = {}
        
    async def run_7day_backtest(self, 
                               market_data: pd.DataFrame,
                               save_results: bool = True) -> Dict:
        """
        Run complete 7-day backtest as specified in §3.8
        
        Returns:
            Comprehensive results dictionary with all metrics
        """
        self.logger.info(f"🚀 Starting 7-day backtest for {self.symbol}")
        start_time = time.time()
        
        # Reset state
        self._reset_backtest_state()
        
        # Process each tick
        total_ticks = len(market_data)
        processed_ticks = 0
        
        for idx, row in market_data.iterrows():
            await self._process_tick(row)
            processed_ticks += 1
            
            # Progress reporting
            if processed_ticks % 100000 == 0:
                progress = (processed_ticks / total_ticks) * 100
                self.logger.info(f"📊 Progress: {progress:.1f}% ({processed_ticks:,}/{total_ticks:,})")
        
        # Finalize results
        results = self._compile_results(market_data)
        
        # Save results if requested
        if save_results:
            self._save_backtest_results(results)
        
        duration = time.time() - start_time
        self.logger.info(f"✅ Backtest completed in {duration:.1f}s")
        
        return results
    
    def _reset_backtest_state(self):
        """Reset all state for fresh backtest"""
        self.quoter = AvellanedaStoikovQuoter(self.symbol)
        self.kpi_tracker = KPITracker(self.symbol)
        self.validator = PerformanceValidator(self.symbol)
        self.inventory_ctrl = InventoryController(self.symbol, self.initial_capital)
        
        self.active_quotes = {}
        self.trade_history = []
        self.quote_history = []
        self.daily_results = []
        self.regime_results = {}
    
    async def _process_tick(self, market_row: pd.Series):
        """Process single market data tick"""
        self.current_time = market_row['timestamp']
        mid_price = market_row['mid_price']
        
        # Update market data
        self.quoter.update_volatility(mid_price)
        self.kpi_tracker.update_mid_price(mid_price)
        self.inventory_ctrl.update_mid_price(mid_price)
        
        # Check risk controls
        should_pause, reason = self._check_risk_controls()
        if should_pause:
            self.logger.debug(f"⏸️ Trading paused: {reason}")
            return
        
        # Compute optimal quotes
        quotes = self._compute_quotes(mid_price)
        if not quotes:
            return
        
        # Simulate order placement and potential fills
        await self._simulate_trading(quotes, market_row)
        
        # Record quote for analysis
        self.quote_history.append({
            'timestamp': self.current_time,
            'mid_price': mid_price,
            'bid_quote': quotes['bid_price'],
            'ask_quote': quotes['ask_price'],
            'spread': quotes['spread'],
            'inventory': self.inventory_ctrl.current_inventory,
            'regime': market_row.get('regime', 'unknown')
        })
    
    def _check_risk_controls(self) -> Tuple[bool, str]:
        """Check all risk controls (§3.6)"""
        # Inventory limits
        should_pause, reason = self.inventory_ctrl.should_pause_trading()
        if should_pause:
            return True, reason
        
        # Volatility spike
        if self.quoter.sigma > mm_config.max_volatility_threshold:
            return True, f"Volatility spike: {self.quoter.sigma:.4f}"
        
        return False, ""
    
    def _compute_quotes(self, mid_price: float) -> Optional[Dict]:
        """Compute optimal quotes using A&S model"""
        inventory = self.inventory_ctrl.current_inventory
        
        # Compute base quotes
        quotes = self.quoter.compute_quotes(
            mid_price=mid_price,
            inventory=inventory,
            time_remaining=None,  # Use default T
            ofi=0.0  # Simplified for backtest
        )
        
        if not quotes:
            return None
        
        # Validate quotes
        if not self.quoter.validate_quotes(quotes, mid_price):
            return None
        
        return quotes
    
    async def _simulate_trading(self, quotes: Dict, market_row: pd.Series):
        """Simulate order placement and fills"""
        mid_price = market_row['mid_price']
        market_bid = market_row['bid_price']
        market_ask = market_row['ask_price']
        
        # Simulate bid fill probability
        bid_quote = quotes['bid_price']
        if bid_quote >= market_bid:
            # Our bid is at or above market bid - high fill probability
            fill_prob = min(0.9, (bid_quote - market_bid) / mid_price * 1000 + 0.3)
        else:
            # Our bid is below market - lower fill probability
            fill_prob = max(0.01, 0.1 - (market_bid - bid_quote) / mid_price * 500)
        
        if np.random.random() < fill_prob:
            await self._execute_fill('bid', bid_quote, quotes['size'], mid_price)
        
        # Simulate ask fill probability
        ask_quote = quotes['ask_price']
        if ask_quote <= market_ask:
            # Our ask is at or below market ask - high fill probability
            fill_prob = min(0.9, (market_ask - ask_quote) / mid_price * 1000 + 0.3)
        else:
            # Our ask is above market - lower fill probability
            fill_prob = max(0.01, 0.1 - (ask_quote - market_ask) / mid_price * 500)
        
        if np.random.random() < fill_prob:
            await self._execute_fill('ask', ask_quote, quotes['size'], mid_price)
    
    async def _execute_fill(self, side: str, price: float, size: float, mid_price: float):
        """Execute a simulated fill"""
        # Determine trade direction and size
        if side == 'bid':
            trade_size = size  # We buy (positive inventory)
        else:
            trade_size = -size  # We sell (negative inventory)
        
        # Create fill record
        fill = {
            'timestamp': self.current_time,
            'side': side,
            'price': price,
            'size': abs(trade_size),
            'trade_size': trade_size,  # Signed size
            'mid_price_at_fill': mid_price
        }
        
        # Update inventory
        self.inventory_ctrl.update_inventory(trade_size, price, mid_price)
        
        # Record fill in KPI tracker
        spread_captured = abs(price - mid_price)
        self.kpi_tracker.record_fill(fill, mid_price, spread_captured)
        
        # Add to trade history
        self.trade_history.append(fill)
        
        self.logger.debug(
            f"💰 Fill: {side} {abs(trade_size):.4f} @ ${price:.2f} "
            f"(mid: ${mid_price:.2f}, spread: {spread_captured:.2f})"
        )
    
    def _compile_results(self, market_data: pd.DataFrame) -> Dict:
        """Compile comprehensive backtest results"""
        # Get final performance summary
        kpi_summary = self.kpi_tracker.get_summary()
        validation_result = self.validator.validate_performance(self.kpi_tracker)
        
        # Analyze by regime
        regime_analysis = self._analyze_by_regime(market_data)
        
        # Daily breakdown
        daily_analysis = self._analyze_by_day(market_data)
        
        # Compile comprehensive results
        results = {
            'metadata': {
                'symbol': self.symbol,
                'backtest_start': market_data.iloc[0]['timestamp'],
                'backtest_end': market_data.iloc[-1]['timestamp'],
                'total_ticks': len(market_data),
                'total_trades': len(self.trade_history),
                'total_quotes': len(self.quote_history),
                'initial_capital': self.initial_capital
            },
            'performance': {
                'kpi_summary': kpi_summary,
                'validation': validation_result,
                'final_inventory': self.inventory_ctrl.current_inventory,
                'final_capital': self.inventory_ctrl.current_capital,
                'total_return_pct': ((self.inventory_ctrl.current_capital - self.initial_capital) / self.initial_capital) * 100
            },
            'regime_analysis': regime_analysis,
            'daily_analysis': daily_analysis,
            'trade_history': self.trade_history[-100:],  # Last 100 trades
            'config_used': {
                'gamma': mm_config.gamma,
                'sigma_initial': mm_config.sigma,
                'k': mm_config.k,
                'max_inventory': mm_config.max_inventory,
                'targets': {
                    'spread_capture': mm_config.target_spread_capture_pct,
                    'rms_inventory': mm_config.target_rms_inventory_ratio,
                    'fill_ratio': mm_config.target_fill_ratio_pct,
                    'cancel_ratio': mm_config.target_cancel_ratio_pct
                }
            }
        }
        
        return results
    
    def _analyze_by_regime(self, market_data: pd.DataFrame) -> Dict:
        """Analyze performance by market regime"""
        regime_stats = {}
        
        for regime in market_data['regime'].unique():
            regime_data = market_data[market_data['regime'] == regime]
            regime_trades = [
                t for t in self.trade_history 
                if any(r['timestamp'] <= t['timestamp'] < r['timestamp'] + timedelta(milliseconds=100) 
                      for _, r in regime_data.iterrows())
            ]
            
            if regime_trades:
                total_pnl = sum(
                    (t['mid_price_at_fill'] - t['price']) * t['trade_size'] 
                    for t in regime_trades
                )
                avg_spread = np.mean([
                    abs(t['price'] - t['mid_price_at_fill']) 
                    for t in regime_trades
                ])
            else:
                total_pnl = 0.0
                avg_spread = 0.0
            
            regime_stats[regime] = {
                'duration_hours': len(regime_data) / 36000,  # 100ms ticks
                'trades': len(regime_trades),
                'pnl': total_pnl,
                'avg_spread_captured': avg_spread,
                'avg_volatility': regime_data['spread_bps'].mean() if 'spread_bps' in regime_data else 0
            }
        return regime_stats
    
    def _analyze_by_day(self, market_data: pd.DataFrame) -> Dict:
        """Analyze performance by day"""
        daily_stats = {}
        
        for day in range(1, 8):  # Days 1-7
            day_data = market_data[market_data['day'] == day]
            day_start = day_data.iloc[0]['timestamp']
            day_end = day_data.iloc[-1]['timestamp']
            
            day_trades = [
                t for t in self.trade_history 
                if day_start <= t['timestamp'] <= day_end
            ]
            
            if day_trades:
                total_pnl = sum(
                    (t['mid_price_at_fill'] - t['price']) * t['trade_size'] 
                    for t in day_trades
                )
            else:
                total_pnl = 0.0
            
            daily_stats[f'day_{day}'] = {
                'date': day_start.date(),
                'regime': day_data.iloc[0]['regime'],
                'trades': len(day_trades),
                'pnl': total_pnl,
                'price_range': {
                    'low': day_data['mid_price'].min(),
                    'high': day_data['mid_price'].max(),
                    'start': day_data.iloc[0]['mid_price'],
                    'end': day_data.iloc[-1]['mid_price']
                }
            }
        return daily_stats
    
    def _save_backtest_results(self, results: Dict):
        """Save backtest results to file"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"backtest_v1alpha_{self.symbol}_{timestamp}.json"
        
        # Create results directory
        results_dir = Path('data/backtest_results')
        results_dir.mkdir(parents=True, exist_ok=True)
        
        filepath = results_dir / filename
        
        # Convert datetime objects to strings for JSON serialization
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            raise TypeError(f"Object of type {type(obj)} is not JSON serializable")
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2, default=json_serializer)
        
        self.logger.info(f"💾 Results saved to {filepath}")
    
    def print_backtest_summary(self, results: Dict):
        """Print comprehensive backtest summary"""
        print(f"\n🎯 V1-α 7-Day Backtest Summary")
        print("=" * 60)
        
        metadata = results['metadata']
        performance = results['performance']
        
        print(f"Symbol: {metadata['symbol']}")
        print(f"Period: {metadata['backtest_start'].strftime('%Y-%m-%d')} to {metadata['backtest_end'].strftime('%Y-%m-%d')}")
        print(f"Total Ticks: {metadata['total_ticks']:,}")
        print(f"Total Trades: {metadata['total_trades']:,}")
        
        # Performance metrics
        kpi = performance['kpi_summary']
        validation = performance['validation']
        
        print(f"\n📊 Performance Metrics:")
        print(f"  Fill Ratio: {kpi['fill_ratio']:.2%}")
        print(f"  Cancel Ratio: {kpi['cancel_ratio']:.2%}")
        print(f"  Spread Captured: {kpi['spread_captured_pct']:.1f}%")
        print(f"  RMS Inventory: {kpi['rms_inventory']:.4f}")
        print(f"  Total PnL: ${kpi['total_pnl']:+.2f}")
        print(f"  Total Return: {performance['total_return_pct']:+.2f}%")
        
        # Validation status
        print(f"\n✅ V1-α Compliance: {validation['status']} ({validation['compliance_pct']:.0f}%)")
        print(f"  Targets Met: {validation['targets_met']}/{validation['total_targets']}")
        
        # Regime analysis
        print(f"\n🌊 Performance by Market Regime:")
        for regime, stats in results['regime_analysis'].items():
            print(f"  {regime.title()}: {stats['trades']} trades, PnL ${stats['pnl']:+.2f}")
        
        # Daily breakdown
        print(f"\n📅 Daily Performance:")
        for day_key, stats in results['daily_analysis'].items():
            day_num = day_key.split('_')[1]
            print(f"  Day {day_num} ({stats['regime']}): {stats['trades']} trades, PnL ${stats['pnl']:+.2f}")
        
        print("=" * 60)


async def run_v1_alpha_validation():
    """
    Run complete V1-α validation suite
    
    This is the main entry point for V1-α backtesting validation.
    """
    print("🚀 Starting V1-α Validation Suite")
    print("=" * 50)
    
    # Test parameters
    symbol = 'BTCUSDT'
    initial_price = 50000.0
    initial_capital = 100000.0
    
    # Generate 7-day market data
    print("📊 Generating 7-day market data...")
    simulator = MarketDataSimulator(symbol, initial_price)
    market_data = simulator.generate_7day_data()
    
    # Run backtest
    print("🔄 Running backtest...")
    engine = BacktestEngine(symbol, initial_capital)
    results = await engine.run_7day_backtest(market_data)
    
    # Print results
    engine.print_backtest_summary(results)
    
    # Validation report
    print("\n🔍 Detailed Validation Report:")
    engine.validator.print_validation_report(engine.kpi_tracker)
    
    return results


if __name__ == "__main__":
    # Run the validation
    results = asyncio.run(run_v1_alpha_validation())
    print("\n✅ V1-α validation completed!")

