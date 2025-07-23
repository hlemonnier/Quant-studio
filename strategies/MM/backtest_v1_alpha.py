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
            
            data.append({\n                'timestamp': tick_time,\n                'mid_price': current_price,\n                'bid_price': bid_price,\n                'ask_price': ask_price,\n                'spread': spread,\n                'spread_bps': (spread / current_price) * 10000,\n                'volume': volume,\n                'regime': regime_name,\n                'day': day + 1\n            })\n        \n        return data\n\n\nclass BacktestEngine:\n    \"\"\"\n    Main backtesting engine for V1-α validation\n    \n    Simulates complete trading loop:\n    1. Market data ingestion\n    2. Quote computation (A&S + OFI)\n    3. Order placement simulation\n    4. Fill simulation with realistic probabilities\n    5. Risk control validation\n    6. Performance measurement\n    \"\"\"\n    \n    def __init__(self, symbol: str, initial_capital: float = 100000.0):\n        self.symbol = symbol\n        self.initial_capital = initial_capital\n        self.logger = logging.getLogger(f\"BacktestEngine-{symbol}\")\n        \n        # Initialize components\n        self.quoter = AvellanedaStoikovQuoter(symbol)\n        self.kpi_tracker = KPITracker(symbol)\n        self.validator = PerformanceValidator(symbol)\n        self.inventory_ctrl = InventoryController(symbol, initial_capital)\n        \n        # Backtest state\n        self.current_time = None\n        self.active_quotes = {}\n        self.trade_history = []\n        self.quote_history = []\n        \n        # Performance tracking\n        self.daily_results = []\n        self.regime_results = {}\n        \n    async def run_7day_backtest(self, \n                               market_data: pd.DataFrame,\n                               save_results: bool = True) -> Dict:\n        \"\"\"\n        Run complete 7-day backtest as specified in §3.8\n        \n        Returns:\n            Comprehensive results dictionary with all metrics\n        \"\"\"\n        self.logger.info(f\"🚀 Starting 7-day backtest for {self.symbol}\")\n        start_time = time.time()\n        \n        # Reset state\n        self._reset_backtest_state()\n        \n        # Process each tick\n        total_ticks = len(market_data)\n        processed_ticks = 0\n        \n        for idx, row in market_data.iterrows():\n            await self._process_tick(row)\n            processed_ticks += 1\n            \n            # Progress reporting\n            if processed_ticks % 100000 == 0:\n                progress = (processed_ticks / total_ticks) * 100\n                self.logger.info(f\"📊 Progress: {progress:.1f}% ({processed_ticks:,}/{total_ticks:,})\")\n        \n        # Finalize results\n        results = self._compile_results(market_data)\n        \n        # Save results if requested\n        if save_results:\n            self._save_backtest_results(results)\n        \n        duration = time.time() - start_time\n        self.logger.info(f\"✅ Backtest completed in {duration:.1f}s\")\n        \n        return results\n    \n    def _reset_backtest_state(self):\n        \"\"\"Reset all state for fresh backtest\"\"\"\n        self.quoter = AvellanedaStoikovQuoter(self.symbol)\n        self.kpi_tracker = KPITracker(self.symbol)\n        self.validator = PerformanceValidator(self.symbol)\n        self.inventory_ctrl = InventoryController(self.symbol, self.initial_capital)\n        \n        self.active_quotes = {}\n        self.trade_history = []\n        self.quote_history = []\n        self.daily_results = []\n        self.regime_results = {}\n    \n    async def _process_tick(self, market_row: pd.Series):\n        \"\"\"Process single market data tick\"\"\"\n        self.current_time = market_row['timestamp']\n        mid_price = market_row['mid_price']\n        \n        # Update market data\n        self.quoter.update_volatility(mid_price)\n        self.kpi_tracker.update_mid_price(mid_price)\n        self.inventory_ctrl.update_mid_price(mid_price)\n        \n        # Check risk controls\n        should_pause, reason = self._check_risk_controls()\n        if should_pause:\n            self.logger.debug(f\"⏸️ Trading paused: {reason}\")\n            return\n        \n        # Compute optimal quotes\n        quotes = self._compute_quotes(mid_price)\n        if not quotes:\n            return\n        \n        # Simulate order placement and potential fills\n        await self._simulate_trading(quotes, market_row)\n        \n        # Record quote for analysis\n        self.quote_history.append({\n            'timestamp': self.current_time,\n            'mid_price': mid_price,\n            'bid_quote': quotes['bid_price'],\n            'ask_quote': quotes['ask_price'],\n            'spread': quotes['spread'],\n            'inventory': self.inventory_ctrl.current_inventory,\n            'regime': market_row.get('regime', 'unknown')\n        })\n    \n    def _check_risk_controls(self) -> Tuple[bool, str]:\n        \"\"\"Check all risk controls (§3.6)\"\"\"\n        # Inventory limits\n        should_pause, reason = self.inventory_ctrl.should_pause_trading()\n        if should_pause:\n            return True, reason\n        \n        # Volatility spike\n        if self.quoter.sigma > mm_config.max_volatility_threshold:\n            return True, f\"Volatility spike: {self.quoter.sigma:.4f}\"\n        \n        return False, \"\"\n    \n    def _compute_quotes(self, mid_price: float) -> Optional[Dict]:\n        \"\"\"Compute optimal quotes using A&S model\"\"\"\n        inventory = self.inventory_ctrl.current_inventory\n        \n        # Compute base quotes\n        quotes = self.quoter.compute_quotes(\n            mid_price=mid_price,\n            inventory=inventory,\n            time_remaining=None,  # Use default T\n            ofi=0.0  # Simplified for backtest\n        )\n        \n        if not quotes:\n            return None\n        \n        # Validate quotes\n        if not self.quoter.validate_quotes(quotes, mid_price):\n            return None\n        \n        return quotes\n    \n    async def _simulate_trading(self, quotes: Dict, market_row: pd.Series):\n        \"\"\"Simulate order placement and fills\"\"\"\n        mid_price = market_row['mid_price']\n        market_bid = market_row['bid_price']\n        market_ask = market_row['ask_price']\n        \n        # Simulate bid fill probability\n        bid_quote = quotes['bid_price']\n        if bid_quote >= market_bid:\n            # Our bid is at or above market bid - high fill probability\n            fill_prob = min(0.9, (bid_quote - market_bid) / mid_price * 1000 + 0.3)\n        else:\n            # Our bid is below market - lower fill probability\n            fill_prob = max(0.01, 0.1 - (market_bid - bid_quote) / mid_price * 500)\n        \n        if np.random.random() < fill_prob:\n            await self._execute_fill('bid', bid_quote, quotes['size'], mid_price)\n        \n        # Simulate ask fill probability\n        ask_quote = quotes['ask_price']\n        if ask_quote <= market_ask:\n            # Our ask is at or below market ask - high fill probability\n            fill_prob = min(0.9, (market_ask - ask_quote) / mid_price * 1000 + 0.3)\n        else:\n            # Our ask is above market - lower fill probability\n            fill_prob = max(0.01, 0.1 - (ask_quote - market_ask) / mid_price * 500)\n        \n        if np.random.random() < fill_prob:\n            await self._execute_fill('ask', ask_quote, quotes['size'], mid_price)\n    \n    async def _execute_fill(self, side: str, price: float, size: float, mid_price: float):\n        \"\"\"Execute a simulated fill\"\"\"\n        # Determine trade direction and size\n        if side == 'bid':\n            trade_size = size  # We buy (positive inventory)\n        else:\n            trade_size = -size  # We sell (negative inventory)\n        \n        # Create fill record\n        fill = {\n            'timestamp': self.current_time,\n            'side': side,\n            'price': price,\n            'size': abs(trade_size),\n            'trade_size': trade_size,  # Signed size\n            'mid_price_at_fill': mid_price\n        }\n        \n        # Update inventory\n        self.inventory_ctrl.update_inventory(trade_size, price, mid_price)\n        \n        # Record fill in KPI tracker\n        spread_captured = abs(price - mid_price)\n        self.kpi_tracker.record_fill(fill, mid_price, spread_captured)\n        \n        # Add to trade history\n        self.trade_history.append(fill)\n        \n        self.logger.debug(\n            f\"💰 Fill: {side} {abs(trade_size):.4f} @ ${price:.2f} \"\n            f\"(mid: ${mid_price:.2f}, spread: {spread_captured:.2f})\"\n        )\n    \n    def _compile_results(self, market_data: pd.DataFrame) -> Dict:\n        \"\"\"Compile comprehensive backtest results\"\"\"\n        # Get final performance summary\n        kpi_summary = self.kpi_tracker.get_summary()\n        validation_result = self.validator.validate_performance(self.kpi_tracker)\n        \n        # Analyze by regime\n        regime_analysis = self._analyze_by_regime(market_data)\n        \n        # Daily breakdown\n        daily_analysis = self._analyze_by_day(market_data)\n        \n        # Compile comprehensive results\n        results = {\n            'metadata': {\n                'symbol': self.symbol,\n                'backtest_start': market_data.iloc[0]['timestamp'],\n                'backtest_end': market_data.iloc[-1]['timestamp'],\n                'total_ticks': len(market_data),\n                'total_trades': len(self.trade_history),\n                'total_quotes': len(self.quote_history),\n                'initial_capital': self.initial_capital\n            },\n            'performance': {\n                'kpi_summary': kpi_summary,\n                'validation': validation_result,\n                'final_inventory': self.inventory_ctrl.current_inventory,\n                'final_capital': self.inventory_ctrl.current_capital,\n                'total_return_pct': ((self.inventory_ctrl.current_capital - self.initial_capital) / self.initial_capital) * 100\n            },\n            'regime_analysis': regime_analysis,\n            'daily_analysis': daily_analysis,\n            'trade_history': self.trade_history[-100:],  # Last 100 trades\n            'config_used': {\n                'gamma': mm_config.gamma,\n                'sigma_initial': mm_config.sigma,\n                'k': mm_config.k,\n                'max_inventory': mm_config.max_inventory,\n                'targets': {\n                    'spread_capture': mm_config.target_spread_capture_pct,\n                    'rms_inventory': mm_config.target_rms_inventory_ratio,\n                    'fill_ratio': mm_config.target_fill_ratio_pct,\n                    'cancel_ratio': mm_config.target_cancel_ratio_pct\n                }\n            }\n        }\n        \n        return results\n    \n    def _analyze_by_regime(self, market_data: pd.DataFrame) -> Dict:\n        \"\"\"Analyze performance by market regime\"\"\"\n        regime_stats = {}\n        \n        for regime in market_data['regime'].unique():\n            regime_data = market_data[market_data['regime'] == regime]\n            regime_trades = [\n                t for t in self.trade_history \n                if any(r['timestamp'] <= t['timestamp'] < r['timestamp'] + timedelta(milliseconds=100) \n                      for _, r in regime_data.iterrows())\n            ]\n            \n            if regime_trades:\n                total_pnl = sum(\n                    (t['mid_price_at_fill'] - t['price']) * t['trade_size'] \n                    for t in regime_trades\n                )\n                avg_spread = np.mean([\n                    abs(t['price'] - t['mid_price_at_fill']) \n                    for t in regime_trades\n                ])\n            else:\n                total_pnl = 0.0\n                avg_spread = 0.0\n            \n            regime_stats[regime] = {\n                'duration_hours': len(regime_data) / 36000,  # 100ms ticks\n                'trades': len(regime_trades),\n                'pnl': total_pnl,\n                'avg_spread_captured': avg_spread,\n                'avg_volatility': regime_data['spread_bps'].mean() if 'spread_bps' in regime_data else 0\n            }\n        \n        return regime_stats\n    \n    def _analyze_by_day(self, market_data: pd.DataFrame) -> Dict:\n        \"\"\"Analyze performance by day\"\"\"\n        daily_stats = {}\n        \n        for day in range(1, 8):  # Days 1-7\n            day_data = market_data[market_data['day'] == day]\n            day_start = day_data.iloc[0]['timestamp']\n            day_end = day_data.iloc[-1]['timestamp']\n            \n            day_trades = [\n                t for t in self.trade_history \n                if day_start <= t['timestamp'] <= day_end\n            ]\n            \n            if day_trades:\n                total_pnl = sum(\n                    (t['mid_price_at_fill'] - t['price']) * t['trade_size'] \n                    for t in day_trades\n                )\n            else:\n                total_pnl = 0.0\n            \n            daily_stats[f'day_{day}'] = {\n                'date': day_start.date(),\n                'regime': day_data.iloc[0]['regime'],\n                'trades': len(day_trades),\n                'pnl': total_pnl,\n                'price_range': {\n                    'low': day_data['mid_price'].min(),\n                    'high': day_data['mid_price'].max(),\n                    'start': day_data.iloc[0]['mid_price'],\n                    'end': day_data.iloc[-1]['mid_price']\n                }\n            }\n        \n        return daily_stats\n    \n    def _save_backtest_results(self, results: Dict):\n        \"\"\"Save backtest results to file\"\"\"\n        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')\n        filename = f\"backtest_v1alpha_{self.symbol}_{timestamp}.json\"\n        \n        # Create results directory\n        results_dir = Path('data/backtest_results')\n        results_dir.mkdir(parents=True, exist_ok=True)\n        \n        filepath = results_dir / filename\n        \n        # Convert datetime objects to strings for JSON serialization\n        def json_serializer(obj):\n            if isinstance(obj, datetime):\n                return obj.isoformat()\n            raise TypeError(f\"Object of type {type(obj)} is not JSON serializable\")\n        \n        with open(filepath, 'w') as f:\n            json.dump(results, f, indent=2, default=json_serializer)\n        \n        self.logger.info(f\"💾 Results saved to {filepath}\")\n    \n    def print_backtest_summary(self, results: Dict):\n        \"\"\"Print comprehensive backtest summary\"\"\"\n        print(f\"\\n🎯 V1-α 7-Day Backtest Summary\")\n        print(\"=\" * 60)\n        \n        metadata = results['metadata']\n        performance = results['performance']\n        \n        print(f\"Symbol: {metadata['symbol']}\")\n        print(f\"Period: {metadata['backtest_start'].strftime('%Y-%m-%d')} to {metadata['backtest_end'].strftime('%Y-%m-%d')}\")\n        print(f\"Total Ticks: {metadata['total_ticks']:,}\")\n        print(f\"Total Trades: {metadata['total_trades']:,}\")\n        \n        # Performance metrics\n        kpi = performance['kpi_summary']\n        validation = performance['validation']\n        \n        print(f\"\\n📊 Performance Metrics:\")\n        print(f\"  Fill Ratio: {kpi['fill_ratio']:.2%}\")\n        print(f\"  Cancel Ratio: {kpi['cancel_ratio']:.2%}\")\n        print(f\"  Spread Captured: {kpi['spread_captured_pct']:.1f}%\")\n        print(f\"  RMS Inventory: {kpi['rms_inventory']:.4f}\")\n        print(f\"  Total PnL: ${kpi['total_pnl']:+.2f}\")\n        print(f\"  Total Return: {performance['total_return_pct']:+.2f}%\")\n        \n        # Validation status\n        print(f\"\\n✅ V1-α Compliance: {validation['status']} ({validation['compliance_pct']:.0f}%)\")\n        print(f\"  Targets Met: {validation['targets_met']}/{validation['total_targets']}\")\n        \n        # Regime analysis\n        print(f\"\\n🌊 Performance by Market Regime:\")\n        for regime, stats in results['regime_analysis'].items():\n            print(f\"  {regime.title()}: {stats['trades']} trades, PnL ${stats['pnl']:+.2f}\")\n        \n        # Daily breakdown\n        print(f\"\\n📅 Daily Performance:\")\n        for day_key, stats in results['daily_analysis'].items():\n            day_num = day_key.split('_')[1]\n            print(f\"  Day {day_num} ({stats['regime']}): {stats['trades']} trades, PnL ${stats['pnl']:+.2f}\")\n        \n        print(\"=\" * 60)\n\n\nasync def run_v1_alpha_validation():\n    \"\"\"\n    Run complete V1-α validation suite\n    \n    This is the main entry point for V1-α backtesting validation.\n    \"\"\"\n    print(\"🚀 Starting V1-α Validation Suite\")\n    print(\"=\" * 50)\n    \n    # Test parameters\n    symbol = 'BTCUSDT'\n    initial_price = 50000.0\n    initial_capital = 100000.0\n    \n    # Generate 7-day market data\n    print(\"📊 Generating 7-day market data...\")\n    simulator = MarketDataSimulator(symbol, initial_price)\n    market_data = simulator.generate_7day_data()\n    \n    # Run backtest\n    print(\"🔄 Running backtest...\")\n    engine = BacktestEngine(symbol, initial_capital)\n    results = await engine.run_7day_backtest(market_data)\n    \n    # Print results\n    engine.print_backtest_summary(results)\n    \n    # Validation report\n    print(\"\\n🔍 Detailed Validation Report:\")\n    engine.validator.print_validation_report(engine.kpi_tracker)\n    \n    return results\n\n\nif __name__ == \"__main__\":\n    # Run the validation\n    results = asyncio.run(run_v1_alpha_validation())\n    print(\"\\n✅ V1-α validation completed!\")\n

