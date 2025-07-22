"""
V1 Trading Engine - Integration Layer (§3.5)

This module implements the real-time operational loop:
1. Measure: Get market data (mid price, volatility, book imbalance)
2. Decide: Compute optimal quotes using A&S + OFI
3. Quote: Place/update orders (simulated)
4. Update: Track inventory, PnL, and KPIs

The engine integrates all V1 components and enforces risk controls.
"""

import asyncio
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import numpy as np

from .config import mm_config
from .avellaneda_stoikov import AvellanedaStoikovQuoter
from .ofi import OFICalculator
from .local_book import LocalBook
from .inventory_control import InventoryController
from .kpi_tracker import KPITracker


@dataclass
class Quote:
    """Represents an active quote in the market"""
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    timestamp: float
    quote_id: str
    filled: float = 0.0
    cancelled: bool = False


@dataclass
class Fill:
    """Represents a trade execution"""
    side: str  # 'bid' or 'ask'
    price: float
    size: float
    timestamp: float
    quote_id: str


class TradingEngine:
    """Main V1 trading engine integrating all components"""
    
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.logger = logging.getLogger(f"TradingEngine-{symbol}")
        
        # Core components
        self.quoter = AvellanedaStoikovQuoter(symbol)
        self.ofi_calc = OFICalculator(symbol)
        self.local_book = LocalBook(symbol)
        self.inventory_ctrl = InventoryController(symbol)
        self.kpi_tracker = KPITracker(symbol)
        
        # Trading state
        self.active_quotes: Dict[str, Quote] = {}
        self.current_mid = 0.0
        self.current_volatility = mm_config.sigma
        self.last_quote_time = 0.0
        self.quote_counter = 0
        
        # Risk flags
        self.trading_paused = False
        self.pause_reason = ""
        
        # Performance tracking
        self.total_quotes_sent = 0
        self.total_fills = 0
        self.session_start = time.time()
        
        # Control flag for trading loop
        self.running = False
        
    async def run_trading_loop(self):
        """Main entry point to run the trading loop"""
        self.logger.info(f"🚀 Starting trading loop for {self.symbol}")
        self.running = True
        
        try:
            # Initialize market data
            if not await self._initialize_market_data():
                self.logger.error("❌ Failed to initialize market data")
                return
            
            # Run main trading loop
            await self._main_trading_loop()
            
        except asyncio.CancelledError:
            self.logger.info("Trading loop cancelled")
        except Exception as e:
            self.logger.error(f"❌ Error in trading loop: {e}")
        finally:
            # Ensure we clean up on exit
            await self._cancel_all_quotes()
            self.logger.info("Trading loop stopped")
    
    async def stop(self):
        """Gracefully stop the trading engine"""
        self.logger.info("🛑 Stopping trading engine...")
        self.running = False
        
        # Cancel all active quotes
        await self._cancel_all_quotes()
        
        # Wait a bit to ensure all cancels are processed
        await asyncio.sleep(0.5)
        
        # Print final status
        self.print_status()
        self.logger.info("✅ Trading engine stopped")
    
    async def start(self):
        """Start the trading engine"""
        self.logger.info(f"🚀 Starting V1 Trading Engine for {self.symbol}")
        
        # Initialize market data
        if not await self._initialize_market_data():
            self.logger.error("❌ Failed to initialize market data")
            return
        
        # Start main trading loop
        await self._main_trading_loop()
    
    async def _initialize_market_data(self) -> bool:
        """Initialize connection to market data"""
        try:
            # Get initial snapshot
            if not self.local_book.get_snapshot():
                return False
            
            self.current_mid = self.local_book.get_mid_price()
            if not self.current_mid:
                return False
                
            self.logger.info(f"✅ Initialized: Mid=${self.current_mid:.2f}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Market data initialization failed: {e}")
            return False
    
    async def _main_trading_loop(self):
        """Main trading loop implementing §3.5"""
        self.logger.info("🔄 Starting main trading loop")
        
        while self.running:
            try:
                loop_start = time.time()
                
                # 1. MEASURE: Update market data
                await self._update_market_data()
                
                # 2. DECIDE: Check risk controls
                should_pause, reason = self._check_risk_controls()
                if should_pause:
                    if not self.trading_paused:
                        self.logger.warning(f"⏸️  Pausing trading: {reason}")
                        self.trading_paused = True
                        self.pause_reason = reason
                        await self._cancel_all_quotes()
                    await asyncio.sleep(1.0)  # Wait before retrying
                    continue
                else:
                    if self.trading_paused:
                        self.logger.info(f"▶️  Resuming trading")
                        self.trading_paused = False
                        self.pause_reason = ""
                
                # 3. DECIDE: Compute optimal quotes
                quotes_data = await self._compute_quotes()
                if not quotes_data:
                    await asyncio.sleep(0.1)
                    continue
                
                # 4. QUOTE: Update quotes in market
                await self._update_quotes(quotes_data)
                
                # 5. UPDATE: Process any fills and update metrics
                await self._process_market_updates()
                
                # Loop timing control
                loop_time = time.time() - loop_start
                self.kpi_tracker.record_latency('loop_time', loop_time * 1000)
                
                # Target ~100ms loop time
                if loop_time < 0.1:
                    await asyncio.sleep(0.1 - loop_time)
                    
            except Exception as e:
                self.logger.error(f"❌ Error in main loop: {e}")
                await asyncio.sleep(1.0)
    
    async def _update_market_data(self):
        """Update market data from local book"""
        # In a real implementation, this would process WebSocket updates
        # For now, we simulate market movement
        
        # Update mid price (simulate small random walk)
        if self.current_mid > 0:
            change_pct = np.random.normal(0, 0.0001)  # 0.01% std dev
            self.current_mid *= (1 + change_pct)
            
            # Update quoter's price history for volatility estimation
            self.quoter.update_volatility(self.current_mid)
            self.current_volatility = self.quoter.sigma
    
    def _check_risk_controls(self) -> Tuple[bool, str]:
        """Check all risk controls from §3.6"""
        
        # 1. Inventory limit
        should_pause, reason = self.inventory_ctrl.should_pause_trading()
        if should_pause:
            return True, reason
        
        # 2. Volatility spike check
        # Use configurable threshold from config instead of fixed 2×σ
        if self.current_volatility > mm_config.max_volatility_threshold:
            return True, (
                f"Volatility spike: {self.current_volatility:.4f} "
                f"> {mm_config.max_volatility_threshold:.4f}"
            )
        
        # 3. Latency check (simulated)
        avg_latency = self.kpi_tracker.get_average_latency('quote_ack')
        if avg_latency > 300:  # 300ms threshold
            return True, f"High latency: {avg_latency:.1f}ms > 300ms"
        
        return False, ""
    
    async def _compute_quotes(self) -> Optional[Dict]:
        """Compute optimal quotes using A&S + OFI"""
        if self.current_mid <= 0:
            return None
        
        try:
            # Get current OFI signal
            current_ofi = self.ofi_calc.current_ofi()
            
            # Get current inventory
            inventory = self.inventory_ctrl.current_inventory
            
            # Compute quotes
            quotes = self.quoter.compute_quotes(
                mid_price=self.current_mid,
                inventory=inventory,
                ofi=current_ofi
            )
            
            # Apply inventory skew
            skewed_quotes = self.inventory_ctrl.apply_skew_to_quotes(quotes)
            
            # Validate quotes
            if not self.quoter.validate_quotes(skewed_quotes, self.current_mid):
                self.logger.warning("⚠️  Invalid quotes generated")
                return None
            
            return skewed_quotes
            
        except Exception as e:
            self.logger.error(f"❌ Error computing quotes: {e}")
            return None
    
    async def _update_quotes(self, quotes_data: Dict):
        """Update quotes in the market (simulated)"""
        try:
            # Calculate optimal sizes
            bid_size = self.inventory_ctrl.calculate_optimal_size('bid', quotes_data['bid_price'])
            ask_size = self.inventory_ctrl.calculate_optimal_size('ask', quotes_data['ask_price'])
            
            current_time = time.time()
            
            # Generate quote IDs
            bid_id = f"bid_{self.quote_counter}"
            ask_id = f"ask_{self.quote_counter}"
            self.quote_counter += 1
            
            # Cancel existing quotes first
            await self._cancel_all_quotes()
            
            # Create new quotes
            bid_quote = Quote(
                side='bid',
                price=quotes_data['bid_price'],
                size=bid_size,
                timestamp=current_time,
                quote_id=bid_id
            )
            
            ask_quote = Quote(
                side='ask',
                price=quotes_data['ask_price'],
                size=ask_size,
                timestamp=current_time,
                quote_id=ask_id
            )
            
            # Store active quotes
            self.active_quotes[bid_id] = bid_quote
            self.active_quotes[ask_id] = ask_quote
            
            self.total_quotes_sent += 2
            self.last_quote_time = current_time
            
            # Record quotes sent in KPI tracker
            self.kpi_tracker.record_quotes_sent(2)
            
            # Simulate quote acknowledgment latency
            ack_latency = np.random.normal(50, 20)  # 50ms ± 20ms
            self.kpi_tracker.record_latency('quote_ack', max(10, ack_latency))
            
            # Log quote update
            self.logger.debug(f"📊 Quotes updated: "
                             f"Bid ${bid_quote.price:.2f}@{bid_quote.size:.4f} | "
                             f"Ask ${ask_quote.price:.2f}@{ask_quote.size:.4f} | "
                             f"OFI: {quotes_data.get('ofi', 0):.3f} | "
                             f"Shift: {quotes_data.get('center_shift', 0):.6f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating quotes: {e}")
    
    async def _cancel_all_quotes(self):
        """Cancel all active quotes"""
        if not self.active_quotes:
            return
        
        cancelled_count = 0
        for quote_id, quote in self.active_quotes.items():
            if not quote.cancelled:
                quote.cancelled = True
                cancelled_count += 1
        
        if cancelled_count > 0:
            self.kpi_tracker.record_cancel(cancelled_count)
            self.logger.debug(f"❌ Cancelled {cancelled_count} quotes")
        
        self.active_quotes.clear()
    
    async def _process_market_updates(self):
        """Process market updates and simulate fills"""
        # Simulate random fills based on market conditions
        for quote_id, quote in list(self.active_quotes.items()):
            if quote.cancelled or quote.filled >= quote.size:
                continue
            
            # Simple fill probability based on quote competitiveness
            # Better quotes (closer to mid) have higher fill probability
            if quote.side == 'bid':
                distance_from_mid = (self.current_mid - quote.price) / self.current_mid
            else:  # ask
                distance_from_mid = (quote.price - self.current_mid) / self.current_mid
            
            # Base fill probability decreases with distance from mid
            fill_prob = max(0.001, 0.02 - distance_from_mid * 100)  # 2% base, reduced by distance
            
            if np.random.random() < fill_prob:
                # Simulate partial fill
                fill_size = min(quote.size - quote.filled, quote.size * np.random.uniform(0.1, 1.0))
                
                fill = Fill(
                    side=quote.side,
                    price=quote.price,
                    size=fill_size,
                    timestamp=time.time(),
                    quote_id=quote_id
                )
                
                await self._process_fill(fill)
    
    async def _process_fill(self, fill: Fill):
        """Process a fill and update inventory/PnL"""
        try:
            # Update quote
            if fill.quote_id in self.active_quotes:
                self.active_quotes[fill.quote_id].filled += fill.size
            
            # Update inventory (bid = buy = positive, ask = sell = negative)
            inventory_change = fill.size if fill.side == 'bid' else -fill.size
            self.inventory_ctrl.update_inventory(inventory_change, fill.price)
            
            # Record fill for KPIs
            spread_at_fill = abs(fill.price - self.current_mid)
            self.kpi_tracker.record_fill(fill, self.current_mid, spread_at_fill)
            
            self.total_fills += 1
            
            # Log fill
            side_emoji = "🟢" if fill.side == 'bid' else "🔴"
            self.logger.info(f"{side_emoji} FILL: {fill.side.upper()} "
                           f"{fill.size:.4f} @ ${fill.price:.2f} | "
                           f"Inventory: {self.inventory_ctrl.current_inventory:+.4f}")
            
            # Add fill to OFI calculation
            self.ofi_calc.register_trade(fill.size, 'buy' if fill.side == 'ask' else 'sell')
            
        except Exception as e:
            self.logger.error(f"❌ Error processing fill: {e}")
    
    def get_status(self) -> Dict:
        """Get current trading status"""
        uptime = time.time() - self.session_start
        
        return {
            'symbol': self.symbol,
            'status': 'PAUSED' if self.trading_paused else 'ACTIVE',
            'pause_reason': self.pause_reason,
            'uptime_minutes': uptime / 60,
            'current_mid': self.current_mid,
            'inventory': self.inventory_ctrl.current_inventory,
            'total_quotes': self.total_quotes_sent,
            'total_fills': self.total_fills,
            'active_quotes': len([q for q in self.active_quotes.values() if not q.cancelled]),
            'current_volatility': self.current_volatility,
            'current_ofi': self.ofi_calc.current_ofi(),
            'kpis': self.kpi_tracker.get_summary()
        }
    
    def print_status(self):
        """Print current status"""
        status = self.get_status()
        kpis = status['kpis']
        
        print(f"\n📊 {self.symbol} Trading Engine Status")
        print("=" * 50)
        print(f"Status: {status['status']} | Uptime: {status['uptime_minutes']:.1f}min")
        if status['pause_reason']:
            print(f"Pause Reason: {status['pause_reason']}")
        print(f"Mid Price: ${status['current_mid']:.2f}")
        print(f"Inventory: {status['inventory']:+.4f}")
        print(f"Volatility: {status['current_volatility']:.4f}")
        print(f"OFI: {status['current_ofi']:+.3f}")
        print(f"Quotes Sent: {status['total_quotes']} | Fills: {status['total_fills']}")
        print(f"Active Quotes: {status['active_quotes']}")
        print("\n📈 KPIs:")
        print(f"  Fill Ratio: {kpis.get('fill_ratio', 0):.2%}")
        print(f"  Cancel Ratio: {kpis.get('cancel_ratio', 0):.2%}")
        print(f"  Avg Spread Captured: {kpis.get('avg_spread_captured', 0):.4f}")
        print(f"  RMS Inventory: {kpis.get('rms_inventory', 0):.4f}")
        print(f"  Total PnL: ${kpis.get('total_pnl', 0):+.2f}")
        print("=" * 50)


# Test function
async def test_trading_engine():
    """Test the trading engine"""
    engine = TradingEngine('BTCUSDT')
    
    # Run for 30 seconds
    print("🧪 Testing V1 Trading Engine for 30 seconds...")
    
    start_task = asyncio.create_task(engine.run_trading_loop())
    await asyncio.sleep(30)
    
    # Stop the engine
    await engine.stop()
    
    # Print final status
    engine.print_status()
    print("✅ Test completed")


if __name__ == "__main__":
    asyncio.run(test_trading_engine())
