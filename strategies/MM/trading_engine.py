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

from .utils.config import mm_config
from .market_making.avellaneda_stoikov import AvellanedaStoikovQuoter
from .market_making.avellaneda_stoikov_v15 import AvellanedaStoikovV15Quoter
from .core.ofi import OFICalculator
from .core.depth_imbalance import DepthImbalanceCalculator
from .market_making.quote_manager import QuoteManager
from .data_capture.local_book import LocalBook
from .utils.inventory_control import InventoryController
from .utils.kpi_tracker import KPITracker
from .data_capture.websocket_manager import TradingEngineWSIntegration


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
    """Main trading engine supporting both V1-α and V1.5"""
    
    def __init__(self, symbol: str, version: str = "V1-α"):
        self.symbol = symbol
        self.version = version
        self.logger = logging.getLogger(f"TradingEngine-{symbol}-{version}")
        
        # Core components (common to both versions)
        self.ofi_calc = OFICalculator(symbol)
        self.local_book = LocalBook(symbol)
        self.inventory_ctrl = InventoryController(symbol)
        self.kpi_tracker = KPITracker(symbol)
        
        # Version-specific components
        if version == "V1.5":
            self.quoter = AvellanedaStoikovV15Quoter(symbol)
            self.di_calc = DepthImbalanceCalculator(symbol)
            self.quote_manager = QuoteManager(symbol)
            self.logger.warning("🚀 V1.5 Enhanced Trading Engine initialized")
        else:
            self.quoter = AvellanedaStoikovQuoter(symbol)
            self.di_calc = None
            self.quote_manager = None
            self.logger.warning("🚀 V1-α Trading Engine initialized")
        
        # Trading state
        self.active_quotes: Dict[str, Quote] = {}
        self.current_mid = 0.0
        self.last_mid_price_latency_ms = 0.0  # Pour contrôle latence V1-α
        self.current_volatility = mm_config.sigma
        self.last_quote_time = 0.0
        self.quote_counter = 0
        
        # Risk flags
        self.trading_paused = False
        self.pause_reason = ""
        
        # Performance tracking
        self.total_quotes_sent = 0
        self.total_fills = 0
        
        # WebSocket integration for real-time data
        self.ws_integration = TradingEngineWSIntegration(self)
        # WebSocket integration initialized silently
        self.session_start = time.time()
        
        # Control flag for trading loop
        self.running = False
        
    async def run_trading_loop(self):
        """Main entry point to run the trading loop"""
        # Trading loop starting silently
        self.running = True
        
        try:
            # Initialize market data
            if not await self._initialize_market_data():
                self.logger.error("❌ Failed to initialize market data")
                return
            
            # Start WebSocket integration for real-time updates
            await self.ws_integration.start_integration()
            
            # Run main trading loop
            await self._main_trading_loop()
            
        except asyncio.CancelledError:
            self.logger.info("Trading loop cancelled")
        except Exception as e:
            self.logger.error(f"❌ Error in trading loop: {e}")
        finally:
            # Ensure we clean up on exit
            try:
                await self._cancel_all_quotes()
            except Exception as e:
                self.logger.warning(f"⚠️ Error cancelling quotes: {e}")
            
            # Stop WebSocket integration
            try:
                await self.ws_integration.stop_integration()
            except Exception as e:
                self.logger.warning(f"⚠️ Error stopping WebSocket integration: {e}")
            
            self.logger.info("Trading loop stopped")
    
    async def stop(self):
        """Gracefully stop the trading engine"""
        self.logger.info("🛑 Stopping trading engine...")
        self.running = False
        
        # Cancel all active quotes
        try:
            await self._cancel_all_quotes()
        except Exception as e:
            self.logger.warning(f"⚠️ Error cancelling quotes: {e}")
        
        # Stop WebSocket integration
        try:
            await self.ws_integration.stop_integration()
        except Exception as e:
            self.logger.warning(f"⚠️ Error stopping WebSocket integration: {e}")
        
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
            # Initialize empty book (WebSocket-only mode)
            if not self.local_book.initialize_empty_book():
                return False
            
            # Wait 2 seconds for WebSocket data to populate the book
            # Wait 2s for WebSocket data to populate book
            await asyncio.sleep(2.0)
            
            self.current_mid = self.local_book.get_mid_price()
            if not self.current_mid:
                self.logger.warning("⚠️ No mid price yet, will update when WebSocket data arrives")
                self.current_mid = 0  # Will be updated by WebSocket
                
            # Initialized silently
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Market data initialization failed: {e}")
            return False
    
    async def _main_trading_loop(self):
        """Main trading loop implementing §3.5"""
        # Main trading loop starting
        
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
        """Update market data from local book (now with real WebSocket data!)"""
        # Get real-time mid price from local book (updated by WebSocket)
        new_mid = self.local_book.get_mid_price()
        
        if new_mid is not None and new_mid > 0:
            # Update mid price with real market data
            self.current_mid = new_mid
            
            # Calculate real latency from WebSocket integration
            # For now, use a small simulated latency as placeholder
            self.last_mid_price_latency_ms = np.random.uniform(10, 50)  # 10-50ms
            
            # Update quoter's price history for volatility estimation
            self.quoter.update_volatility(self.current_mid)
            self.current_volatility = self.quoter.sigma
            
            # Update KPI tracker with current mid price for mark-to-market PnL
            self.kpi_tracker.update_mid_price(self.current_mid)
            
            # CORRIGÉ: Mettre à jour aussi l'InventoryController avec le prix mid
            self.inventory_ctrl.update_mid_price(self.current_mid)
            
            # V1.5: Update Depth Imbalance if available
            if self.version == "V1.5" and self.di_calc:
                # Get L1-L5 depth data from local book
                bid_depth, ask_depth = self.local_book.get_depth_l1_l5_both()
                self.di_calc.update_depth(bid_depth, ask_depth)
    
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
        
        # 3. Latency checks (§3.2 V1-α)
        # Mid-price latency check retiré suite à la directive produit
        
        # 3b. Quote latency ≤300ms (kill-switch)
        avg_latency = self.kpi_tracker.get_average_latency('quote_ack')
        if avg_latency > mm_config.max_quote_latency_ms:
            return True, (
                f"Quote latency: {avg_latency:.1f}ms "
                f"> {mm_config.max_quote_latency_ms}ms"
            )
        
        return False, ""
    
    async def _compute_quotes(self) -> Optional[Dict]:
        """Compute optimal quotes using version-specific logic"""
        if self.current_mid <= 0:
            return None
        
        try:
            # Get current signals
            current_ofi = self.ofi_calc.current_ofi()
            current_inventory = self.inventory_ctrl.current_inventory
            
            # Version-specific quote computation
            if self.version == "V1.5":
                quotes = await self._compute_quotes_v15(current_ofi, current_inventory)
            else:
                quotes = await self._compute_quotes_v1_alpha(current_ofi, current_inventory)
            
            if not quotes:
                return None
            
            # Apply inventory skew (common to both versions)
            skewed_quotes = self.inventory_ctrl.apply_skew_to_quotes(quotes)
            
            # Version-specific validation
            if self.version == "V1.5":
                if not self.quoter.validate_quotes_v15(skewed_quotes):
                    self.logger.warning("⚠️  Invalid V1.5 quotes generated")
                    return None
            else:
                if not self.quoter.validate_quotes(skewed_quotes, self.current_mid):
                    self.logger.warning("⚠️  Invalid V1-α quotes generated")
                    return None
            
            return skewed_quotes
            
        except Exception as e:
            self.logger.error(f"❌ Error computing quotes: {e}")
            return None
    
    async def _compute_quotes_v1_alpha(self, ofi: float, inventory: float) -> Optional[Dict]:
        """Compute V1-α quotes (original logic)"""
        quotes = self.quoter.compute_quotes(
            mid_price=self.current_mid,
            inventory=inventory,
            ofi=ofi
        )
        return quotes
    
    async def _compute_quotes_v15(self, ofi: float, inventory: float) -> Optional[Dict]:
        """Compute V1.5 quotes with enhanced multi-signal approach"""
        # Get current DI signal
        current_di = self.di_calc.get_current_di() if self.di_calc else 0.0
        
        # Check if quote refresh is needed (ageing or signal change)
        if self.quote_manager:
            refresh_needed, reason = self.quote_manager.check_refresh_needed(ofi, current_di)
            if refresh_needed:
                self.logger.debug(f"🔄 Quote refresh triggered: {reason}")
        
        # Compute V1.5 quotes with multi-signal approach
        quotes = self.quoter.compute_quotes_v15(
            mid_price=self.current_mid,
            inventory=inventory,
            ofi=ofi,
            di=current_di
        )
        
        return quotes
    
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
            
            # Cancel existing quotes first (normal update, don't count as cancellations)
            await self._cancel_all_quotes(count_as_cancel=False)
            
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
            
            # V1.5: Add quotes to quote manager for ageing tracking
            if self.version == "V1.5" and self.quote_manager:
                current_ofi = quotes_data.get('ofi', 0.0)
                current_di = quotes_data.get('di', 0.0)
                
                self.quote_manager.add_quote(
                    bid_id, 'bid', bid_quote.price, bid_quote.size, current_ofi, current_di
                )
                self.quote_manager.add_quote(
                    ask_id, 'ask', ask_quote.price, ask_quote.size, current_ofi, current_di
                )
            
            self.total_quotes_sent += 2
            self.last_quote_time = current_time
            
            # Record quotes sent in KPI tracker
            self.kpi_tracker.record_quotes_sent(2)
            
            # Simulate quote acknowledgment latency
            ack_latency = np.random.normal(50, 20)  # 50ms ± 20ms
            self.kpi_tracker.record_latency('quote_ack', max(10, ack_latency))
            
            # Enhanced logging for V1.5
            if self.version == "V1.5":
                self.logger.debug(f"📊 V1.5 Quotes updated: "
                                 f"Bid ${bid_quote.price:.2f}@{bid_quote.size:.4f} | "
                                 f"Ask ${ask_quote.price:.2f}@{ask_quote.size:.4f} | "
                                 f"OFI: {quotes_data.get('ofi', 0):.3f} | "
                                 f"DI: {quotes_data.get('di', 0):.3f} | "
                                 f"Centre: ${quotes_data.get('centre_price', 0):.2f} | "
                                 f"Dynamic Spread: {quotes_data.get('spread_bps', 0):.1f}bps")
            else:
                self.logger.debug(f"📊 V1-α Quotes updated: "
                                 f"Bid ${bid_quote.price:.2f}@{bid_quote.size:.4f} | "
                                 f"Ask ${ask_quote.price:.2f}@{ask_quote.size:.4f} | "
                                 f"OFI: {quotes_data.get('ofi', 0):.3f} | "
                                 f"Shift: {quotes_data.get('center_shift', 0):.6f}")
            
        except Exception as e:
            self.logger.error(f"❌ Error updating quotes: {e}")
    
    async def _cancel_all_quotes(self, count_as_cancel: bool = True):
        """Cancel all active quotes"""
        if not self.active_quotes:
            return
        
        cancelled_count = 0
        for quote_id, quote in self.active_quotes.items():
            if not quote.cancelled:
                quote.cancelled = True
                cancelled_count += 1
        
        if cancelled_count > 0:
            if count_as_cancel:
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
            # Pour un spread typique de 5 bps (0.0005), distance_from_mid = 0.00025
            # On veut une probabilité raisonnable, par exemple 1% par tick pour des quotes compétitives
            fill_prob = max(0.005, 0.05 - distance_from_mid * 50)  # 5% base, réduit par distance
            
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
            # CORRIGÉ: Passer le prix mid actuel pour le calcul mark-to-market
            self.inventory_ctrl.update_inventory(inventory_change, fill.price, self.current_mid)
            
            # Record inventory for KPIs
            self.kpi_tracker.record_inventory(self.inventory_ctrl.current_inventory)
            
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
        """Get current trading status with version-specific information"""
        uptime = time.time() - self.session_start
        
        base_status = {
            'symbol': self.symbol,
            'version': self.version,
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
        
        # Add V1.5 specific status information
        if self.version == "V1.5":
            base_status.update({
                'current_di': self.di_calc.get_current_di() if self.di_calc else 0.0,
                'di_stats': self.di_calc.get_signal_stats() if self.di_calc else {},
                'quote_manager_stats': self.quote_manager.get_management_stats() if self.quote_manager else {},
                'v15_params': self.quoter.get_v15_params() if hasattr(self.quoter, 'get_v15_params') else {}
            })
        
        return base_status
    
    def print_status(self):
        """Print current status with version-specific details"""
        status = self.get_status()
        kpis = status['kpis']
        
        print(f"\n📊 {self.symbol} Trading Engine Status ({status['version']})")
        print("=" * 60)
        print(f"Status: {status['status']} | Uptime: {status['uptime_minutes']:.1f}min")
        if status['pause_reason']:
            print(f"Pause Reason: {status['pause_reason']}")
        print(f"Mid Price: ${status['current_mid']:.2f}")
        print(f"Inventory: {status['inventory']:+.4f}")
        print(f"Volatility: {status['current_volatility']:.4f}")
        print(f"OFI: {status['current_ofi']:+.3f}")
        
        # V1.5 specific signals
        if status['version'] == "V1.5":
            print(f"DI: {status['current_di']:+.3f}")
            
            # Quote manager stats
            qm_stats = status.get('quote_manager_stats', {})
            if qm_stats:
                print(f"Quote Ageing: {qm_stats.get('aged_out', 0)} aged out, "
                      f"{qm_stats.get('signal_refreshed', 0)} signal refreshed")
        
        # WebSocket integration stats
        ws_stats = self.ws_integration.get_integration_stats()
        if ws_stats:
            print(f"WebSocket Updates: {ws_stats.get('updates', 0)} received, "
                  f"{ws_stats.get('errors', 0)} errors "
                  f"({ws_stats.get('success_rate', 0):.1f}% success)")
        
        print(f"Quotes Sent: {status['total_quotes']} | Fills: {status['total_fills']}")
        print(f"Active Quotes: {status['active_quotes']}")
        
        print("\n📈 KPIs:")
        print(f"  Fill Ratio: {kpis.get('fill_ratio', 0):.2%}")
        print(f"  Cancel Ratio: {kpis.get('cancel_ratio', 0):.2%}")
        print(f"  Avg Spread Captured: {kpis.get('avg_spread_captured', 0):.4f}")
        print(f"  RMS Inventory: {kpis.get('rms_inventory', 0):.4f}")
        print(f"  Total PnL: ${kpis.get('total_pnl', 0):+.2f}")
        
        # V1.5 specific KPIs
        if status['version'] == "V1.5":
            di_stats = status.get('di_stats', {})
            if di_stats:
                print(f"\n🔬 V1.5 Signal Stats:")
                print(f"  DI Observations: {di_stats.get('n_observations', 0)}")
                print(f"  DI Mean: {di_stats.get('mean', 0):.4f}")
                print(f"  DI Std: {di_stats.get('std', 0):.4f}")
            
            v15_params = status.get('v15_params', {})
            if v15_params:
                print(f"\n⚙️  V1.5 Parameters:")
                print(f"  β_di: {v15_params.get('beta_di', 0):.3f}")
                print(f"  κ_inv: {v15_params.get('kappa_inv', 0):.3f}")
                print(f"  κ_vol: {v15_params.get('kappa_vol', 0):.3f}")
        
        print("=" * 60)


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
